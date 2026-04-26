"""Sweep batch size across seeds on the toy task.

Multi-seed (3 seeds) sweep of BATCH ∈ {8, 16, 32, 64}. Holds total optimizer
steps fixed — i.e. larger batches see strictly more samples — so this is a
"how much does batch size matter at fixed compute (in steps)?" test.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import OpenMythos, build_tiny_mla_config

VOCAB = 32
PROMPT_LEN = 6
SEQ_LEN = 12
STEPS = 200
LR = 3e-3
N_LOOPS = 3
EVAL_BATCH = 32
EVAL_BATCHES = 16
BATCH_GRID = [8, 16, 32, 64]
N_SEEDS = 3


def make_batch(rng, b):
    prompt = torch.randint(0, VOCAB, (b, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def train_one(batch: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = N_LOOPS
    model = OpenMythos(cfg)
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        x, y = make_batch(rng, batch)
        logits = model(x, n_loops=N_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    correct = total = 0
    losses = []
    eval_rng = torch.Generator().manual_seed(seed + 9999)
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = make_batch(eval_rng, EVAL_BATCH)
            logits = model(x, n_loops=N_LOOPS)
            losses.append(F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item())
            preds = logits.argmax(dim=-1)
            t = y[:, PROMPT_LEN - 1 : 2 * PROMPT_LEN - 1]
            p = preds[:, PROMPT_LEN - 1 : 2 * PROMPT_LEN - 1]
            correct += (p == t).sum().item()
            total += t.numel()
    return correct / max(1, total), sum(losses) / len(losses)


def main() -> None:
    print(f"[bs] sweeping batch_size={BATCH_GRID} N_SEEDS={N_SEEDS} STEPS={STEPS}")
    rows = []
    for b in BATCH_GRID:
        accs = []
        ces = []
        t0 = time.time()
        for s in range(N_SEEDS):
            a, ce = train_one(b, s)
            accs.append(a * 100)
            ces.append(ce)
        wall = time.time() - t0
        rows.append({
            "batch": b,
            "ce_mean": statistics.mean(ces),
            "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs),
            "samples_seen": b * STEPS,
            "wall": wall,
        })
        print(f"[bs] batch={b:>3}  ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
              f"acc={rows[-1]['acc_mean']:.2f}%  samples={b*STEPS:>5}  wall={wall:.1f}s")

    print("\n  batch  samples_seen  eval_ce (mean +/- std)    rev_acc% (mean +/- std)")
    for r in rows:
        print(f"  {r['batch']:>4}   {r['samples_seen']:>10}   "
              f"{r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    best = min(rows, key=lambda r: r["ce_mean"])
    worst = max(rows, key=lambda r: r["ce_mean"])
    delta = worst["ce_mean"] - best["ce_mean"]
    pooled = math.sqrt(best["ce_std"] ** 2 + worst["ce_std"] ** 2)
    z = delta / pooled if pooled > 0 else float("inf")
    print(f"\n[bs] best_batch={best['batch']}  worst_batch={worst['batch']}  "
          f"delta_ce={delta:+.4f}  pooled_std={pooled:.4f}  z={z:.2f}")

    uniform_ce = math.log(VOCAB)
    for r in rows:
        assert r["ce_mean"] < uniform_ce
    print("\n[bs] OK")


if __name__ == "__main__":
    main()
