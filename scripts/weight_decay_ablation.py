"""Sweep AdamW weight_decay across seeds on the toy task.

Multi-seed (3 seeds) sweep of weight_decay ∈ {0.0, 0.01, 0.1} to see whether
regularization moves the needle on this tiny model. Reports mean +/- std
eval CE per setting.
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
BATCH = 32
STEPS = 200
LR = 3e-3
N_LOOPS = 3
EVAL_BATCHES = 16
WD_GRID = [0.0, 0.01, 0.1]
N_SEEDS = 3


def make_batch(rng):
    prompt = torch.randint(0, VOCAB, (BATCH, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def train_one(wd: float, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = N_LOOPS
    model = OpenMythos(cfg)
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)
    model.train()
    for _ in range(STEPS):
        x, y = make_batch(rng)
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
            x, y = make_batch(eval_rng)
            logits = model(x, n_loops=N_LOOPS)
            losses.append(F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item())
            preds = logits.argmax(dim=-1)
            t = y[:, PROMPT_LEN - 1 : 2 * PROMPT_LEN - 1]
            p = preds[:, PROMPT_LEN - 1 : 2 * PROMPT_LEN - 1]
            correct += (p == t).sum().item()
            total += t.numel()
    return correct / max(1, total), sum(losses) / len(losses)


def main() -> None:
    print(f"[wd] sweeping weight_decay={WD_GRID} N_SEEDS={N_SEEDS}")
    rows = []
    for wd in WD_GRID:
        accs = []
        ces = []
        t0 = time.time()
        for s in range(N_SEEDS):
            a, c = train_one(wd, s)
            accs.append(a * 100)
            ces.append(c)
        wall = time.time() - t0
        rows.append({
            "wd": wd,
            "ce_mean": statistics.mean(ces),
            "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs),
            "wall": wall,
        })
        print(f"[wd] wd={wd}  ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
              f"acc={rows[-1]['acc_mean']:.2f} +/- {rows[-1]['acc_std']:.2f}%  wall={wall:.1f}s")

    print("\n  weight_decay  eval_ce (mean +/- std)    rev_acc% (mean +/- std)")
    for r in rows:
        print(f"  {r['wd']:>10.3f}    {r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    best = min(rows, key=lambda r: r["ce_mean"])
    base = next(r for r in rows if r["wd"] == 0.0)
    delta = base["ce_mean"] - best["ce_mean"]
    pooled = math.sqrt(base["ce_std"] ** 2 + best["ce_std"] ** 2)
    z = delta / pooled if pooled > 0 else float("inf")
    print(f"\n[wd] best wd={best['wd']}  delta_vs_zero={delta:+.4f}  pooled_std={pooled:.4f}  z={z:.2f}")

    uniform_ce = math.log(VOCAB)
    for r in rows:
        assert r["ce_mean"] < uniform_ce
    print("\n[wd] OK")


if __name__ == "__main__":
    main()
