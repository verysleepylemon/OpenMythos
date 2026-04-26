"""Sweep seeds × n_loops to report mean +/- std of the toy task eval loss.

loops_grad_study.py reports a single training run per n_loops, which is OK
for spotting trends but vulnerable to seed noise. This script trains
N_SEEDS independent models per n_loops setting and reports mean and stdev
across seeds. The point is to confirm that the n=4 win observed in
loops_grad_study.py is real, not seed-noise.
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
LOOP_GRID = [1, 2, 4, 8]
N_SEEDS = 4
EVAL_BATCHES = 16


def make_batch(rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = torch.randint(0, VOCAB, (BATCH, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def eval_model(model: OpenMythos, n_loops: int, rng: torch.Generator) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    losses = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = make_batch(rng)
            logits = model(x, n_loops=n_loops)
            losses.append(F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item())
            preds = logits.argmax(dim=-1)
            t = y[:, PROMPT_LEN - 1 : 2 * PROMPT_LEN - 1]
            p = preds[:, PROMPT_LEN - 1 : 2 * PROMPT_LEN - 1]
            correct += (p == t).sum().item()
            total += t.numel()
    return correct / max(1, total), sum(losses) / len(losses)


def train_one(n_loops: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = max(LOOP_GRID)  # support all settings
    model = OpenMythos(cfg)
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        x, y = make_batch(rng)
        logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    eval_rng = torch.Generator().manual_seed(seed + 9999)
    return eval_model(model, n_loops, eval_rng)


def main() -> None:
    print(f"[seed] sweeping LOOP_GRID={LOOP_GRID} N_SEEDS={N_SEEDS} STEPS={STEPS}")
    rows = []
    for n_loops in LOOP_GRID:
        accs: list[float] = []
        ces: list[float] = []
        t0 = time.time()
        for s in range(N_SEEDS):
            acc, ce = train_one(n_loops, seed=s)
            accs.append(acc * 100)
            ces.append(ce)
        wall = time.time() - t0
        rows.append({
            "n_loops": n_loops,
            "ce_mean": statistics.mean(ces),
            "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs),
            "wall_s": wall,
            "ces": ces,
            "accs": accs,
        })
        print(
            f"[seed] n_loops={n_loops}  "
            f"ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
            f"acc={rows[-1]['acc_mean']:.2f} +/- {rows[-1]['acc_std']:.2f}%  "
            f"wall={wall:.1f}s"
        )

    print("\n n_loops    eval_ce (mean +/- std)        rev_acc% (mean +/- std)")
    for r in rows:
        print(f"   {r['n_loops']:>3}      {r['ce_mean']:.4f} +/- {r['ce_std']:.4f}        "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    best = min(rows, key=lambda r: r["ce_mean"])
    base = next(r for r in rows if r["n_loops"] == 1)
    diff = base["ce_mean"] - best["ce_mean"]
    pooled = math.sqrt(base["ce_std"] ** 2 + best["ce_std"] ** 2)
    z = diff / pooled if pooled > 0 else float("inf")
    print(f"\n[seed] best n_loops={best['n_loops']}  ce={best['ce_mean']:.4f}")
    print(f"[seed] vs n=1 baseline: delta={diff:+.4f}  pooled_std={pooled:.4f}  z={z:.2f}")

    uniform_ce = math.log(VOCAB)
    for r in rows:
        assert r["ce_mean"] < uniform_ce, (
            f"n_loops={r['n_loops']} failed to learn: ce={r['ce_mean']:.4f}"
        )
    print("\n[seed] OK")


if __name__ == "__main__":
    main()
