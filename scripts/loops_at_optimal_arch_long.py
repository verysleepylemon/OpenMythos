"""Extreme-convergence loop sweep at the optimal architecture.

loops_at_optimal_arch (STEPS=1000) showed for the first time:
  - monotonic ce improvement with n_loops (0.94 -> 0.92 -> 0.91)
  - monotonic stddev collapse (0.041 -> 0.019 -> 0.014)
  - mean-ce gap still sub-sigma (z=-0.46, z=-0.71)

Question: at the optimal arch, if we push training MUCH longer
(STEPS=2500), does the monotonic mean-ce trend open up enough to
clear the noise floor? Or does the variance reduction stay
constant while the mean gap stays small?

3 seeds x 2500 steps x 3 loop counts.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import OpenMythos, build_tiny_mla_config
from scripts._eval import ReverseCopyTask, evaluate, train_reverse_copy

LOOP_GRID = [1, 2, 4]
N_SEEDS = 3
STEPS = 2500
PRELUDE = 1
CODA = 2


def run(n_loops: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=4)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = n_loops
    cfg.prelude_layers = PRELUDE
    cfg.coda_layers = CODA
    model = OpenMythos(cfg)
    train_reverse_copy(model, task, n_loops=n_loops, steps=STEPS, seed=seed)
    ce, acc = evaluate(model, task, n_loops=n_loops, seed=seed + 9999)
    return acc * 100, ce


def main() -> None:
    print(f"[loops_opt_long] LOOP_GRID={LOOP_GRID} N_SEEDS={N_SEEDS} STEPS={STEPS}  "
          f"prelude={PRELUDE} coda={CODA}")
    rows = []
    for n in LOOP_GRID:
        accs, ces = [], []
        t0 = time.time()
        for s in range(N_SEEDS):
            a, ce = run(n, s)
            accs.append(a)
            ces.append(ce)
        wall = time.time() - t0
        rows.append({
            "n_loops": n,
            "ce_mean": statistics.mean(ces), "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs), "acc_std": statistics.pstdev(accs),
            "wall": wall,
        })
        print(f"[loops_opt_long] n_loops={n}  "
              f"ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
              f"acc={rows[-1]['acc_mean']:.2f}% +/- {rows[-1]['acc_std']:.2f}  "
              f"wall={wall:.1f}s")

    print("\n  n_loops   eval_ce (mean +/- std)    acc% (mean +/- std)")
    for r in rows:
        print(f"     {r['n_loops']}      "
              f"{r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    base = next(r for r in rows if r["n_loops"] == 1)
    print()
    for r in rows:
        if r["n_loops"] == 1:
            continue
        delta = r["ce_mean"] - base["ce_mean"]
        pooled = math.sqrt(r["ce_std"] ** 2 + base["ce_std"] ** 2)
        z = delta / pooled if pooled > 0 else float("inf")
        print(f"  n_loops={r['n_loops']} vs 1: delta_ce={delta:+.4f}  z={z:+.2f}")

    chance_ce = math.log(8)
    for r in rows:
        assert r["ce_mean"] < chance_ce
    print("\n[loops_opt_long] OK")


if __name__ == "__main__":
    main()
