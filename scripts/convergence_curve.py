"""Convergence-curve sweep at the optimal architecture.

loops_at_optimal_arch (STEPS=1000) showed n_loops=4 wins (acc 99.1
vs 97.7) AND collapses variance 3x.
loops_at_optimal_arch_long (STEPS=2500) showed all loop counts
converge to identical ce.

So: where does the loop benefit fade? This sweep measures the
convergence curve at several STEPS values for each n_loops.

3 seeds x {500, 1000, 2000} steps x {1, 2, 4} loops.
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
STEPS_GRID = [500, 1000, 2000]
N_SEEDS = 3
PRELUDE = 1
CODA = 2


def run(n_loops: int, steps: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=4)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = n_loops
    cfg.prelude_layers = PRELUDE
    cfg.coda_layers = CODA
    model = OpenMythos(cfg)
    train_reverse_copy(model, task, n_loops=n_loops, steps=steps, seed=seed)
    ce, acc = evaluate(model, task, n_loops=n_loops, seed=seed + 9999)
    return acc * 100, ce


def main() -> None:
    print(f"[conv_curve] LOOPS={LOOP_GRID} STEPS={STEPS_GRID} SEEDS={N_SEEDS}  "
          f"prelude={PRELUDE} coda={CODA}")

    grid = {}  # (n_loops, steps) -> {ce_mean, ce_std, acc_mean, acc_std}
    for n in LOOP_GRID:
        for s_steps in STEPS_GRID:
            ces, accs = [], []
            t0 = time.time()
            for seed in range(N_SEEDS):
                a, ce = run(n, s_steps, seed)
                ces.append(ce); accs.append(a)
            grid[(n, s_steps)] = {
                "ce_mean": statistics.mean(ces),
                "ce_std": statistics.pstdev(ces),
                "acc_mean": statistics.mean(accs),
                "acc_std": statistics.pstdev(accs),
                "wall": time.time() - t0,
            }
            r = grid[(n, s_steps)]
            print(f"[conv_curve] n={n} steps={s_steps:4d}  "
                  f"ce={r['ce_mean']:.4f}+/-{r['ce_std']:.4f}  "
                  f"acc={r['acc_mean']:.2f}%+/-{r['acc_std']:.2f}  "
                  f"wall={r['wall']:.1f}s")

    print("\n  CE matrix (rows=steps, cols=n_loops):")
    print("  steps   " + "  ".join(f"n={n:>3}        " for n in LOOP_GRID))
    for s_steps in STEPS_GRID:
        line = f"  {s_steps:5d}   "
        for n in LOOP_GRID:
            r = grid[(n, s_steps)]
            line += f"{r['ce_mean']:.4f}+/-{r['ce_std']:.4f}  "
        print(line)

    print("\n  ACC matrix (rows=steps, cols=n_loops):")
    print("  steps   " + "  ".join(f"n={n:>3}        " for n in LOOP_GRID))
    for s_steps in STEPS_GRID:
        line = f"  {s_steps:5d}   "
        for n in LOOP_GRID:
            r = grid[(n, s_steps)]
            line += f"{r['acc_mean']:5.2f}+/-{r['acc_std']:.2f}  "
        print(line)

    # Compare n=4 vs n=1 at each steps point
    print("\n  n_loops=4 vs n_loops=1 advantage by steps:")
    for s_steps in STEPS_GRID:
        r1 = grid[(1, s_steps)]; r4 = grid[(4, s_steps)]
        delta = r4["ce_mean"] - r1["ce_mean"]
        pooled = math.sqrt(r4["ce_std"] ** 2 + r1["ce_std"] ** 2)
        z = delta / pooled if pooled > 0 else float("inf")
        d_acc = r4["acc_mean"] - r1["acc_mean"]
        print(f"    steps={s_steps:4d}: delta_ce={delta:+.4f}  z={z:+.2f}  "
              f"delta_acc={d_acc:+.2f}pp")

    chance_ce = math.log(8)
    for r in grid.values():
        assert r["ce_mean"] < chance_ce
    print("\n[conv_curve] OK")


if __name__ == "__main__":
    main()
