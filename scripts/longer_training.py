"""Longer-training probe.

Most ablations on this branch use STEPS=200, where the model
plateaus around ce~1.68 on reverse-copy. Three questions:

1. Does longer training (1000 steps) actually keep improving ce?
2. Does the gap between n_loops={1, 2, 4} that was below noise at
   200 steps OPEN UP with more compute?
3. Where does the model actually stop learning?

3 seeds per (n_loops, steps) cell.
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
STEP_GRID = [200, 1000]
N_SEEDS = 3


def run(n_loops: int, steps: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=4)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = max(LOOP_GRID)
    model = OpenMythos(cfg)
    train_reverse_copy(model, task, n_loops=n_loops, steps=steps, seed=seed)
    ce, acc = evaluate(model, task, n_loops=n_loops, seed=seed + 9999)
    return acc * 100, ce


def main() -> None:
    print(f"[longer] LOOP_GRID={LOOP_GRID} STEP_GRID={STEP_GRID} N_SEEDS={N_SEEDS}")
    rows = []
    for steps in STEP_GRID:
        for n in LOOP_GRID:
            accs, ces = [], []
            t0 = time.time()
            for s in range(N_SEEDS):
                a, c = run(n, steps, s)
                accs.append(a)
                ces.append(c)
            wall = time.time() - t0
            rows.append({
                "steps": steps,
                "n": n,
                "ce_mean": statistics.mean(ces),
                "ce_std": statistics.pstdev(ces),
                "acc_mean": statistics.mean(accs),
                "acc_std": statistics.pstdev(accs),
                "wall": wall,
            })
            print(f"[longer] steps={steps:>4} n_loops={n}  "
                  f"ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
                  f"acc={rows[-1]['acc_mean']:.2f}%  wall={wall:.1f}s")

    print("\n  steps  n_loops  eval_ce (mean +/- std)    acc% (mean +/- std)")
    for r in rows:
        print(f"  {r['steps']:>4}    {r['n']}     "
              f"{r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    # For each n_loops, did 1000 vs 200 reduce ce significantly?
    print("\n  Per-n_loops short -> long delta:")
    for n in LOOP_GRID:
        short = next(r for r in rows if r["steps"] == 200 and r["n"] == n)
        long = next(r for r in rows if r["steps"] == 1000 and r["n"] == n)
        delta = short["ce_mean"] - long["ce_mean"]
        pooled = math.sqrt(short["ce_std"] ** 2 + long["ce_std"] ** 2)
        z = delta / pooled if pooled > 0 else float("inf")
        print(f"    n_loops={n}: ce {short['ce_mean']:.4f} -> {long['ce_mean']:.4f}  "
              f"delta={delta:+.4f}  z={z:.2f}")

    # At 1000 steps, does n_loops gap open up?
    print("\n  At 1000 steps, n_loops gap vs n=1 baseline:")
    n1_long = next(r for r in rows if r["steps"] == 1000 and r["n"] == 1)
    for n in LOOP_GRID:
        if n == 1:
            continue
        r = next(r for r in rows if r["steps"] == 1000 and r["n"] == n)
        delta = n1_long["ce_mean"] - r["ce_mean"]
        pooled = math.sqrt(n1_long["ce_std"] ** 2 + r["ce_std"] ** 2)
        z = delta / pooled if pooled > 0 else float("inf")
        print(f"    n_loops={n}: delta_vs_n1={delta:+.4f}  z={z:.2f}")

    chance_ce = math.log(8)
    for r in rows:
        assert r["ce_mean"] < chance_ce
    print("\n[longer] OK")


if __name__ == "__main__":
    main()
