"""Does n=8 keep dominating at prompt_len=14, or hit a ceiling?

harder_n8 found n=8 captures most of prompt_len=12's headroom
(82% -> 95.18% acc). Push to prompt_len=14: n=1 acc should drop
further. If n=8 still closes most of the gap, recurrence is a
genuine compute-allocation lever. If n=8 starts losing ground to
n=4 or stays well below n=1's saturated peak, we have found a
real ceiling tied to model capacity at dim=128.

dim=128 vocab=8 STEPS=2000 prompt_len=14 n in {1, 4, 8} 3 seeds.
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

LOOP_GRID = [1, 4, 8]
N_SEEDS = 3
STEPS = 2000
PROMPT_LEN = 14
DIM = 128


def run(n_loops: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=PROMPT_LEN)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = n_loops
    cfg.prelude_layers = 1
    cfg.coda_layers = 2
    cfg.dim = DIM
    cfg.expert_dim = DIM // 2
    model = OpenMythos(cfg)
    train_reverse_copy(model, task, n_loops=n_loops, steps=STEPS, seed=seed)
    ce, acc = evaluate(model, task, n_loops=n_loops, seed=seed + 9999)
    return acc * 100, ce


def main() -> None:
    print(f"[harder_pl14] dim={DIM} prompt_len={PROMPT_LEN} STEPS={STEPS} "
          f"LOOPS={LOOP_GRID} SEEDS={N_SEEDS}")
    grid = {}
    for n in LOOP_GRID:
        ces, accs = [], []
        t0 = time.time()
        for seed in range(N_SEEDS):
            a, ce = run(n, seed)
            ces.append(ce); accs.append(a)
        grid[n] = {
            "ce_mean": statistics.mean(ces),
            "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs),
            "wall": time.time() - t0,
        }
        r = grid[n]
        print(f"[harder_pl14] n={n}  ce={r['ce_mean']:.4f}+/-{r['ce_std']:.4f}  "
              f"acc={r['acc_mean']:.2f}%+/-{r['acc_std']:.2f}  wall={r['wall']:.1f}s")

    print("\n  n>1 advantage vs n=1:")
    base = grid[1]
    for n in LOOP_GRID:
        if n == 1:
            continue
        r = grid[n]
        delta = r["ce_mean"] - base["ce_mean"]
        pooled = math.sqrt(r["ce_std"] ** 2 + base["ce_std"] ** 2)
        z = delta / pooled if pooled > 0 else float("inf")
        d_acc = r["acc_mean"] - base["acc_mean"]
        print(f"    n={n}: delta_ce={delta:+.4f}  z={z:+.2f}  delta_acc={d_acc:+.2f}pp")

    chance_ce = math.log(8)
    for r in grid.values():
        assert r["ce_mean"] < chance_ce
    print("\n[harder_pl14] OK")


if __name__ == "__main__":
    main()
