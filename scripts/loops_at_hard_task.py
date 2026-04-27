"""How far does the n_loops advantage scale at a hard task?

task_difficulty_scaling showed at prompt_len=8 the n=4 vs n=1 acc
gap is +3.11pp AND n=4 collapses variance 43x. That made n=4 the
clear winner at n in {1, 4}. Question: does increasing further to
n=8 keep helping?

Test: prompt_len=8, n_loops in {1, 2, 4, 8}, STEPS=2000,
prelude=1, coda=2, 3 seeds.
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

LOOP_GRID = [1, 2, 4, 8]
N_SEEDS = 3
STEPS = 2000
PROMPT_LEN = 8
PRELUDE = 1
CODA = 2


def run(n_loops: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=PROMPT_LEN)
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
    print(f"[loops_hard] LOOPS={LOOP_GRID} STEPS={STEPS} SEEDS={N_SEEDS}  "
          f"prompt_len={PROMPT_LEN} prelude={PRELUDE} coda={CODA}")
    rows = []
    for n in LOOP_GRID:
        ces, accs = [], []
        t0 = time.time()
        for seed in range(N_SEEDS):
            a, ce = run(n, seed)
            ces.append(ce); accs.append(a)
        wall = time.time() - t0
        rows.append({
            "n_loops": n,
            "ce_mean": statistics.mean(ces), "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs), "acc_std": statistics.pstdev(accs),
            "wall": wall,
        })
        print(f"[loops_hard] n_loops={n}  "
              f"ce={rows[-1]['ce_mean']:.4f}+/-{rows[-1]['ce_std']:.4f}  "
              f"acc={rows[-1]['acc_mean']:.2f}%+/-{rows[-1]['acc_std']:.2f}  "
              f"wall={wall:.1f}s")

    print("\n  n_loops   eval_ce            acc%             wall")
    for r in rows:
        print(f"     {r['n_loops']:>2}      "
              f"{r['ce_mean']:.4f}+/-{r['ce_std']:.4f}  "
              f"{r['acc_mean']:5.2f}+/-{r['acc_std']:.2f}  "
              f"{r['wall']:6.1f}s")

    base = next(r for r in rows if r["n_loops"] == 1)
    print()
    for r in rows:
        if r["n_loops"] == 1:
            continue
        delta = r["ce_mean"] - base["ce_mean"]
        pooled = math.sqrt(r["ce_std"] ** 2 + base["ce_std"] ** 2)
        z = delta / pooled if pooled > 0 else float("inf")
        d_acc = r["acc_mean"] - base["acc_mean"]
        cost_ratio = r["wall"] / base["wall"]
        print(f"  n_loops={r['n_loops']} vs 1: delta_ce={delta:+.4f}  z={z:+.2f}  "
              f"delta_acc={d_acc:+.2f}pp  cost={cost_ratio:.2f}x")

    chance_ce = math.log(8)
    for r in rows:
        assert r["ce_mean"] < chance_ce
    print("\n[loops_hard] OK")


if __name__ == "__main__":
    main()
