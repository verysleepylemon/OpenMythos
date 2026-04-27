"""Task-difficulty x recurrent-depth scaling at optimal architecture.

convergence_curve found a Goldilocks zone for n_loops at the easy
ReverseCopy task (vocab=8, prompt_len=4). Question: does the
benefit of recurrent depth GROW as the task gets harder?

If recurrent depth is a real architectural property, harder tasks
should reward more loops (more compute needed per token, more
benefit from re-iterating). If loops are just a regularizer, the
benefit should stay constant or shrink with task difficulty.

Test: prompt_len in {4, 6, 8} (seq_len = 2*prompt_len, so
8/12/16 tokens). Fixed n_loops in {1, 4}, fixed STEPS=2000
(Goldilocks band per convergence_curve), prelude=1, coda=2,
3 seeds.
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

PROMPT_LENS = [4, 6, 8]
LOOP_GRID = [1, 4]
N_SEEDS = 3
STEPS = 2000
PRELUDE = 1
CODA = 2


def run(prompt_len: int, n_loops: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=prompt_len)
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
    print(f"[task_diff] PROMPT_LENS={PROMPT_LENS} LOOPS={LOOP_GRID} "
          f"STEPS={STEPS} SEEDS={N_SEEDS}  prelude={PRELUDE} coda={CODA}")

    grid = {}
    for pl in PROMPT_LENS:
        for n in LOOP_GRID:
            ces, accs = [], []
            t0 = time.time()
            for seed in range(N_SEEDS):
                a, ce = run(pl, n, seed)
                ces.append(ce); accs.append(a)
            grid[(pl, n)] = {
                "ce_mean": statistics.mean(ces),
                "ce_std": statistics.pstdev(ces),
                "acc_mean": statistics.mean(accs),
                "acc_std": statistics.pstdev(accs),
                "wall": time.time() - t0,
            }
            r = grid[(pl, n)]
            print(f"[task_diff] prompt_len={pl} n={n}  "
                  f"ce={r['ce_mean']:.4f}+/-{r['ce_std']:.4f}  "
                  f"acc={r['acc_mean']:.2f}%+/-{r['acc_std']:.2f}  "
                  f"wall={r['wall']:.1f}s")

    print("\n  Summary table:")
    print("  prompt_len  n_loops   eval_ce            acc%")
    for pl in PROMPT_LENS:
        for n in LOOP_GRID:
            r = grid[(pl, n)]
            print(f"     {pl:>4}        {n}      "
                  f"{r['ce_mean']:.4f}+/-{r['ce_std']:.4f}  "
                  f"{r['acc_mean']:5.2f}+/-{r['acc_std']:.2f}")

    print("\n  n_loops=4 vs n_loops=1 advantage by prompt_len:")
    for pl in PROMPT_LENS:
        r1 = grid[(pl, 1)]; r4 = grid[(pl, 4)]
        delta = r4["ce_mean"] - r1["ce_mean"]
        pooled = math.sqrt(r4["ce_std"] ** 2 + r1["ce_std"] ** 2)
        z = delta / pooled if pooled > 0 else float("inf")
        d_acc = r4["acc_mean"] - r1["acc_mean"]
        print(f"    prompt_len={pl}: delta_ce={delta:+.4f}  z={z:+.2f}  "
              f"delta_acc={d_acc:+.2f}pp")

    chance_ce = math.log(8)
    for r in grid.values():
        assert r["ce_mean"] < chance_ce
    print("\n[task_diff] OK")


if __name__ == "__main__":
    main()
