"""Does the n_loops sweet spot shift when the model gets bigger?

loops_at_hard_task found at prompt_len=8 the optimum is n=2 (n=4
ties, n=8 hurts). Open question: is n=2 still optimal as model
capacity grows? Two competing hypotheses:

  H1 (recurrence-as-depth-amplifier): bigger models still benefit
  from extra loops because each loop adds 'effective depth.'

  H2 (recurrence-as-capacity-substitute): bigger models need fewer
  loops because they have enough capacity per pass.

Test: dim in {64, 128} x n_loops in {1, 2, 4} at prompt_len=8,
STEPS=2000, prelude=1, coda=2, 3 seeds. Scale expert_dim in
proportion (32 vs 64).
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

DIM_GRID = [64, 128]
LOOP_GRID = [1, 2, 4]
N_SEEDS = 3
STEPS = 2000
PROMPT_LEN = 8
PRELUDE = 1
CODA = 2


def _params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(dim: int, n_loops: int, seed: int) -> tuple[float, float, int]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=PROMPT_LEN)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = n_loops
    cfg.prelude_layers = PRELUDE
    cfg.coda_layers = CODA
    cfg.dim = dim
    cfg.expert_dim = dim // 2
    model = OpenMythos(cfg)
    n_params = _params(model)
    train_reverse_copy(model, task, n_loops=n_loops, steps=STEPS, seed=seed)
    ce, acc = evaluate(model, task, n_loops=n_loops, seed=seed + 9999)
    return acc * 100, ce, n_params


def main() -> None:
    print(f"[dim_x_loops] DIM={DIM_GRID} LOOPS={LOOP_GRID} STEPS={STEPS} "
          f"SEEDS={N_SEEDS}  prompt_len={PROMPT_LEN} prelude={PRELUDE} coda={CODA}")

    grid = {}
    for dim in DIM_GRID:
        for n in LOOP_GRID:
            ces, accs = [], []
            params = None
            t0 = time.time()
            for seed in range(N_SEEDS):
                a, ce, p = run(dim, n, seed)
                ces.append(ce); accs.append(a); params = p
            grid[(dim, n)] = {
                "ce_mean": statistics.mean(ces),
                "ce_std": statistics.pstdev(ces),
                "acc_mean": statistics.mean(accs),
                "acc_std": statistics.pstdev(accs),
                "params": params,
                "wall": time.time() - t0,
            }
            r = grid[(dim, n)]
            print(f"[dim_x_loops] dim={dim} n={n} params={r['params']/1000:.1f}k  "
                  f"ce={r['ce_mean']:.4f}+/-{r['ce_std']:.4f}  "
                  f"acc={r['acc_mean']:.2f}%+/-{r['acc_std']:.2f}  "
                  f"wall={r['wall']:.1f}s")

    print("\n  dim   n_loops  params      eval_ce            acc%")
    for dim in DIM_GRID:
        for n in LOOP_GRID:
            r = grid[(dim, n)]
            print(f"  {dim:>3}      {n}    {r['params']/1000:6.1f}k  "
                  f"{r['ce_mean']:.4f}+/-{r['ce_std']:.4f}  "
                  f"{r['acc_mean']:5.2f}+/-{r['acc_std']:.2f}")

    print("\n  Per-dim n_loops>1 advantage vs n_loops=1:")
    for dim in DIM_GRID:
        base = grid[(dim, 1)]
        for n in LOOP_GRID:
            if n == 1:
                continue
            r = grid[(dim, n)]
            delta = r["ce_mean"] - base["ce_mean"]
            pooled = math.sqrt(r["ce_std"] ** 2 + base["ce_std"] ** 2)
            z = delta / pooled if pooled > 0 else float("inf")
            d_acc = r["acc_mean"] - base["acc_mean"]
            print(f"    dim={dim} n={n} vs n=1: delta_ce={delta:+.4f}  "
                  f"z={z:+.2f}  delta_acc={d_acc:+.2f}pp")

    chance_ce = math.log(8)
    for r in grid.values():
        assert r["ce_mean"] < chance_ce
    print("\n[dim_x_loops] OK")


if __name__ == "__main__":
    main()
