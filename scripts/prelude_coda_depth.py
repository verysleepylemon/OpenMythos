"""Prelude/coda depth sweep.

`recurrent_vs_stacked` showed extra coda layers help (K=4 stacked
beat K=4 recurrent by 0.09 ce). This script asks where the marginal
coda layer stops paying, and whether prelude has the same property.

We sweep:
  prelude in {1, 2}, coda in {1, 2, 3} at fixed n_loops=2

Goal: identify the cheapest config that's not significantly worse
than the largest config (cheapest-not-worse strategy).
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

PRELUDE_GRID = [1, 2]
CODA_GRID = [1, 2, 3]
N_SEEDS = 3
STEPS = 200
N_LOOPS = 2


def run(prelude: int, coda: int, seed: int) -> tuple[float, float, int]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=4)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = N_LOOPS
    cfg.prelude_layers = prelude
    cfg.coda_layers = coda
    model = OpenMythos(cfg)
    train_reverse_copy(model, task, n_loops=N_LOOPS, steps=STEPS, seed=seed)
    ce, acc = evaluate(model, task, n_loops=N_LOOPS, seed=seed + 9999)
    n_params = sum(p.numel() for p in model.parameters())
    return acc * 100, ce, n_params


def main() -> None:
    print(f"[pc_depth] PRELUDE_GRID={PRELUDE_GRID} CODA_GRID={CODA_GRID} N_SEEDS={N_SEEDS} STEPS={STEPS}")
    rows = []
    for p in PRELUDE_GRID:
        for c in CODA_GRID:
            accs, ces, params = [], [], None
            t0 = time.time()
            for s in range(N_SEEDS):
                a, ce, params = run(p, c, s)
                accs.append(a)
                ces.append(ce)
            wall = time.time() - t0
            rows.append({
                "prelude": p, "coda": c, "params": params,
                "ce_mean": statistics.mean(ces), "ce_std": statistics.pstdev(ces),
                "acc_mean": statistics.mean(accs), "acc_std": statistics.pstdev(accs),
                "wall": wall,
            })
            print(f"[pc_depth] prelude={p} coda={c}  "
                  f"params={params:>6d}  "
                  f"ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
                  f"acc={rows[-1]['acc_mean']:.2f}%  wall={wall:.1f}s")

    print("\n  prelude  coda  params   eval_ce (mean +/- std)    acc% (mean +/- std)")
    for r in rows:
        print(f"     {r['prelude']}      {r['coda']}   {r['params']:>6d}   "
              f"{r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    best = min(rows, key=lambda r: r["ce_mean"])
    smallest = min(rows, key=lambda r: r["params"])
    print(f"\n  best: prelude={best['prelude']} coda={best['coda']}  "
          f"params={best['params']}  ce={best['ce_mean']:.4f}")
    print(f"  smallest: prelude={smallest['prelude']} coda={smallest['coda']}  "
          f"params={smallest['params']}  ce={smallest['ce_mean']:.4f}")
    delta = smallest["ce_mean"] - best["ce_mean"]
    pooled = math.sqrt(smallest["ce_std"] ** 2 + best["ce_std"] ** 2)
    z = delta / pooled if pooled > 0 else float("inf")
    print(f"  smallest vs best: delta={delta:+.4f}  z={z:+.2f}")

    chance_ce = math.log(8)
    for r in rows:
        assert r["ce_mean"] < chance_ce
    print("\n[pc_depth] OK")


if __name__ == "__main__":
    main()
