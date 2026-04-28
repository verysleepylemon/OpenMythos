"""pl_sweep_n4.py - map the SHAPE of the surviving recurrence-win region.

After replicate_pl8_n4 confirmed the only rock-solid (dim, pl, n) point
on the entire branch (dim=128 pl=8 n=4: 99.81+/-0.17 vs n=1 98.41+/-0.85,
+1.40pp, 5x variance collapse, 6 seeds), the next question is whether
the win is a SINGLE POINT or a REGION.

This script sweeps prompt_len in {6, 8, 10} at dim=128 with n in {1, 4}
and 6 seeds per cell. 6 cells x 6 seeds = 36 runs.

Outcomes:
  - Win extends to pl=6 and pl=10  -> ship "n_loops=4 at dim=128 pl in [6,10]".
  - Win is pl=8 only                -> ship "n_loops=4 at dim=128 pl=8 only".
  - Win disappears at pl=8 too      -> the only confirmed point was lucky after all.

All other settings match replicate_pl8_n4 (prelude=1 coda=2 STEPS=2000 vocab=8).
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

PL_GRID = [6, 8, 10]
LOOP_GRID = [1, 4]
N_SEEDS = 6
STEPS = 2000
DIM = 128


def run(prompt_len, n_loops, seed):
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=prompt_len)
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


def main():
    print(
        f"[pl_sweep_n4] dim={DIM} STEPS={STEPS} "
        f"pl_grid={PL_GRID} loops={LOOP_GRID} seeds={N_SEEDS}"
    )
    cells = {}
    t_total = time.time()
    for pl in PL_GRID:
        for n in LOOP_GRID:
            ces, accs = [], []
            t0 = time.time()
            for seed in range(N_SEEDS):
                a, ce = run(pl, n, seed)
                ces.append(ce)
                accs.append(a)
                print(
                    f"[pl_sweep_n4] pl={pl} n={n} seed={seed} "
                    f"ce={ce:.4f} acc={a:.2f}%"
                )
            cells[(pl, n)] = {
                "ce_mean": statistics.fmean(ces),
                "ce_std": statistics.stdev(ces) if len(ces) > 1 else 0.0,
                "acc_mean": statistics.fmean(accs),
                "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                "wall": time.time() - t0,
            }
            r = cells[(pl, n)]
            print(
                f"[pl_sweep_n4] pl={pl} n={n} SUMMARY "
                f"ce={r['ce_mean']:.4f}+/-{r['ce_std']:.4f} "
                f"acc={r['acc_mean']:.2f}+/-{r['acc_std']:.2f} wall={r['wall']:.0f}s"
            )

    print()
    print(f"  Region map (delta from n=1 to n=4 at each pl, {N_SEEDS} seeds):")
    region_summary = []
    for pl in PL_GRID:
        base = cells[(pl, 1)]
        cmp_ = cells[(pl, 4)]
        d_ce = cmp_["ce_mean"] - base["ce_mean"]
        d_acc = cmp_["acc_mean"] - base["acc_mean"]
        d_acc_std = cmp_["acc_std"] - base["acc_std"]
        pooled = math.sqrt(cmp_["ce_std"] ** 2 + base["ce_std"] ** 2)
        z = d_ce / pooled if pooled > 0 else 0.0
        verdict = "<- helps" if d_acc > 0 else "<- hurts"
        line = (
            f"    pl={pl}: n=1 acc={base['acc_mean']:.2f}+/-{base['acc_std']:.2f} "
            f"n=4 acc={cmp_['acc_mean']:.2f}+/-{cmp_['acc_std']:.2f} "
            f"d_acc={d_acc:+.2f}pp d_std={d_acc_std:+.2f}pp z_ce={z:+.2f} {verdict}"
        )
        print(line)
        region_summary.append(line)

    chance_ce = math.log(8)
    base_pl8 = cells[(8, 1)]
    assert base_pl8["ce_mean"] < chance_ce, (
        f"pl=8 n=1 ce {base_pl8['ce_mean']} >= chance {chance_ce}"
    )

    print()
    print(f"[pl_sweep_n4] total wall {time.time() - t_total:.0f}s")
    print("[pl_sweep_n4] OK")


if __name__ == "__main__":
    main()
