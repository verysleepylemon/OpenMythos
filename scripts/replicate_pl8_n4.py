"""replicate_pl8_n4.py - 6-seed audit of the ONLY surviving recurrence win.

dim_x_loops_at_hard.py reported the only z<-3 result on this branch:
at dim=128 prompt_len=8 with prelude=1 coda=2 STEPS=2000, n=4 vs n=1
gives ce delta z=-3.20 and acc +1.87pp. That used 3 seeds.

After headline_replicate killed the +13.17pp pl=12 claim as a 3-seed
artifact, the methodological lesson is: 3 seeds is statistical underwear
when seed std is +/-12pp. This script reruns dim=128 pl=8 n in {1,4} with
6 seeds to confirm the z=-3.20 win survives at proper sample size.

If it survives: dim_x_loops_at_hard remains the rock-solid headline of
the entire branch, and "n_loops=4 at dim>=128 pl in {6,8}" is a real
shippable recipe.

If it dies too: the entire recurrent-depth thesis on this branch is
"no significant effect at this scale", and that itself is a publishable
honest result.
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

LOOP_GRID = [1, 4]
N_SEEDS = 6
STEPS = 2000
PROMPT_LEN = 8
DIM = 128


def run(n_loops, seed):
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


def main():
    print(
        f"[replicate_pl8_n4] dim={DIM} pl={PROMPT_LEN} STEPS={STEPS} "
        f"loops={LOOP_GRID} seeds={N_SEEDS}"
    )
    rows = {}
    for n in LOOP_GRID:
        ces, accs = [], []
        t0 = time.time()
        for seed in range(N_SEEDS):
            a, ce = run(n, seed)
            ces.append(ce)
            accs.append(a)
            print(
                f"[replicate_pl8_n4] n={n} seed={seed} ce={ce:.4f} acc={a:.2f}%"
            )
        rows[n] = {
            "ce_mean": statistics.fmean(ces),
            "ce_std": statistics.stdev(ces) if len(ces) > 1 else 0.0,
            "acc_mean": statistics.fmean(accs),
            "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "wall": time.time() - t0,
        }
        r = rows[n]
        print(
            f"[replicate_pl8_n4] n={n} SUMMARY ce={r['ce_mean']:.4f}+/-{r['ce_std']:.4f} "
            f"acc={r['acc_mean']:.2f}+/-{r['acc_std']:.2f} wall={r['wall']:.0f}s"
        )

    base = rows[LOOP_GRID[0]]
    print()
    print(f"  delta vs n={LOOP_GRID[0]} ({N_SEEDS} seeds):")
    for n in LOOP_GRID[1:]:
        d_ce = rows[n]["ce_mean"] - base["ce_mean"]
        d_acc = rows[n]["acc_mean"] - base["acc_mean"]
        pooled = math.sqrt(rows[n]["ce_std"] ** 2 + base["ce_std"] ** 2)
        z = d_ce / pooled if pooled > 0 else 0.0
        verdict = "<- helps" if d_acc > 0 else "<- hurts"
        print(
            f"    n={n}: delta_ce={d_ce:+.4f} z={z:+.2f} delta_acc={d_acc:+.2f}pp {verdict}"
        )

    chance_ce = math.log(8)
    assert base["ce_mean"] < chance_ce, (
        f"n={LOOP_GRID[0]} ce {base['ce_mean']} >= chance {chance_ce}"
    )

    print()
    print("[replicate_pl8_n4] OK")


if __name__ == "__main__":
    main()
