"""accel_curve.py - quantify the acceleration vs asymptote tradeoff.

prior loops_at_optimal_arch_long.py (3 seeds, STEPS=2500) showed all
loop counts converge to identical ce at long training. The +1.40pp /
5x variance collapse from replicate_pl8_n4 at STEPS=2000 is therefore
an ACCELERATION effect, not an ASYMPTOTE effect.

This script characterizes the acceleration curve at 6 seeds: how many
fewer training steps does n=4 need to match n=1's STEPS=2000 quality?

At dim=128 pl=8 prelude=1 coda=2, train n in {1, 4} for STEPS in
{500, 1000, 1500, 2000, 3000} at 6 seeds each. Plot/print
acc-vs-steps curves to find the equivalent-quality step ratio.

If n=4 at STEPS=1000 matches n=1 at STEPS=2000: 2x speedup.
If n=4 at STEPS=1500 matches n=1 at STEPS=2000: 1.33x speedup.
With n=4 cost ~1.55x n=1, breakeven is at speedup >= 1.55x.

This makes the ship recipe quantitative: "use n=4 if you can afford
1.55x training cost; you'll get +1.40pp acc OR equivalent quality
at fewer total steps."

NOTE: 5 step counts x 2 loops x 6 seeds = 60 runs. ~5h. Run only
overnight or as a CI nightly.
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
STEPS_GRID = [500, 1000, 1500, 2000, 3000]
N_SEEDS = 6
PROMPT_LEN = 8
DIM = 128


def run(n_loops, steps, seed):
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
    train_reverse_copy(model, task, n_loops=n_loops, steps=steps, seed=seed)
    ce, acc = evaluate(model, task, n_loops=n_loops, seed=seed + 9999)
    return acc * 100, ce


def main():
    print(
        f"[accel_curve] dim={DIM} pl={PROMPT_LEN} "
        f"steps_grid={STEPS_GRID} loops={LOOP_GRID} seeds={N_SEEDS}",
        flush=True,
    )
    cells = {}
    t_total = time.time()
    for steps in STEPS_GRID:
        for n in LOOP_GRID:
            ces, accs = [], []
            t0 = time.time()
            for seed in range(N_SEEDS):
                a, ce = run(n, steps, seed)
                ces.append(ce)
                accs.append(a)
                print(
                    f"[accel_curve] steps={steps} n={n} seed={seed} "
                    f"ce={ce:.4f} acc={a:.2f}%",
                    flush=True,
                )
            cells[(steps, n)] = {
                "ce_mean": statistics.fmean(ces),
                "ce_std": statistics.stdev(ces) if len(ces) > 1 else 0.0,
                "acc_mean": statistics.fmean(accs),
                "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                "wall": time.time() - t0,
            }
            r = cells[(steps, n)]
            print(
                f"[accel_curve] steps={steps} n={n} SUMMARY "
                f"ce={r['ce_mean']:.4f}+/-{r['ce_std']:.4f} "
                f"acc={r['acc_mean']:.2f}+/-{r['acc_std']:.2f} wall={r['wall']:.0f}s",
                flush=True,
            )

    print()
    print(f"  Acceleration curve at dim={DIM} pl={PROMPT_LEN}, {N_SEEDS} seeds:")
    print(f"  {'steps':>6} | {'n=1 acc':>16} | {'n=4 acc':>16} | d_acc")
    for s in STEPS_GRID:
        r1 = cells[(s, 1)]
        r4 = cells[(s, 4)]
        d = r4["acc_mean"] - r1["acc_mean"]
        print(
            f"  {s:>6} | {r1['acc_mean']:>6.2f}+/-{r1['acc_std']:.2f} "
            f"| {r4['acc_mean']:>6.2f}+/-{r4['acc_std']:.2f} | {d:+.2f}pp"
        )

    target = cells[(2000, 1)]["acc_mean"]
    print()
    print(
        f"  Target n=1@steps=2000 acc = {target:.2f}%. "
        f"Smallest STEPS where n=4 mean acc >= target:"
    )
    for s in STEPS_GRID:
        r4 = cells[(s, 4)]
        if r4["acc_mean"] >= target:
            print(f"    n=4 reaches {r4['acc_mean']:.2f}% at STEPS={s} "
                  f"(target {target:.2f}% at STEPS=2000) -> "
                  f"step ratio {2000/s:.2f}x")
            break
    else:
        print("    n=4 does NOT reach target at any STEPS in grid")

    chance_ce = math.log(8)
    sample = cells[(2000, 1)]
    assert sample["ce_mean"] < chance_ce, (
        f"steps=2000 n=1 ce {sample['ce_mean']} >= chance {chance_ce}"
    )

    print()
    print(f"[accel_curve] total wall {time.time() - t_total:.0f}s")
    print("[accel_curve] OK")


if __name__ == "__main__":
    main()
