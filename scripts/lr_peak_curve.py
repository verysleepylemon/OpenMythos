"""Peak LR sweep — real LR magnitudes, multi-seed.

We've ablated schedule shape (cosine_lr_warmup) but not raw LR magnitude.
This sweeps peak LR in {1e-3, 3e-3, 1e-2} on the canonical reverse-copy task,
3 seeds each. Reports z-score relative to LR=3e-3 (our standard choice)
to show whether the chosen LR is roughly optimal.
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

LR_GRID = [1e-3, 3e-3, 1e-2]
N_SEEDS = 3
STEPS = 200
N_LOOPS = 2


def run(lr: float, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=4)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = N_LOOPS
    model = OpenMythos(cfg)
    train_reverse_copy(model, task, n_loops=N_LOOPS, steps=STEPS, lr=lr, seed=seed)
    ce, acc = evaluate(model, task, n_loops=N_LOOPS, seed=seed + 9999)
    return acc * 100, ce


def main() -> None:
    print(f"[lr_peak] LR_GRID={LR_GRID} N_SEEDS={N_SEEDS} STEPS={STEPS}")
    rows = []
    for lr in LR_GRID:
        accs, ces = [], []
        t0 = time.time()
        for s in range(N_SEEDS):
            a, c = run(lr, s)
            accs.append(a)
            ces.append(c)
        wall = time.time() - t0
        rows.append({
            "lr": lr,
            "ce_mean": statistics.mean(ces),
            "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs),
            "wall": wall,
        })
        print(f"[lr_peak] lr={lr:.0e}  ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
              f"acc={rows[-1]['acc_mean']:.2f}%  wall={wall:.1f}s")

    print("\n  lr        eval_ce (mean +/- std)    acc% (mean +/- std)")
    for r in rows:
        print(f"  {r['lr']:.0e}   {r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    # Compare best vs baseline (3e-3)
    base = next(r for r in rows if abs(r["lr"] - 3e-3) < 1e-9)
    best = min(rows, key=lambda r: r["ce_mean"])
    delta = base["ce_mean"] - best["ce_mean"]
    pooled = math.sqrt(base["ce_std"] ** 2 + best["ce_std"] ** 2)
    z = delta / pooled if pooled > 0 else float("inf")
    print(f"\n[lr_peak] best_lr={best['lr']:.0e}  baseline_lr=3e-3  delta={delta:+.4f}  z={z:.2f}")
    if z > 1.5 and best["lr"] != 3e-3:
        print("[lr_peak] verdict: a different LR materially beats 3e-3.")
    else:
        print("[lr_peak] verdict: 3e-3 is within seed noise of the best.")

    # Sanity: at least one config should achieve non-chance ce on copy task
    best_ce = best["ce_mean"]
    chance_ce = math.log(8)
    assert best_ce < chance_ce, f"best ce {best_ce:.4f} >= chance {chance_ce:.4f}"
    print("\n[lr_peak] OK")


if __name__ == "__main__":
    main()
