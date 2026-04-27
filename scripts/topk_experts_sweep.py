"""top-k experts sweep.

`n_experts_per_tok` controls how many experts each token routes to per
forward pass. Default tiny config uses k=2 of n=4 experts. We sweep k in
{1, 2, 3} on the canonical reverse-copy task, 3 seeds each, and report
whether routing to more experts helps.

Hypothesis: on a tiny task with 4 small experts and a shared expert,
k=2 should already capture most of the benefit; k=3 adds compute but
little quality; k=1 may underfit.
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

K_GRID = [1, 2, 3]
N_SEEDS = 3
STEPS = 200
N_LOOPS = 2


def run(k: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=4)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = N_LOOPS
    cfg.n_experts_per_tok = k
    model = OpenMythos(cfg)
    train_reverse_copy(model, task, n_loops=N_LOOPS, steps=STEPS, seed=seed)
    ce, acc = evaluate(model, task, n_loops=N_LOOPS, seed=seed + 9999)
    return acc * 100, ce


def main() -> None:
    print(f"[topk] K_GRID={K_GRID} N_SEEDS={N_SEEDS} STEPS={STEPS} (out of n_experts=4)")
    rows = []
    for k in K_GRID:
        accs, ces = [], []
        t0 = time.time()
        for s in range(N_SEEDS):
            a, c = run(k, s)
            accs.append(a)
            ces.append(c)
        wall = time.time() - t0
        rows.append({
            "k": k,
            "ce_mean": statistics.mean(ces),
            "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs),
            "wall": wall,
        })
        print(f"[topk] k={k}  ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
              f"acc={rows[-1]['acc_mean']:.2f}%  wall={wall:.1f}s")

    print("\n  k    eval_ce (mean +/- std)    acc% (mean +/- std)")
    for r in rows:
        print(f"  {r['k']}    {r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    base = next(r for r in rows if r["k"] == 2)
    best = min(rows, key=lambda r: r["ce_mean"])
    delta = base["ce_mean"] - best["ce_mean"]
    pooled = math.sqrt(base["ce_std"] ** 2 + best["ce_std"] ** 2)
    z = delta / pooled if pooled > 0 else float("inf")
    print(f"\n[topk] best_k={best['k']}  baseline_k=2  delta={delta:+.4f}  z={z:.2f}")
    if z > 1.5 and best["k"] != 2:
        print("[topk] verdict: a different k materially beats k=2.")
    else:
        print("[topk] verdict: k=2 is within seed noise of the best.")

    chance_ce = math.log(8)
    for r in rows:
        assert r["ce_mean"] < chance_ce, f"k={r['k']} ce {r['ce_mean']:.4f} >= chance"
    print("\n[topk] OK")


if __name__ == "__main__":
    main()
