"""Shared-expert ablation.

The tiny config has n_experts=4 routed + n_shared_experts=1 always-on.
Does removing the shared expert hurt? Does adding more shared experts help?

We sweep n_shared_experts in {0, 1, 2} on canonical reverse-copy,
3 seeds each, 200 steps. baseline = 1 shared expert.
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

SHARED_GRID = [0, 1, 2]
N_SEEDS = 3
STEPS = 200
N_LOOPS = 2


def run(n_shared: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=4)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = N_LOOPS
    cfg.n_shared_experts = n_shared
    model = OpenMythos(cfg)
    train_reverse_copy(model, task, n_loops=N_LOOPS, steps=STEPS, seed=seed)
    ce, acc = evaluate(model, task, n_loops=N_LOOPS, seed=seed + 9999)
    return acc * 100, ce


def main() -> None:
    print(f"[shared] SHARED_GRID={SHARED_GRID} N_SEEDS={N_SEEDS} STEPS={STEPS}")
    rows = []
    for s_n in SHARED_GRID:
        accs, ces = [], []
        t0 = time.time()
        for s in range(N_SEEDS):
            try:
                a, c = run(s_n, s)
            except Exception as e:
                print(f"[shared] n_shared={s_n} seed={s} FAILED: {e}")
                raise
            accs.append(a)
            ces.append(c)
        wall = time.time() - t0
        rows.append({
            "n": s_n,
            "ce_mean": statistics.mean(ces),
            "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs),
            "wall": wall,
        })
        print(f"[shared] n_shared={s_n}  ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
              f"acc={rows[-1]['acc_mean']:.2f}%  wall={wall:.1f}s")

    print("\n  n_shared  eval_ce (mean +/- std)    acc% (mean +/- std)")
    for r in rows:
        print(f"  {r['n']:>3}      {r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    base = next(r for r in rows if r["n"] == 1)
    best = min(rows, key=lambda r: r["ce_mean"])
    delta = base["ce_mean"] - best["ce_mean"]
    pooled = math.sqrt(base["ce_std"] ** 2 + best["ce_std"] ** 2)
    z = delta / pooled if pooled > 0 else float("inf")
    print(f"\n[shared] best_n_shared={best['n']}  baseline=1  delta={delta:+.4f}  z={z:.2f}")
    if z > 1.5 and best["n"] != 1:
        print("[shared] verdict: a different shared-expert count materially beats 1.")
    else:
        print("[shared] verdict: 1 shared expert is within seed noise of the best.")

    chance_ce = math.log(8)
    for r in rows:
        assert r["ce_mean"] < chance_ce, f"n_shared={r['n']} ce {r['ce_mean']:.4f} >= chance"
    print("\n[shared] OK")


if __name__ == "__main__":
    main()
