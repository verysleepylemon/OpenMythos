"""Expert pool size sweep.

Default n_experts=4 with n_experts_per_tok=2. We sweep total expert
pool in {2, 4, 8}, keeping the activation ratio fixed at top-k=2
(except n=2 case where k must be <=2, so we keep k=2; this means
n=2 routes to ALL experts always).

3 seeds x 200 steps.
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

N_EXPERTS_GRID = [2, 4, 8]
N_SEEDS = 3
STEPS = 200
N_LOOPS = 2


def run(n_experts: int, seed: int) -> tuple[float, float, int]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=4)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = N_LOOPS
    cfg.n_experts = n_experts
    cfg.n_experts_per_tok = min(2, n_experts)
    model = OpenMythos(cfg)
    train_reverse_copy(model, task, n_loops=N_LOOPS, steps=STEPS, seed=seed)
    ce, acc = evaluate(model, task, n_loops=N_LOOPS, seed=seed + 9999)
    n_params = sum(p.numel() for p in model.parameters())
    return acc * 100, ce, n_params


def main() -> None:
    print(f"[ne_sweep] N_EXPERTS_GRID={N_EXPERTS_GRID} N_SEEDS={N_SEEDS} STEPS={STEPS}")
    rows = []
    for n in N_EXPERTS_GRID:
        accs, ces, params = [], [], None
        t0 = time.time()
        for s in range(N_SEEDS):
            a, ce, params = run(n, s)
            accs.append(a)
            ces.append(ce)
        wall = time.time() - t0
        rows.append({
            "n_experts": n, "params": params,
            "ce_mean": statistics.mean(ces), "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs), "acc_std": statistics.pstdev(accs),
            "wall": wall,
        })
        print(f"[ne_sweep] n_experts={n}  "
              f"params={params:>6d}  "
              f"ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
              f"acc={rows[-1]['acc_mean']:.2f}%  wall={wall:.1f}s")

    print("\n  n_experts  params   eval_ce (mean +/- std)    acc% (mean +/- std)")
    for r in rows:
        print(f"      {r['n_experts']}      {r['params']:>6d}   "
              f"{r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    base = next(r for r in rows if r["n_experts"] == 4)
    print()
    for r in rows:
        if r["n_experts"] == 4:
            continue
        delta = r["ce_mean"] - base["ce_mean"]
        pooled = math.sqrt(r["ce_std"] ** 2 + base["ce_std"] ** 2)
        z = delta / pooled if pooled > 0 else float("inf")
        dp = r["params"] - base["params"]
        print(f"  n_experts={r['n_experts']} vs 4: "
              f"delta_ce={delta:+.4f}  z={z:+.2f}  dparams={dp:+d}")

    chance_ce = math.log(8)
    for r in rows:
        assert r["ce_mean"] < chance_ce
    print("\n[ne_sweep] OK")


if __name__ == "__main__":
    main()
