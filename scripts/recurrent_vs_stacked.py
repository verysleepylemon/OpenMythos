"""Recurrent vs stacked depth comparison.

The architecture is: Prelude (P layers) -> RecurrentBlock (1 shared block looped T times) -> Coda (C layers).

The whole point of this design is that one shared block looped T times
is supposed to substitute for T stacked layers, with T-fold parameter savings.

We test that head-on:
  RECURRENT: prelude=1, coda=1, n_loops=K  (1 shared block called K times)
  STACKED:   prelude=1, coda=K, n_loops=1  (K independent coda blocks, 1 trivial loop)

Both pay roughly K-block compute. Stacked has K x more recurrent-block-equivalent params.
3 seeds each, 200 steps. Reports per-K ce, acc, and param count.
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

K_GRID = [1, 2, 4]
N_SEEDS = 3
STEPS = 200


def make_model(mode: str, K: int, task: ReverseCopyTask) -> torch.nn.Module:
    cfg = build_tiny_mla_config()
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    if mode == "recurrent":
        cfg.prelude_layers = 1
        cfg.coda_layers = 1
        cfg.max_loop_iters = K
    elif mode == "stacked":
        cfg.prelude_layers = 1
        cfg.coda_layers = K
        cfg.max_loop_iters = 1
    else:
        raise ValueError(mode)
    return OpenMythos(cfg)


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run(mode: str, K: int, seed: int) -> tuple[float, float, int]:
    torch.manual_seed(seed)
    task = ReverseCopyTask(vocab=8, prompt_len=4)
    model = make_model(mode, K, task)
    n_loops = K if mode == "recurrent" else 1
    train_reverse_copy(model, task, n_loops=n_loops, steps=STEPS, seed=seed)
    ce, acc = evaluate(model, task, n_loops=n_loops, seed=seed + 9999)
    return acc * 100, ce, n_params(model)


def main() -> None:
    print(f"[rec_vs_stk] K_GRID={K_GRID} N_SEEDS={N_SEEDS} STEPS={STEPS}")
    rows = []
    for mode in ["recurrent", "stacked"]:
        for K in K_GRID:
            accs, ces, params = [], [], None
            t0 = time.time()
            for s in range(N_SEEDS):
                a, c, p = run(mode, K, s)
                accs.append(a)
                ces.append(c)
                params = p
            wall = time.time() - t0
            rows.append({
                "mode": mode, "K": K, "params": params,
                "ce_mean": statistics.mean(ces), "ce_std": statistics.pstdev(ces),
                "acc_mean": statistics.mean(accs), "acc_std": statistics.pstdev(accs),
                "wall": wall,
            })
            print(f"[rec_vs_stk] mode={mode:>9} K={K}  "
                  f"params={params:>6d}  "
                  f"ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
                  f"acc={rows[-1]['acc_mean']:.2f}%  wall={wall:.1f}s")

    print("\n  mode       K  params   eval_ce (mean +/- std)    acc% (mean +/- std)")
    for r in rows:
        print(f"  {r['mode']:>9}  {r['K']}  {r['params']:>6d}   "
              f"{r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    # Per-K head-to-head: same K, different mode.
    print("\n  Head-to-head (same K, recurrent - stacked):")
    for K in K_GRID:
        rec = next(r for r in rows if r["mode"] == "recurrent" and r["K"] == K)
        stk = next(r for r in rows if r["mode"] == "stacked" and r["K"] == K)
        delta = rec["ce_mean"] - stk["ce_mean"]
        pooled = math.sqrt(rec["ce_std"] ** 2 + stk["ce_std"] ** 2)
        z = delta / pooled if pooled > 0 else float("inf")
        param_ratio = stk["params"] / rec["params"]
        print(f"    K={K}: rec_ce {rec['ce_mean']:.4f} - stk_ce {stk['ce_mean']:.4f} = "
              f"{delta:+.4f}  z={z:+.2f}  stk_params={param_ratio:.2f}x rec_params")

    chance_ce = math.log(8)
    for r in rows:
        assert r["ce_mean"] < chance_ce, f"{r['mode']} K={r['K']} ce {r['ce_mean']:.4f} >= chance"
    print("\n[rec_vs_stk] OK")


if __name__ == "__main__":
    main()
