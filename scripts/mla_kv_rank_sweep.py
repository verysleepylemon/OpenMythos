"""MLA latent rank sweep.

DeepSeek MLA's whole point is compressing K/V via low-rank latent
projections (kv_lora_rank). Smaller rank -> less KV cache, fewer
params; larger rank -> more capacity but loses the MLA win.

Defaults: q_lora_rank=32, kv_lora_rank=16. We sweep kv_lora_rank in
{8, 16, 32} (q_lora_rank held at 32). 3 seeds x 200 steps.
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

KV_RANK_GRID = [8, 16, 32]
N_SEEDS = 3
STEPS = 200
N_LOOPS = 2


def run(kv_lora_rank: int, seed: int) -> tuple[float, float, int]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=4)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = N_LOOPS
    cfg.kv_lora_rank = kv_lora_rank
    model = OpenMythos(cfg)
    train_reverse_copy(model, task, n_loops=N_LOOPS, steps=STEPS, seed=seed)
    ce, acc = evaluate(model, task, n_loops=N_LOOPS, seed=seed + 9999)
    n_params = sum(p.numel() for p in model.parameters())
    return acc * 100, ce, n_params


def main() -> None:
    print(f"[mla_rank] KV_RANK_GRID={KV_RANK_GRID} N_SEEDS={N_SEEDS} STEPS={STEPS}")
    rows = []
    for r in KV_RANK_GRID:
        accs, ces, params = [], [], None
        t0 = time.time()
        for s in range(N_SEEDS):
            a, ce, params = run(r, s)
            accs.append(a)
            ces.append(ce)
        wall = time.time() - t0
        rows.append({
            "kv_lora_rank": r, "params": params,
            "ce_mean": statistics.mean(ces), "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs), "acc_std": statistics.pstdev(accs),
            "wall": wall,
        })
        print(f"[mla_rank] kv_lora_rank={r:>2d}  "
              f"params={params:>6d}  "
              f"ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
              f"acc={rows[-1]['acc_mean']:.2f}%  wall={wall:.1f}s")

    print("\n  kv_lora_rank  params   eval_ce (mean +/- std)    acc% (mean +/- std)")
    for r in rows:
        print(f"      {r['kv_lora_rank']:>2d}        {r['params']:>6d}   "
              f"{r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    base = next(r for r in rows if r["kv_lora_rank"] == 16)
    print()
    for r in rows:
        if r["kv_lora_rank"] == 16:
            continue
        delta = r["ce_mean"] - base["ce_mean"]
        pooled = math.sqrt(r["ce_std"] ** 2 + base["ce_std"] ** 2)
        z = delta / pooled if pooled > 0 else float("inf")
        dp = r["params"] - base["params"]
        print(f"  kv_lora_rank={r['kv_lora_rank']:>2d} vs 16: "
              f"delta_ce={delta:+.4f}  z={z:+.2f}  dparams={dp:+d}")

    chance_ce = math.log(8)
    for r in rows:
        assert r["ce_mean"] < chance_ce
    print("\n[mla_rank] OK")


if __name__ == "__main__":
    main()
