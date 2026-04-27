"""headline_replicate.py - audit the +13.17pp n=8 headline with 6 seeds.

harder_n8 reported +13.17pp acc gain (n=1 -> n=8) at pl=12 dim=128 STEPS=2000
across 3 seeds. Subsequent work (valley_mechanism, clip_universal,
clip_recipe_pl) revealed seed variance bands of +/-12pp at n=8 and +/-34pp at
pl=14. clip_universal at 3 seeds also showed n=4 (96.82+/-3.07) actually
beating n=8 (80.11+/-12.39) at pl=12 - directly contradicting the +13.17pp
headline.

This script reruns the headline configuration with DOUBLE the seeds (6 instead
of 3) to get a tighter CI on the headline claim.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import OpenMythos, build_tiny_mla_config
from scripts._eval import ReverseCopyTask, evaluate

LOOP_GRID = [1, 4, 8]
N_SEEDS = 6
STEPS = 2000
PROMPT_LEN = 12
DIM = 128
LR = 3e-3
BATCH = 32
CLIP = 1.0


def train(model, task, n_loops, seed):
    model.train()
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for _ in range(STEPS):
        x, y = task.make_batch(BATCH, rng)
        logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(logits.reshape(-1, task.vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()


def run(n_loops, seed):
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=PROMPT_LEN)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = max(LOOP_GRID)
    cfg.prelude_layers = 1
    cfg.coda_layers = 2
    cfg.dim = DIM
    cfg.expert_dim = DIM // 2
    model = OpenMythos(cfg)
    train(model, task, n_loops, seed)
    ce, acc = evaluate(model, task, n_loops=n_loops, seed=seed + 9999)
    return acc * 100, ce


def main():
    print(
        f"[headline_replicate] dim={DIM} pl={PROMPT_LEN} STEPS={STEPS} "
        f"loops={LOOP_GRID} seeds={N_SEEDS} clip={CLIP}"
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
                f"[headline_replicate] n={n} seed={seed} ce={ce:.4f} acc={a:.2f}%"
            )
        rows[n] = {
            "ce_mean": statistics.fmean(ces),
            "ce_std": statistics.stdev(ces) if len(ces) > 1 else 0.0,
            "acc_mean": statistics.fmean(accs),
            "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "wall": time.time() - t0,
        }
        print(
            f"[headline_replicate] n={n} SUMMARY ce={rows[n]['ce_mean']:.4f}"
            f"+/-{rows[n]['ce_std']:.4f} acc={rows[n]['acc_mean']:.2f}"
            f"+/-{rows[n]['acc_std']:.2f} wall={rows[n]['wall']:.0f}s"
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
    print("[headline_replicate] OK")


if __name__ == "__main__":
    main()
