"""Generalize the depth-aware clip recipe across prompt lengths.

clip_universal showed clip=2.0 rescues n=8 at pl=12 by +10.12pp.
Does this generalize to other prompt lengths? Sweep clip in {1.0,
2.0} at n=8 across pl in {8, 12, 14}, 3 seeds.

If +10pp at pl=12 reproduces at pl=14 and shows up at pl=8 too,
the recipe ships. If it's pl=12-specific, it's not a recipe.
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

PL_GRID = [8, 12, 14]
CLIP_GRID = [1.0, 2.0]
N_LOOPS = 8
N_SEEDS = 3
STEPS = 2000
DIM = 128
LR = 3e-3
BATCH = 32


def train(model, task, clip, seed):
    model.train()
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for _ in range(STEPS):
        x, y = task.make_batch(BATCH, rng)
        logits = model(x, n_loops=N_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, task.vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()


def run(pl, clip, seed):
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=pl)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = N_LOOPS
    cfg.prelude_layers = 1
    cfg.coda_layers = 2
    cfg.dim = DIM
    cfg.expert_dim = DIM // 2
    model = OpenMythos(cfg)
    train(model, task, clip, seed)
    ce, acc = evaluate(model, task, n_loops=N_LOOPS, seed=seed + 9999)
    return acc * 100, ce


def main():
    print(f"[clip_recipe_pl] dim={DIM} n_loops={N_LOOPS} STEPS={STEPS} "
          f"pls={PL_GRID} clips={CLIP_GRID}")
    grid = {}
    for pl in PL_GRID:
        for clip in CLIP_GRID:
            ces, accs = [], []
            t0 = time.time()
            for seed in range(N_SEEDS):
                a, ce = run(pl, clip, seed)
                ces.append(ce); accs.append(a)
            grid[(pl, clip)] = {
                "ce_mean": statistics.mean(ces),
                "acc_mean": statistics.mean(accs),
                "acc_std": statistics.pstdev(accs),
                "wall": time.time() - t0,
            }
            r = grid[(pl, clip)]
            print(f"[clip_recipe_pl] pl={pl} clip={clip:.1f}  "
                  f"acc={r['acc_mean']:.2f}+/-{r['acc_std']:.2f}  "
                  f"ce={r['ce_mean']:.4f}  wall={r['wall']:.0f}s")

    print("\n  rescue (clip=2.0 - clip=1.0) per pl, n_loops=8:")
    for pl in PL_GRID:
        b = grid[(pl, 1.0)]; r = grid[(pl, 2.0)]
        d_acc = r["acc_mean"] - b["acc_mean"]
        d_ce = r["ce_mean"] - b["ce_mean"]
        marker = " <- helps" if d_acc > 1.0 else (" <- hurts" if d_acc < -1.0 else " (neutral)")
        print(f"    pl={pl}: delta_ce={d_ce:+.4f}  delta_acc={d_acc:+.2f}pp{marker}")

    chance_ce = math.log(8)
    for r in grid.values():
        assert r["ce_mean"] < chance_ce
    print("\n[clip_recipe_pl] OK")


if __name__ == "__main__":
    main()
