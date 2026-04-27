"""Generality of the clip=2.0 rescue: does it help all n_loops?

clip_rescue showed clip=2.0 rescues n=2 by +5.82pp at pl=12 dim=128.
Question: is clip=2.0 a universal recipe, or only fixes the valley?

Sweep: n_loops in {1, 2, 4, 8} x clip in {1.0, 2.0}, 3 seeds.
ReverseCopy pl=12 dim=128 STEPS=2000.
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

LOOP_GRID = [1, 2, 4, 8]
CLIP_GRID = [1.0, 2.0]
N_SEEDS = 3
STEPS = 2000
PROMPT_LEN = 12
DIM = 128
LR = 3e-3
BATCH = 32


def train(model, task, n_loops, clip, seed):
    model.train()
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for _ in range(STEPS):
        x, y = task.make_batch(BATCH, rng)
        logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(logits.reshape(-1, task.vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()


def run(n_loops, clip, seed):
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
    train(model, task, n_loops, clip, seed)
    ce, acc = evaluate(model, task, n_loops=n_loops, seed=seed + 9999)
    return acc * 100, ce


def main():
    print(f"[clip_universal] dim={DIM} pl={PROMPT_LEN} STEPS={STEPS} "
          f"loops={LOOP_GRID} clips={CLIP_GRID}")
    grid = {}
    for n in LOOP_GRID:
        for clip in CLIP_GRID:
            ces, accs = [], []
            t0 = time.time()
            for seed in range(N_SEEDS):
                a, ce = run(n, clip, seed)
                ces.append(ce); accs.append(a)
            grid[(n, clip)] = {
                "ce_mean": statistics.mean(ces),
                "acc_mean": statistics.mean(accs),
                "acc_std": statistics.pstdev(accs),
                "wall": time.time() - t0,
            }
            r = grid[(n, clip)]
            print(f"[clip_universal] n={n} clip={clip:.1f}  "
                  f"acc={r['acc_mean']:.2f}+/-{r['acc_std']:.2f}  "
                  f"ce={r['ce_mean']:.4f}  wall={r['wall']:.0f}s")

    print("\n  delta (clip=2.0 vs clip=1.0) per n_loops:")
    for n in LOOP_GRID:
        b = grid[(n, 1.0)]
        r = grid[(n, 2.0)]
        d_acc = r["acc_mean"] - b["acc_mean"]
        d_ce = r["ce_mean"] - b["ce_mean"]
        marker = " <- helps" if d_acc > 1.0 else (" <- hurts" if d_acc < -1.0 else "")
        print(f"    n={n}: delta_ce={d_ce:+.4f}  delta_acc={d_acc:+.2f}pp{marker}")

    chance_ce = math.log(8)
    for r in grid.values():
        assert r["ce_mean"] < chance_ce
    print("\n[clip_universal] OK")


if __name__ == "__main__":
    main()
