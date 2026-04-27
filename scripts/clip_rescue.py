"""Causal test: does raising clip threshold rescue the valley?

valley_mechanism showed n=2 fires gradient clip on 37% of training
steps (vs 12% at n=4, 10% at n=8) and correlates with the valley
acc deficit. If clip-fighting is causal, raising the threshold
should let the optimizer follow the natural gradient and shrink
the valley. If it is merely correlated (clip is a symptom of some
deeper instability), raising clip will not help or will diverge.

dim=128 vocab=8 prompt_len=12 STEPS=2000 n=2 only (the worst
valley point), clip in {1.0, 2.0, 4.0, infinity}, 3 seeds.
Compare against valley_mechanism's n=1 baseline (91.75%) to see
if the clip rescue closes the gap.
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

CLIP_GRID = [1.0, 2.0, 4.0, float("inf")]
N_SEEDS = 3
STEPS = 2000
PROMPT_LEN = 12
DIM = 128
N_LOOPS = 2
LR = 3e-3
BATCH = 32


def train(model, task, clip, seed):
    model.train()
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    fired = 0
    for _ in range(STEPS):
        x, y = task.make_batch(BATCH, rng)
        logits = model(x, n_loops=N_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, task.vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        if math.isfinite(clip):
            sqsum = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    sqsum += float(p.grad.detach().pow(2).sum().item())
            if math.sqrt(sqsum) > clip:
                fired += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
    return fired / STEPS


def run(clip, seed):
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=PROMPT_LEN)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = N_LOOPS
    cfg.prelude_layers = 1
    cfg.coda_layers = 2
    cfg.dim = DIM
    cfg.expert_dim = DIM // 2
    model = OpenMythos(cfg)
    fire_rate = train(model, task, clip, seed)
    ce, acc = evaluate(model, task, n_loops=N_LOOPS, seed=seed + 9999)
    return acc * 100, ce, fire_rate


def main():
    print(f"[clip_rescue] dim={DIM} pl={PROMPT_LEN} STEPS={STEPS} "
          f"n_loops={N_LOOPS} CLIPS={CLIP_GRID}")
    grid = {}
    for clip in CLIP_GRID:
        ces, accs, fires = [], [], []
        t0 = time.time()
        for seed in range(N_SEEDS):
            a, ce, f = run(clip, seed)
            ces.append(ce); accs.append(a); fires.append(f)
        grid[clip] = {
            "ce_mean": statistics.mean(ces),
            "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs),
            "fire_mean": statistics.mean(fires),
            "wall": time.time() - t0,
        }
        r = grid[clip]
        c_str = "inf" if math.isinf(clip) else f"{clip:.1f}"
        print(f"[clip_rescue] clip={c_str}  "
              f"ce={r['ce_mean']:.4f}+/-{r['ce_std']:.4f}  "
              f"acc={r['acc_mean']:.2f}%+/-{r['acc_std']:.2f}  "
              f"fire={r['fire_mean']:.2f}  wall={r['wall']:.0f}s")

    base = grid[1.0]
    print("\n  rescue vs baseline (clip=1.0):")
    for clip in CLIP_GRID:
        if clip == 1.0:
            continue
        r = grid[clip]
        d_acc = r["acc_mean"] - base["acc_mean"]
        d_ce = r["ce_mean"] - base["ce_mean"]
        c_str = "inf" if math.isinf(clip) else f"{clip:.1f}"
        print(f"    clip={c_str}: delta_ce={d_ce:+.4f}  delta_acc={d_acc:+.2f}pp")

    chance_ce = math.log(8)
    for r in grid.values():
        assert r["ce_mean"] < chance_ce
    print("\n[clip_rescue] OK")


if __name__ == "__main__":
    main()
