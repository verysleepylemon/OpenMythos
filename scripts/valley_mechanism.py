"""Mechanism study: WHY does the valley of death exist at intermediate n_loops?

harder_n8 + harder_pl14 found a non-monotonic loop landscape at hard
ReverseCopy (pl=12, dim=128): n=1 baseline, n in (2,4) HURTS heavily,
n=8 wins. This script asks "why?" by tracking three observables across
the full training trajectory:

  - smoothed train loss every K steps
  - gradient L2 norm every K steps (to detect noisy or unstable updates)
  - gradient norm post-clip ratio (how often clip fires; clip=1.0)

Hypothesis to test:
  H_NOISE: intermediate n_loops have HIGHER gradient noise than either
           the shallow (n=1) or deep (n=8) regime, so they make slower
           progress per step and never recover.
  H_UNROLL: clip fires MORE OFTEN at intermediate n_loops because the
            unrolled gradient through the recurrent block accumulates
            destructively, while at n=8 the model adapts to it.
  H_SLOW: intermediate n_loops just train slower; given more steps, they
          would catch up. (If true, valley is a budget artifact.)

dim=128 vocab=8 prompt_len=12 STEPS=2000 LOG_EVERY=100 n in {1,2,4,8}
3 seeds. Saves per-step diagnostics to artifacts/valley_curves.json
for later plotting.
"""

from __future__ import annotations

import json
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
N_SEEDS = 3
STEPS = 2000
LOG_EVERY = 100
PROMPT_LEN = 12
DIM = 128
CLIP = 1.0
LR = 3e-3
BATCH = 32


def train_with_diagnostics(model, task, n_loops, seed):
    """Custom train loop: log smoothed loss, raw grad norm, clip rate."""
    model.train()
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    bucket_loss = []
    bucket_gnorm = []
    bucket_clipped = 0
    log = {"step": [], "loss": [], "grad_norm_mean": [], "clip_rate": []}
    for step in range(STEPS):
        x, y = task.make_batch(BATCH, rng)
        logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(logits.reshape(-1, task.vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        # raw grad norm before clip
        sqsum = 0.0
        for p in model.parameters():
            if p.grad is not None:
                sqsum += float(p.grad.detach().pow(2).sum().item())
        gnorm = math.sqrt(sqsum)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        bucket_loss.append(loss.item())
        bucket_gnorm.append(gnorm)
        if gnorm > CLIP:
            bucket_clipped += 1
        if (step + 1) % LOG_EVERY == 0:
            log["step"].append(step + 1)
            log["loss"].append(statistics.mean(bucket_loss))
            log["grad_norm_mean"].append(statistics.mean(bucket_gnorm))
            log["clip_rate"].append(bucket_clipped / LOG_EVERY)
            bucket_loss, bucket_gnorm, bucket_clipped = [], [], 0
    return log


def run(n_loops, seed):
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=PROMPT_LEN)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = n_loops
    cfg.prelude_layers = 1
    cfg.coda_layers = 2
    cfg.dim = DIM
    cfg.expert_dim = DIM // 2
    model = OpenMythos(cfg)
    log = train_with_diagnostics(model, task, n_loops, seed)
    ce, acc = evaluate(model, task, n_loops=n_loops, seed=seed + 9999)
    return log, ce, acc * 100


def main():
    print(f"[valley] dim={DIM} pl={PROMPT_LEN} STEPS={STEPS} LOG={LOG_EVERY} "
          f"LOOPS={LOOP_GRID} SEEDS={N_SEEDS}")
    out = {}
    for n in LOOP_GRID:
        runs = []
        ces, accs = [], []
        t0 = time.time()
        for seed in range(N_SEEDS):
            log, ce, acc = run(n, seed)
            runs.append(log)
            ces.append(ce); accs.append(acc)
        # average across seeds
        n_pts = len(runs[0]["step"])
        avg = {
            "step": runs[0]["step"],
            "loss": [statistics.mean(r["loss"][i] for r in runs) for i in range(n_pts)],
            "grad_norm_mean": [statistics.mean(r["grad_norm_mean"][i] for r in runs) for i in range(n_pts)],
            "clip_rate": [statistics.mean(r["clip_rate"][i] for r in runs) for i in range(n_pts)],
        }
        out[str(n)] = {
            "curve": avg,
            "ce_final": statistics.mean(ces),
            "acc_final": statistics.mean(accs),
        }
        wall = time.time() - t0
        # Print compact ASCII summary
        midpt = n_pts // 2
        endpt = n_pts - 1
        print(f"[valley] n={n}  acc={statistics.mean(accs):.2f}%  "
              f"loss[step={avg['step'][midpt]}]={avg['loss'][midpt]:.3f}  "
              f"loss[step={avg['step'][endpt]}]={avg['loss'][endpt]:.3f}  "
              f"gnorm[mid]={avg['grad_norm_mean'][midpt]:.2f}  "
              f"gnorm[end]={avg['grad_norm_mean'][endpt]:.2f}  "
              f"clip[mid]={avg['clip_rate'][midpt]:.2f}  "
              f"wall={wall:.0f}s")

    # Highlight valley signal
    print("\n  step-by-step loss (mean across seeds):")
    print("   step      " + "    ".join(f"n={n:<3}" for n in LOOP_GRID))
    n_pts = len(out[str(LOOP_GRID[0])]["curve"]["step"])
    for i in range(n_pts):
        row = [f"{out[str(LOOP_GRID[0])]['curve']['step'][i]:>5}"]
        for n in LOOP_GRID:
            row.append(f"  {out[str(n)]['curve']['loss'][i]:6.3f}")
        if i % 2 == 0:
            print("   " + "  ".join(row))

    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    out_path = art / "valley_curves.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[valley] wrote {out_path}")
    # final assertion - at least one config must be below chance
    assert any(out[k]["ce_final"] < math.log(8) for k in out)
    print("[valley] OK")


if __name__ == "__main__":
    main()
