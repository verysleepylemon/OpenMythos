"""Sweep gradient clipping threshold across seeds on the toy task.

Compares clip_grad_norm thresholds {0.5, 1.0, 2.0, +inf (no clip)} with
3 seeds each. Reports mean +/- std eval CE and rev_acc, plus the average
fraction of steps that actually triggered clipping.
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

VOCAB = 32
PROMPT_LEN = 6
SEQ_LEN = 12
BATCH = 32
STEPS = 200
LR = 3e-3
N_LOOPS = 3
EVAL_BATCHES = 16
CLIP_GRID = [0.5, 1.0, 2.0, float("inf")]
N_SEEDS = 3


def make_batch(rng):
    prompt = torch.randint(0, VOCAB, (BATCH, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def train_one(clip: float, seed: int) -> tuple[float, float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = N_LOOPS
    model = OpenMythos(cfg)
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    triggered = 0
    model.train()
    for _ in range(STEPS):
        x, y = make_batch(rng)
        logits = model(x, n_loops=N_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        if math.isfinite(clip):
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()
            if total_norm > clip:
                triggered += 1
        opt.step()

    model.eval()
    correct = total = 0
    losses = []
    eval_rng = torch.Generator().manual_seed(seed + 9999)
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = make_batch(eval_rng)
            logits = model(x, n_loops=N_LOOPS)
            losses.append(F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item())
            preds = logits.argmax(dim=-1)
            t = y[:, PROMPT_LEN - 1 : 2 * PROMPT_LEN - 1]
            p = preds[:, PROMPT_LEN - 1 : 2 * PROMPT_LEN - 1]
            correct += (p == t).sum().item()
            total += t.numel()
    return correct / max(1, total), sum(losses) / len(losses), triggered / STEPS


def fmt(c: float) -> str:
    return "no-clip" if not math.isfinite(c) else f"{c}"


def main() -> None:
    print(f"[clip] sweeping clip={[fmt(c) for c in CLIP_GRID]} N_SEEDS={N_SEEDS}")
    rows = []
    for c in CLIP_GRID:
        accs = []
        ces = []
        trigs = []
        t0 = time.time()
        for s in range(N_SEEDS):
            a, ce, frac = train_one(c, s)
            accs.append(a * 100)
            ces.append(ce)
            trigs.append(frac)
        wall = time.time() - t0
        rows.append({
            "clip": c,
            "ce_mean": statistics.mean(ces),
            "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs),
            "trig_mean": statistics.mean(trigs),
            "wall": wall,
        })
        print(f"[clip] clip={fmt(c):<8}  ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
              f"acc={rows[-1]['acc_mean']:.2f}%  triggered={rows[-1]['trig_mean']*100:.1f}%  wall={wall:.1f}s")

    print("\n  clip       eval_ce (mean +/- std)    rev_acc% (mean +/- std)    trig%")
    for r in rows:
        print(f"  {fmt(r['clip']):<8}  {r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}      {r['trig_mean']*100:5.1f}")

    best = min(rows, key=lambda r: r["ce_mean"])
    base = next(r for r in rows if not math.isfinite(r["clip"]))
    delta = base["ce_mean"] - best["ce_mean"]
    pooled = math.sqrt(base["ce_std"] ** 2 + best["ce_std"] ** 2)
    z = delta / pooled if pooled > 0 else float("inf")
    print(f"\n[clip] best={fmt(best['clip'])}  delta_vs_no_clip={delta:+.4f}  "
          f"pooled_std={pooled:.4f}  z={z:.2f}")

    uniform_ce = math.log(VOCAB)
    for r in rows:
        assert r["ce_mean"] < uniform_ce
    print("\n[clip] OK")


if __name__ == "__main__":
    main()
