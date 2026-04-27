"""Parity prediction task — does it actually reward more loops?

Earlier `seed_robustness.py` showed that on the reverse-copy toy the
single-seed `n_loops=4` win was just noise. The natural next question is:
**is there ANY synthetic task in our setup that actually rewards depth?**

Parity is a textbook "needs depth" problem for transformers: the model has
to XOR an arbitrary-length prefix of bits to predict the next parity bit.
Shallow attention layers struggle; recurrent depth (loops) should help.

Setup:
- vocab = 2 (binary)
- input: random bit string of length L
- target at position i: parity (XOR) of bits[0..i] (cumulative)
- BATCH=64, STEPS=300, multi-seed sweep over n_loops in {1, 2, 4, 8}.
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

VOCAB = 2
SEQ_LEN = 16
BATCH = 64
STEPS = 300
LR = 3e-3
EVAL_BATCHES = 16
LOOP_GRID = [1, 2, 4, 8]
N_SEEDS = 3


def make_batch(rng):
    bits = torch.randint(0, 2, (BATCH, SEQ_LEN), generator=rng)
    cum = torch.cumsum(bits, dim=1) % 2
    # input = bits, target at pos i = parity(bits[0..i+1]) shifted left
    # so the model predicts cum[t] from input[..t]
    x = bits[:, :-1]
    y = cum[:, 1:]
    return x, y


def train_one(n_loops: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = max(LOOP_GRID)
    model = OpenMythos(cfg)
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        x, y = make_batch(rng)
        logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    correct = total = 0
    losses = []
    eval_rng = torch.Generator().manual_seed(seed + 9999)
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = make_batch(eval_rng)
            logits = model(x, n_loops=n_loops)
            losses.append(F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item())
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / max(1, total), sum(losses) / len(losses)


def main() -> None:
    print(f"[parity] LOOP_GRID={LOOP_GRID} N_SEEDS={N_SEEDS} STEPS={STEPS}")
    rows = []
    for n in LOOP_GRID:
        accs = []
        ces = []
        t0 = time.time()
        for s in range(N_SEEDS):
            a, c = train_one(n, s)
            accs.append(a * 100)
            ces.append(c)
        wall = time.time() - t0
        rows.append({
            "n": n,
            "ce_mean": statistics.mean(ces),
            "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs),
            "wall": wall,
        })
        print(f"[parity] n_loops={n}  ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
              f"acc={rows[-1]['acc_mean']:.2f}%  wall={wall:.1f}s")

    print("\n  n_loops  eval_ce (mean +/- std)    parity_acc% (mean +/- std)")
    for r in rows:
        print(f"  {r['n']:>5}    {r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    base = rows[0]   # n=1
    best = min(rows, key=lambda r: r["ce_mean"])
    delta = base["ce_mean"] - best["ce_mean"]
    pooled = math.sqrt(base["ce_std"] ** 2 + best["ce_std"] ** 2)
    z = delta / pooled if pooled > 0 else float("inf")
    print(f"\n[parity] best_n_loops={best['n']}  delta_vs_n1={delta:+.4f}  "
          f"pooled_std={pooled:.4f}  z={z:.2f}")
    if z > 1.5:
        print("[parity] verdict: depth helps on parity (signal above noise).")
    else:
        print("[parity] verdict: depth gain still within seed noise; try longer training.")

    uniform_ce = math.log(VOCAB)
    for r in rows:
        assert r["ce_mean"] < uniform_ce + 0.05  # parity is hard; allow small slack
    print("\n[parity] OK")


if __name__ == "__main__":
    main()
