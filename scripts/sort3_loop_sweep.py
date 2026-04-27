"""Sort-3 task — does sorting reward depth?

We ask the model to sort 3 numbers from a small alphabet:
- input: [a, b, c, SEP, ?, ?, ?]   where SEP is a sentinel token
- target: [_, _, _, _, sorted(a,b,c)]

Sorting requires comparing pairs, which is a non-trivial computation.
We expect more loops to help (each loop can refine the ordering).

vocab = 8 numbers + 1 SEP = 9 tokens
seq_len = 7
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

NUM_VOCAB = 8        # numbers 0..7
SEP = 8              # sentinel
VOCAB = 9
SEQ_LEN = 7
INPUT_LEN = 4        # a, b, c, SEP
OUTPUT_LEN = 3
BATCH = 64
STEPS = 400
LR = 3e-3
EVAL_BATCHES = 16
LOOP_GRID = [1, 2, 4]
N_SEEDS = 3


def make_batch(rng):
    nums = torch.randint(0, NUM_VOCAB, (BATCH, 3), generator=rng)
    sorted_nums, _ = torch.sort(nums, dim=1)
    sep = torch.full((BATCH, 1), SEP, dtype=nums.dtype)
    seq = torch.cat([nums, sep, sorted_nums], dim=1)  # length 7
    x = seq[:, :-1]   # length 6
    y = seq[:, 1:]    # length 6
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
            # Score only the 3 sorted-output positions (positions 3,4,5 of y)
            t = y[:, 3:6]
            p = preds[:, 3:6]
            correct += (p == t).sum().item()
            total += t.numel()
    return correct / max(1, total), sum(losses) / len(losses)


def main() -> None:
    print(f"[sort3] LOOP_GRID={LOOP_GRID} N_SEEDS={N_SEEDS} STEPS={STEPS} VOCAB={VOCAB}")
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
        print(f"[sort3] n_loops={n}  ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
              f"sort_acc={rows[-1]['acc_mean']:.2f}%  wall={wall:.1f}s")

    print("\n  n_loops  eval_ce (mean +/- std)    sort_acc% (mean +/- std)")
    for r in rows:
        print(f"  {r['n']:>5}    {r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    base = rows[0]
    best = min(rows, key=lambda r: r["ce_mean"])
    delta = base["ce_mean"] - best["ce_mean"]
    pooled = math.sqrt(base["ce_std"] ** 2 + best["ce_std"] ** 2)
    z = delta / pooled if pooled > 0 else float("inf")
    print(f"\n[sort3] best_n_loops={best['n']}  delta_vs_n1={delta:+.4f}  "
          f"pooled_std={pooled:.4f}  z={z:.2f}")
    if z > 1.5:
        print("[sort3] verdict: more loops help on sort-3 (real signal).")
    else:
        print("[sort3] verdict: depth gain still within seed noise.")

    # Random baseline accuracy = 1/NUM_VOCAB ~ 12.5%
    chance = 100.0 / NUM_VOCAB
    for r in rows:
        assert r["acc_mean"] > chance, f"sort_acc {r['acc_mean']:.2f}% below chance {chance:.2f}%"
    print(f"[sort3] all configs > chance ({chance:.2f}%)")
    print("\n[sort3] OK")


if __name__ == "__main__":
    main()
