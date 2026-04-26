"""LoRA adapter rank ablation for the recurrent block.

The recurrent block in OpenMythos uses a depth-wise LoRAAdapter (Bae et al.,
2024): a shared down/up projection plus a per-loop scale vector. The rank of
the bottleneck controls how much per-loop variation the otherwise weight-tied
recurrent block can express.

This script trains a fresh tiny model for each rank in {2, 4, 8, 16} on the
reverse-copy task and reports:

  - parameter count
  - final cross-entropy on a held-out batch
  - reverse-half greedy accuracy
  - wall time

Expected pattern: very small ranks under-parameterize the per-loop variation
and plateau at a higher loss; larger ranks learn faster but eventually
saturate (the toy task is easy enough that diminishing returns kick in).
"""

from __future__ import annotations

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
RANKS = [2, 4, 8, 16]
EVAL_BATCHES = 16
SEED = 0


def make_batch(rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = torch.randint(0, VOCAB, (BATCH, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def reverse_acc(model: OpenMythos, rng: torch.Generator) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    losses = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = make_batch(rng)
            logits = model(x, n_loops=N_LOOPS)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            losses.append(loss.item())
            preds = logits.argmax(dim=-1)
            t = y[:, PROMPT_LEN - 1 : 2 * PROMPT_LEN - 1]
            p = preds[:, PROMPT_LEN - 1 : 2 * PROMPT_LEN - 1]
            correct += (p == t).sum().item()
            total += t.numel()
    return correct / max(1, total), sum(losses) / len(losses)


def build_model(rank: int) -> OpenMythos:
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = N_LOOPS
    cfg.lora_rank = rank
    return OpenMythos(cfg)


def train(model: OpenMythos, seed: int) -> float:
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    last = float("nan")
    for _ in range(STEPS):
        x, y = make_batch(rng)
        logits = model(x, n_loops=N_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = loss.item()
    return last


def main() -> None:
    print(f"[lora] sweeping LoRA ranks {RANKS}  task=reverse-copy  steps={STEPS}  n_loops={N_LOOPS}")
    rows: list[tuple[int, int, float, float, float, float]] = []
    eval_rng = torch.Generator().manual_seed(99)
    for rank in RANKS:
        torch.manual_seed(SEED + rank)
        model = build_model(rank)
        lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora" in n.lower())
        total_params = sum(p.numel() for p in model.parameters())
        t0 = time.time()
        final_loss = train(model, seed=SEED + rank)
        wall = time.time() - t0
        # Re-seed the eval rng per row so each rank sees the same eval batches.
        eval_rng2 = torch.Generator().manual_seed(99)
        acc, ce = reverse_acc(model, eval_rng2)
        rows.append((rank, total_params, lora_params, final_loss, ce, acc * 100, wall))
        print(f"[lora] rank={rank:2d} params={total_params:,} (lora={lora_params:,})  "
              f"train_loss={final_loss:.4f}  eval_ce={ce:.4f}  rev_acc={acc*100:.2f}%  wall={wall:.1f}s")

    print()
    print("  rank   total_params  lora_params  train_loss  eval_ce  rev_acc%  wall_s")
    for rank, tot, lp, fl, ce, acc, wall in rows:
        print(f"  {rank:4d}  {tot:13,d}  {lp:11,d}    {fl:8.4f}  {ce:7.4f}   {acc:6.2f}  {wall:6.1f}")

    losses = [r[4] for r in rows]  # eval_ce
    best_idx = min(range(len(losses)), key=lambda i: losses[i])
    best_rank = rows[best_idx][0]
    print(f"\n[lora] best eval_ce at rank={best_rank}  ce={losses[best_idx]:.4f}")
    print(f"[lora] OK")


if __name__ == "__main__":
    main()
