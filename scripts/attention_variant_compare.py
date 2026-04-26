"""Train MLA vs GQA on the same toy task and compare convergence.

bench_attention.py already measures *forward / generate* wall time for both
attention variants, but does not train them. This script trains a fresh tiny
model with each `attn_type` setting on the reverse-copy task with identical
hyperparameters and reports per-variant:

  - parameter count
  - learning curve milestones (loss at step 50, 100, 200)
  - eval cross-entropy
  - reverse-half greedy accuracy
  - wall time

Useful for confirming that the attention abstraction in OpenMythos is a true
drop-in: both variants should learn the same task, even if at different rates
or final accuracies given the same toy budget.
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

from scripts._common import (
    OpenMythos,
    build_tiny_gqa_config,
    build_tiny_mla_config,
)

VOCAB = 32
PROMPT_LEN = 6
SEQ_LEN = 12
BATCH = 32
STEPS = 200
LR = 3e-3
N_LOOPS = 3
MILESTONES = [50, 100, 200]
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
            losses.append(F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item())
            preds = logits.argmax(dim=-1)
            t = y[:, PROMPT_LEN - 1 : 2 * PROMPT_LEN - 1]
            p = preds[:, PROMPT_LEN - 1 : 2 * PROMPT_LEN - 1]
            correct += (p == t).sum().item()
            total += t.numel()
    return correct / max(1, total), sum(losses) / len(losses)


def make_model(attn_type: str) -> OpenMythos:
    if attn_type == "mla":
        cfg = build_tiny_mla_config()
    elif attn_type == "gqa":
        cfg = build_tiny_gqa_config()
    else:
        raise ValueError(f"unknown attn_type {attn_type!r}")
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = N_LOOPS
    return OpenMythos(cfg)


def train(model: OpenMythos, seed: int) -> dict[int, float]:
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    milestones: dict[int, float] = {}
    for step in range(1, STEPS + 1):
        x, y = make_batch(rng)
        logits = model(x, n_loops=N_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in MILESTONES:
            milestones[step] = loss.item()
    return milestones


def main() -> None:
    print(f"[attn] comparing MLA vs GQA on reverse-copy  steps={STEPS}  n_loops={N_LOOPS}")
    rows = []
    for attn in ("mla", "gqa"):
        torch.manual_seed(SEED)
        model = make_model(attn)
        n_params = sum(p.numel() for p in model.parameters())
        t0 = time.time()
        milestones = train(model, seed=SEED)
        wall = time.time() - t0
        eval_rng = torch.Generator().manual_seed(99)
        acc, ce = reverse_acc(model, eval_rng)
        rows.append({
            "attn": attn,
            "params": n_params,
            "milestones": milestones,
            "eval_ce": ce,
            "rev_acc": acc * 100,
            "wall": wall,
        })
        ms_str = "  ".join(f"step{k}={v:.4f}" for k, v in milestones.items())
        print(f"[attn] {attn.upper():<3}  params={n_params:,}  {ms_str}  "
              f"eval_ce={ce:.4f}  rev_acc={acc*100:.2f}%  wall={wall:.1f}s")

    print()
    print("  attn  params         step50    step100   step200   eval_ce   rev_acc%  wall_s")
    for r in rows:
        m = r["milestones"]
        print(f"  {r['attn'].upper():<4}  {r['params']:8,}     "
              f"{m[50]:7.4f}   {m[100]:7.4f}   {m[200]:7.4f}   "
              f"{r['eval_ce']:7.4f}   {r['rev_acc']:6.2f}    {r['wall']:5.1f}")

    mla = next(r for r in rows if r["attn"] == "mla")
    gqa = next(r for r in rows if r["attn"] == "gqa")
    delta_ce = gqa["eval_ce"] - mla["eval_ce"]
    print(f"\n[attn] eval_ce delta (GQA - MLA): {delta_ce:+.4f}")
    if abs(delta_ce) < 0.05:
        print("[attn] verdict: variants are within noise on this toy task")
    elif delta_ce > 0:
        print("[attn] verdict: MLA wins by a measurable margin")
    else:
        print("[attn] verdict: GQA wins by a measurable margin")

    # Both variants must learn (eval_ce strictly below uniform).
    import math
    uniform_ce = math.log(VOCAB)
    for r in rows:
        assert r["eval_ce"] < uniform_ce, (
            f"{r['attn']} failed to learn: ce={r['eval_ce']:.4f} >= uniform={uniform_ce:.4f}"
        )

    print("\n[attn] OK")


if __name__ == "__main__":
    main()
