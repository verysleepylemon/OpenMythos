"""Sequence-length extrapolation: train short, test longer.

Trains a tiny model on the simple "prompt then reverse" task at PROMPT_LEN=6
(total sequence 12), then evaluates next-token reverse accuracy on prompt
lengths 4, 6, 8, 10, 12 (sequences up to length 24 = 2x training horizon).

This probes whether the loop-indexed RoPE in the recurrent block keeps the
model coherent past its training horizon. A model that shares weights and
re-rotates positions per loop is expected to degrade more gracefully than a
deep transformer baseline.
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
PROMPT_LEN_TRAIN = 6
PROMPT_LENS_EVAL = [4, 6, 8, 10, 12]
MAX_SEQ_LEN = 2 * max(PROMPT_LENS_EVAL)
BATCH = 32
STEPS = 400
LR = 3e-3
N_LOOPS = 3
EVAL_BATCHES = 16

torch.manual_seed(0)


def make_batch(rng: torch.Generator, prompt_len: int, batch: int = BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = torch.randint(0, VOCAB, (batch, prompt_len), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def reverse_acc(model: OpenMythos, prompt_len: int, rng: torch.Generator) -> tuple[float, float]:
    """Return (accuracy on the reverse half, mean cross-entropy on full sequence)."""
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            inputs, targets = make_batch(rng, prompt_len)
            logits = model(inputs, n_loops=N_LOOPS)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
            losses.append(loss.item())
            preds = logits.argmax(dim=-1)
            t = targets[:, prompt_len - 1 : prompt_len - 1 + prompt_len]
            p = preds[:, prompt_len - 1 : prompt_len - 1 + prompt_len]
            correct += (p == t).sum().item()
            total += t.numel()
    return correct / max(1, total), sum(losses) / len(losses)


def main() -> None:
    print("[extrap] sequence-length extrapolation (prompt-then-reverse)")
    print(f"[extrap] train prompt_len={PROMPT_LEN_TRAIN} (seq={2*PROMPT_LEN_TRAIN}), "
          f"eval prompt_lens={PROMPT_LENS_EVAL}, n_loops={N_LOOPS}")

    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = MAX_SEQ_LEN
    cfg.max_loop_iters = N_LOOPS
    model = OpenMythos(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[extrap] params={n_params:,}, max_seq_len={MAX_SEQ_LEN}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    rng_train = torch.Generator().manual_seed(1)
    rng_eval = torch.Generator().manual_seed(99)

    t0 = time.time()
    model.train()
    for step in range(STEPS):
        inputs, targets = make_batch(rng_train, PROMPT_LEN_TRAIN)
        logits = model(inputs, n_loops=N_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    print(f"[extrap] trained {STEPS} steps in {time.time()-t0:.1f}s, final_loss={loss.item():.4f}")

    base_acc, base_ce = reverse_acc(model, PROMPT_LEN_TRAIN, rng_eval)
    print(f"[extrap] eval @ training len {PROMPT_LEN_TRAIN}: rev_acc={base_acc*100:.2f}%  ce={base_ce:.4f}")

    print("\n  prompt_len  seq_len  rev_acc    ce  delta_acc_vs_train")
    for pl in PROMPT_LENS_EVAL:
        acc, ce = reverse_acc(model, pl, rng_eval)
        delta = (acc - base_acc) * 100
        print(f"  {pl:10d}  {2*pl:7d}  {acc*100:6.2f}%  {ce:6.3f}  {delta:+7.2f}pp")

    long_pl = max(PROMPT_LENS_EVAL)
    long_acc, _ = reverse_acc(model, long_pl, rng_eval)
    assert long_acc == long_acc, "NaN accuracy at long sequence"
    print(f"\n[extrap] OK  (longest prompt_len={long_pl}, seq={2*long_pl}, acc={long_acc*100:.2f}%)")


if __name__ == "__main__":
    main()
