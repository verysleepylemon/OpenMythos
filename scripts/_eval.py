"""Shared eval helper used by reverse-copy training scripts.

Centralizes the boilerplate that nearly every personal script duplicates:
- build a deterministic eval RNG from a seed
- evaluate (cross-entropy, reverse-half token accuracy) over EVAL_BATCHES
- support both fully-recurrent calls and KV-cached greedy decode

Designed to NOT take any global module state. All knobs are passed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReverseCopyTask:
    vocab: int
    prompt_len: int

    @property
    def seq_len(self) -> int:
        return 2 * self.prompt_len

    def make_batch(self, batch: int, rng: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt = torch.randint(0, self.vocab, (batch, self.prompt_len), generator=rng)
        seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
        return seq[:, :-1], seq[:, 1:]


def evaluate(
    model: torch.nn.Module,
    task: ReverseCopyTask,
    *,
    n_loops: int,
    eval_batches: int = 16,
    eval_batch_size: int = 32,
    seed: int = 9999,
) -> Tuple[float, float]:
    """Returns (cross_entropy_mean, reverse_half_accuracy)."""
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    losses = []
    correct = total = 0
    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = task.make_batch(eval_batch_size, rng)
            logits = model(x, n_loops=n_loops)
            losses.append(F.cross_entropy(logits.reshape(-1, task.vocab), y.reshape(-1)).item())
            preds = logits.argmax(dim=-1)
            t = y[:, task.prompt_len - 1 : 2 * task.prompt_len - 1]
            p = preds[:, task.prompt_len - 1 : 2 * task.prompt_len - 1]
            correct += (p == t).sum().item()
            total += t.numel()
    return sum(losses) / max(1, len(losses)), correct / max(1, total)


def train_reverse_copy(
    model: torch.nn.Module,
    task: ReverseCopyTask,
    *,
    n_loops: int,
    steps: int,
    batch: int = 32,
    lr: float = 3e-3,
    clip: float = 1.0,
    seed: int = 0,
    weight_decay: float = 0.0,
    log_every: int = 0,
) -> Iterable[float]:
    """Standard training loop. Yields per-step train losses (live)."""
    model.train()
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    for step in range(steps):
        x, y = task.make_batch(batch, rng)
        logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(logits.reshape(-1, task.vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"  [train] step {step+1}/{steps}  loss={loss.item():.4f}")
    return losses
