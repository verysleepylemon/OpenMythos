"""Adaptive early-exit at inference: halt loops once logits converge.

The recurrent block is contractive (rho(A) < 1) so logits converge to a
fixed point as n_loops grows. In practice that fixed point is reached well
before ``cfg.max_loop_iters`` for many inputs. This script measures the
inference compute saving you get by halting loops as soon as the change
between successive loop outputs drops below a threshold.

Procedure:

1. Train a tiny model briefly on the reverse-copy task.
2. For each input, run the model with n_loops in [1, 2, ... MAX_LOOPS] and
   record the KL divergence between successive iterations.
3. For each early-exit threshold ``eps``, find the smallest n_loops where
   KL drops below ``eps``. Compare the resulting loss + accuracy against
   running the full ``MAX_LOOPS`` always.

Run:

    python scripts/early_exit.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts._common import OpenMythos, build_tiny_mla_config

VOCAB = 32
PROMPT_LEN = 6
SEQ_LEN = 12
BATCH_TRAIN = 32
BATCH_EVAL = 256
STEPS = 200
LR = 3e-3
TRAIN_LOOPS = 4
MAX_LOOPS = 8
SEED = 0

EPS_GRID = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def make_batch(rng: torch.Generator, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = torch.randint(0, VOCAB, (batch, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def build_model() -> OpenMythos:
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = MAX_LOOPS
    cfg.expert_dim = 32
    cfg.num_experts = 4
    cfg.num_shared_experts = 1
    cfg.top_k = 2
    return OpenMythos(cfg)


def train(model: OpenMythos) -> None:
    rng = torch.Generator().manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(1, STEPS + 1):
        x, y = make_batch(rng, BATCH_TRAIN)
        logits = model(x, n_loops=TRAIN_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (1, STEPS):
            print(f"[exit] train step {step:3d}/{STEPS}  loss={loss.item():.4f}")


def loop_logit_trajectory(model: OpenMythos, x: torch.Tensor) -> list[torch.Tensor]:
    """Return logits at n_loops = 1, 2, ..., MAX_LOOPS for the same input."""
    model.eval()
    out = []
    with torch.no_grad():
        for n in range(1, MAX_LOOPS + 1):
            out.append(model(x, n_loops=n))
    return out


def per_token_kl_against_prev(
    cur: torch.Tensor, prev: torch.Tensor
) -> torch.Tensor:
    """Mean-over-vocab KL(prev || cur) per (batch, position). Shape: (B, T)."""
    log_p = F.log_softmax(prev, dim=-1)
    log_q = F.log_softmax(cur, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


def reverse_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    rev = pred[:, PROMPT_LEN - 1 :]
    rev_t = target[:, PROMPT_LEN - 1 :]
    return (rev == rev_t).float().mean().item()


def main() -> None:
    torch.manual_seed(SEED)
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[exit] params={n_params:,}  train_loops={TRAIN_LOOPS}  "
          f"max_loops={MAX_LOOPS}")

    train(model)

    eval_rng = torch.Generator().manual_seed(SEED + 1234)
    x, y = make_batch(eval_rng, BATCH_EVAL)

    trajectory = loop_logit_trajectory(model, x)
    final_logits = trajectory[-1]
    final_loss = F.cross_entropy(
        final_logits.reshape(-1, VOCAB), y.reshape(-1)
    ).item()
    final_acc = reverse_accuracy(final_logits, y)
    print(f"[exit] full-budget (n_loops={MAX_LOOPS})  "
          f"loss={final_loss:.4f}  rev_acc={final_acc * 100:.2f}%")
    print()

    # KL between successive iterations, averaged over the eval batch.
    print(f"[exit] mean KL between successive loop iterations:")
    for n in range(2, MAX_LOOPS + 1):
        kl = per_token_kl_against_prev(trajectory[n - 1], trajectory[n - 2])
        print(f"[exit]   KL(n={n - 1} || n={n}) = {kl.mean().item():.6f}")
    print()

    # For each threshold eps, decide *per-token* what loop count to halt at,
    # then assemble the early-exit logits by picking each (b, t) position
    # from its halt step.
    print(f"{'eps':>10} {'avg_loops':>11} {'frac_full':>10} {'loss':>10} {'rev_acc':>10}")
    for eps in EPS_GRID:
        # halt_step[b, t] = smallest n in [2..MAX_LOOPS] where KL drops below
        # eps; if it never drops below eps we use MAX_LOOPS.
        halt = torch.full((x.shape[0], x.shape[1]), MAX_LOOPS, dtype=torch.long)
        # Fill from MAX_LOOPS backwards so the first match wins.
        for n in range(MAX_LOOPS, 1, -1):
            kl = per_token_kl_against_prev(trajectory[n - 1], trajectory[n - 2])
            mask = kl < eps
            halt[mask] = n

        # Build the halted-logits tensor by gathering from each step.
        ee_logits = torch.empty_like(final_logits)
        for n in range(1, MAX_LOOPS + 1):
            mask = halt == n
            if mask.any():
                ee_logits[mask] = trajectory[n - 1][mask]

        ee_loss = F.cross_entropy(
            ee_logits.reshape(-1, VOCAB), y.reshape(-1)
        ).item()
        ee_acc = reverse_accuracy(ee_logits, y)
        avg_loops = halt.float().mean().item()
        frac_full = (halt == MAX_LOOPS).float().mean().item()
        print(f"{eps:>10.0e} {avg_loops:>11.2f} {frac_full * 100:>9.1f}% "
              f"{ee_loss:>10.4f} {ee_acc * 100:>9.2f}%")

    print()
    print("[exit] OK")


if __name__ == "__main__":
    main()
