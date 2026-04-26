"""Train the tiny OpenMythos on a synthetic 'reverse the prompt' copy task.

Goal: prove the Recurrent-Depth Transformer + LTI injection + MoE FFN actually
learns on CPU in seconds. Each example is a sequence of random tokens, where
the target at position t is the token at position (T - 1 - t) of the *prompt*
half. A model that learns the task should drive cross-entropy below uniform
entropy = log(vocab_size).

Output: per-step loss, final accuracy on a held-out batch, and the final
spectral radius of the recurrent injection (must remain < 1 to stay stable).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts._common import OpenMythos, build_tiny_mla_config  # noqa: E402

VOCAB = 32  # small vocab makes the task learnable in seconds
PROMPT_LEN = 6
SEQ_LEN = PROMPT_LEN * 2  # prompt followed by its reverse
BATCH = 32
STEPS = 200
LR = 3e-3
N_LOOPS = 3


def make_batch(rng: torch.Generator, batch: int = BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (input_ids, target_ids) where target = reverse of the prompt half."""
    prompt = torch.randint(0, VOCAB, (batch, PROMPT_LEN), generator=rng)
    reverse = prompt.flip(dims=[1])
    seq = torch.cat([prompt, reverse], dim=1)
    # Predict next token: input = seq[:, :-1], target = seq[:, 1:]
    return seq[:, :-1], seq[:, 1:]


def main() -> None:
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(0)

    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = N_LOOPS
    model = OpenMythos(cfg)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params: {n_params:,}  vocab={VOCAB}  seq_len={SEQ_LEN}  n_loops={N_LOOPS}")
    uniform_ce = math.log(VOCAB)
    print(f"[train] uniform CE baseline: {uniform_ce:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    # ----- training loop --------------------------------------------------
    t0 = time.time()
    last_loss = float("nan")
    losses: list[float] = []
    for step in range(1, STEPS + 1):
        x, y = make_batch(rng)
        logits = model(x, n_loops=N_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        last_loss = loss.item()
        losses.append(last_loss)
        if step == 1 or step % 20 == 0 or step == STEPS:
            print(f"[train] step {step:>3}/{STEPS}  loss={last_loss:.4f}")
    dt = time.time() - t0

    # ----- evaluation -----------------------------------------------------
    model.eval()
    eval_rng = torch.Generator().manual_seed(123)
    x, y = make_batch(eval_rng, batch=128)
    with torch.no_grad():
        logits = model(x, n_loops=N_LOOPS)
        eval_loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item()
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
        # Accuracy on just the *reverse* half (the part that requires solving the task)
        reverse_preds = preds[:, PROMPT_LEN - 1 :]
        reverse_targets = y[:, PROMPT_LEN - 1 :]
        reverse_acc = (reverse_preds == reverse_targets).float().mean().item()

    rho = model.recurrent.injection.get_A().abs().max().item()

    print()
    print(f"[train] wall_time:        {dt:.1f}s")
    print(f"[train] first_loss:       {losses[0]:.4f}")
    print(f"[train] final_loss:       {last_loss:.4f}")
    print(f"[train] eval_loss:        {eval_loss:.4f}  (uniform={uniform_ce:.4f})")
    print(f"[train] eval_acc_full:    {acc * 100:.2f}%")
    print(f"[train] eval_acc_reverse: {reverse_acc * 100:.2f}%  (random={100/VOCAB:.2f}%)")
    print(f"[train] rho(A) final:     {rho:.4f}  (must be < 1)")

    assert eval_loss < uniform_ce, "model failed to learn anything"
    assert rho < 1.0, "recurrent injection became non-contractive"
    print("[train] OK")


if __name__ == "__main__":
    main()
