"""Depth-extrapolation experiment.

Trains a tiny model at ``n_loops = TRAIN_LOOPS`` on the reverse-copy task,
then evaluates the *same trained weights* at a sweep of inference-time loop
counts ``[1, 2, 4, 8, 16, 32]``.

The point: because the recurrent injection is provably contractive
(``rho(A) < 1``), running more loops at inference than were used at training
is numerically stable — logits converge to a fixed point rather than blowing
up. This is the headline property of the recurrent-depth design.

Run:

    python scripts/depth_extrapolation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts._common import OpenMythos, build_tiny_mla_config  # noqa: E402

# ---------- task config ----------
VOCAB = 32
PROMPT_LEN = 6
SEQ_LEN = 12  # prompt + reverse(prompt)
BATCH = 32
STEPS = 200
LR = 3e-3
TRAIN_LOOPS = 3
EVAL_LOOPS = [1, 2, 4, 8, 16, 32]
SEED = 0


def make_batch(rng: torch.Generator, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = torch.randint(0, VOCAB, (batch, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def reverse_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    rev = pred[:, PROMPT_LEN - 1 :]
    rev_t = target[:, PROMPT_LEN - 1 :]
    return (rev == rev_t).float().mean().item()


def build_model() -> tuple[OpenMythos, "object"]:
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = TRAIN_LOOPS
    cfg.expert_dim = 32
    cfg.num_experts = 4
    cfg.num_shared_experts = 1
    cfg.top_k = 2
    return OpenMythos(cfg), cfg


def train(model: OpenMythos) -> None:
    rng = torch.Generator().manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(1, STEPS + 1):
        x, y = make_batch(rng, BATCH)
        logits = model(x, n_loops=TRAIN_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step == STEPS:
            print(f"[depth] train step {step:3d}/{STEPS}  loss={loss.item():.4f}")


def evaluate_at(model: OpenMythos, n_loops: int, eval_x: torch.Tensor, eval_y: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(eval_x, n_loops=n_loops)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), eval_y.reshape(-1)).item()
        rev_acc = reverse_accuracy(logits, eval_y)
        # Logit summary statistics — if loops blow up, std would explode.
        mean_logit = logits.mean().item()
        std_logit = logits.std().item()
        max_abs = logits.abs().max().item()
    return {
        "n_loops": n_loops,
        "loss": loss,
        "rev_acc": rev_acc,
        "mean": mean_logit,
        "std": std_logit,
        "max_abs": max_abs,
    }


def kl_against(reference: torch.Tensor, other: torch.Tensor) -> float:
    p = F.log_softmax(reference, dim=-1)
    q = F.log_softmax(other, dim=-1)
    # KL(reference || other) averaged over batch and position.
    return F.kl_div(q, p, reduction="batchmean", log_target=True).item()


def main() -> None:
    torch.manual_seed(SEED)

    model, cfg = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[depth] params={n_params:,}  vocab={VOCAB}  seq_len={SEQ_LEN}  "
        f"train_loops={TRAIN_LOOPS}"
    )
    print(f"[depth] uniform CE baseline: {torch.log(torch.tensor(VOCAB)).item():.4f}")

    t0 = time.time()
    train(model)
    train_time = time.time() - t0
    print(f"[depth] training wall_time: {train_time:.1f}s")

    # Fixed eval batch so all loop counts see identical inputs.
    eval_rng = torch.Generator().manual_seed(SEED + 1234)
    eval_x, eval_y = make_batch(eval_rng, batch=128)

    print()
    print(f"{'n_loops':>8} {'loss':>10} {'rev_acc':>10} {'std':>10} {'max|l|':>10}")
    rows = []
    for n in EVAL_LOOPS:
        r = evaluate_at(model, n, eval_x, eval_y)
        rows.append(r)
        print(
            f"{r['n_loops']:>8d} {r['loss']:>10.4f} "
            f"{r['rev_acc'] * 100:>9.2f}% {r['std']:>10.4f} {r['max_abs']:>10.4f}"
        )

    # Stability assertions — these are the headline guarantees.
    max_std = max(r["std"] for r in rows)
    max_abs = max(r["max_abs"] for r in rows)
    assert max_std < 100.0, f"logit std blew up: {max_std}"
    assert max_abs < 1000.0, f"logit max abs blew up: {max_abs}"

    # KL convergence — should approach 0 between successive loop doublings.
    print()
    print("[depth] KL between successive (doubling) loop counts:")
    model.eval()
    with torch.no_grad():
        prev_logits = None
        prev_n = None
        for n in EVAL_LOOPS:
            logits = model(eval_x, n_loops=n)
            if prev_logits is not None:
                kl = kl_against(prev_logits, logits)
                print(f"[depth]   KL(n={prev_n} || n={n}) = {kl:.6f}")
            prev_logits = logits
            prev_n = n

    print()
    print("[depth] OK")


if __name__ == "__main__":
    main()
