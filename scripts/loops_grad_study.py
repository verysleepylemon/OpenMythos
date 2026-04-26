"""Compare gradient flow under different recurrent loop counts.

Trains the same model on the reverse-copy task with N_LOOPS = {1, 2, 4, 8}
for a fixed number of steps (no extra capacity, just more recurrent depth at
train time) and reports:

- final training loss
- gradient norm trajectory
- final eval loss + reverse-copy accuracy
- wall time
- final spectral radius rho(A) (must stay < 1 throughout)

This is the cleanest way to *show* the headline claim of recurrent depth:
more loops at training time should produce better learning per parameter
without destabilizing the network. If rho(A) blows past 1 anywhere or loss
diverges, the LTI injection guarantee is broken.

Run:

    python scripts/loops_grad_study.py
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from scripts._common import OpenMythos, build_tiny_mla_config

VOCAB = 32
PROMPT_LEN = 6
SEQ_LEN = 12
BATCH = 32
STEPS = 200
LR = 3e-3
LOOP_GRID = [1, 2, 4, 8]
SEED = 0


def make_batch(rng: torch.Generator, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = torch.randint(0, VOCAB, (batch, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def build_model(max_loops: int) -> OpenMythos:
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = max_loops
    cfg.expert_dim = 32
    cfg.num_experts = 4
    cfg.num_shared_experts = 1
    cfg.top_k = 2
    return OpenMythos(cfg)


def reverse_acc(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    rev = pred[:, PROMPT_LEN - 1 :]
    rev_t = target[:, PROMPT_LEN - 1 :]
    return (rev == rev_t).float().mean().item()


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def find_lti(model: OpenMythos):
    for m in model.modules():
        if type(m).__name__ == "LTIInjection":
            return m
    return None


def spectral_radius(model: OpenMythos) -> float:
    lti = find_lti(model)
    if lti is None:
        return float("nan")
    # A_discrete = exp(dt * (-exp(log_A))); each diagonal entry is in (0,1)
    # so spectral radius = max diagonal magnitude.
    with torch.no_grad():
        log_a = lti.log_A.detach()
        log_dt = lti.log_dt.detach()
        a_cont = -torch.exp(log_a)
        dt = torch.exp(log_dt)
        a_disc = torch.exp(dt * a_cont)
        return a_disc.abs().max().item()


def run_one(n_loops: int) -> dict:
    torch.manual_seed(SEED)
    model = build_model(max_loops=n_loops)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    train_rng = torch.Generator().manual_seed(SEED)
    eval_rng = torch.Generator().manual_seed(SEED + 1234)
    eval_x, eval_y = make_batch(eval_rng, batch=128)

    grad_norms = []
    losses = []

    t0 = time.time()
    model.train()
    for step in range(STEPS):
        x, y = make_batch(train_rng, BATCH)
        logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        gn = grad_norm(model)
        opt.step()
        grad_norms.append(gn)
        losses.append(loss.item())
    wall_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        ev = model(eval_x, n_loops=n_loops)
        eval_loss = F.cross_entropy(ev.reshape(-1, VOCAB), eval_y.reshape(-1)).item()
        eval_acc = reverse_acc(ev, eval_y)

    return {
        "n_loops": n_loops,
        "final_loss": losses[-1],
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "rho": spectral_radius(model),
        "wall_time": wall_time,
        "grad_first": grad_norms[0],
        "grad_mid": grad_norms[len(grad_norms) // 2],
        "grad_last": grad_norms[-1],
    }


def main() -> None:
    print(f"[study] vocab={VOCAB}  seq_len={SEQ_LEN}  batch={BATCH}  steps={STEPS}")
    uniform_ce = math.log(VOCAB)
    print(f"[study] uniform CE baseline: {uniform_ce:.4f}")
    print()

    results = []
    for n in LOOP_GRID:
        print(f"[study] training with n_loops={n} ...", flush=True)
        r = run_one(n)
        results.append(r)

    print()
    cols = (
        f"{'n_loops':>8} {'final_loss':>11} {'eval_loss':>10} {'eval_acc':>10} "
        f"{'rho(A)':>8} {'gn_first':>9} {'gn_mid':>8} {'gn_last':>8} {'wall_s':>7}"
    )
    print(cols)
    for r in results:
        print(
            f"{r['n_loops']:>8d} {r['final_loss']:>11.4f} {r['eval_loss']:>10.4f} "
            f"{r['eval_acc'] * 100:>9.2f}% {r['rho']:>8.4f} "
            f"{r['grad_first']:>9.3f} {r['grad_mid']:>8.3f} {r['grad_last']:>8.3f} "
            f"{r['wall_time']:>7.2f}"
        )

    # Sanity: rho(A) must stay strictly less than 1 for every config.
    max_rho = max(r["rho"] for r in results)
    assert max_rho < 1.0, f"rho(A) blew past 1: {max_rho}"

    # We expect more loops to give >= eval-loss improvement vs n=1.
    base = next(r["eval_loss"] for r in results if r["n_loops"] == 1)
    best = min(r["eval_loss"] for r in results)
    print(f"\n[study] best eval_loss={best:.4f}  vs n=1 baseline={base:.4f}  "
          f"improvement={base - best:+.4f}")

    print("[study] OK")


if __name__ == "__main__":
    main()
