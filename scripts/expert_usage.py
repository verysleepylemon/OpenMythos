"""Expert routing utilization analysis.

Hooks every ``MoEFFN`` module in the model, captures the top-K expert IDs
selected per token at every forward pass, and reports the empirical
distribution after training.

The router bias is updated externally by the training loop in upstream
OpenMythos, but in this script we don't run that bias update — we just
measure what the *raw router* learns to do on a tiny toy task. A roughly
uniform distribution means the load-balancing pressure is healthy; one
expert collapsing to 0% utilization (or 100%) would indicate routing
failure.

Run:

    python scripts/expert_usage.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts._common import OpenMythos, build_tiny_mla_config

VOCAB = 32
PROMPT_LEN = 6
SEQ_LEN = 12
BATCH = 32
STEPS = 200
LR = 3e-3
N_LOOPS = 3
N_EXPERTS = 4
TOPK = 2
SEED = 0


def make_batch(rng: torch.Generator, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = torch.randint(0, VOCAB, (batch, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def build_model() -> OpenMythos:
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = N_LOOPS
    cfg.expert_dim = 32
    cfg.num_experts = N_EXPERTS
    cfg.num_shared_experts = 1
    cfg.top_k = TOPK
    return OpenMythos(cfg)


def install_router_hooks(model: OpenMythos) -> dict[str, torch.Tensor]:
    """Patch each MoEFFN.router so we can count its top-K selections.

    Returns a dict mapping a stable module-id string to a counter tensor of
    shape (N_EXPERTS,) that gets incremented on every forward.
    """
    counters: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if type(module).__name__ != "MoEFFN":
            continue
        counter = torch.zeros(N_EXPERTS, dtype=torch.long)
        counters[name] = counter
        original_router = module.router

        def make_hook(c: torch.Tensor, m: torch.nn.Module):
            def hook(_module, _inp, out):
                # out shape: (B*T, n_experts)
                # We replicate the same top-K selection logic the MoE uses
                # internally so our counts match what the experts actually
                # ran on.
                _, topk_idx = (out + m.router_bias).topk(TOPK, dim=-1)
                flat = topk_idx.reshape(-1)
                bins = torch.bincount(flat, minlength=N_EXPERTS).to(c.dtype)
                c.add_(bins)

            return hook

        original_router.register_forward_hook(make_hook(counter, module))
    return counters


def train(model: OpenMythos) -> None:
    rng = torch.Generator().manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(1, STEPS + 1):
        x, y = make_batch(rng, BATCH)
        logits = model(x, n_loops=N_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 50 == 0 or step == STEPS:
            print(f"[expert] train step {step:3d}/{STEPS}  loss={loss.item():.4f}")


def report(counters: dict[str, torch.Tensor]) -> None:
    print()
    print(f"[expert] usage histograms after {STEPS} steps  "
          f"(top-{TOPK} of {N_EXPERTS} routed experts):")
    print()
    for name, c in counters.items():
        total = c.sum().item()
        if total == 0:
            print(f"[expert] {name}: NO ACTIVATIONS (something is wrong)")
            continue
        pct = (c.float() / total * 100).tolist()
        ideal = 100.0 / N_EXPERTS  # share-of-selections; sums to 100%
        bars = " ".join(f"E{i}={p:5.1f}%" for i, p in enumerate(pct))
        max_dev = max(abs(p - ideal) for p in pct)
        print(f"[expert] {name}  {bars}   max_dev_from_ideal={max_dev:.1f}pp")

    # Aggregate across all MoE layers.
    if counters:
        agg = torch.zeros(N_EXPERTS, dtype=torch.long)
        for c in counters.values():
            agg += c
        total = agg.sum().item()
        pct = (agg.float() / total * 100).tolist()
        ideal = 100.0 / N_EXPERTS
        max_dev = max(abs(p - ideal) for p in pct)
        bars = " ".join(f"E{i}={p:5.1f}%" for i, p in enumerate(pct))
        print()
        print(f"[expert] AGGREGATE  {bars}   ideal={ideal:.1f}%/expert  "
              f"max_dev={max_dev:.1f}pp")


def main() -> None:
    torch.manual_seed(SEED)
    model = build_model()
    n_moe_layers = sum(1 for m in model.modules() if type(m).__name__ == "MoEFFN")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[expert] params={n_params:,}  MoE layers found={n_moe_layers}  "
          f"experts={N_EXPERTS}  topk={TOPK}")
    if n_moe_layers == 0:
        raise SystemExit("[expert] no MoEFFN modules found; aborting")

    counters = install_router_hooks(model)
    train(model)
    report(counters)
    print()
    print("[expert] OK")


if __name__ == "__main__":
    main()
