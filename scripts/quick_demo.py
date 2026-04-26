"""Tiny CPU smoke test for the OpenMythos Recurrent-Depth Transformer.

Smaller than the upstream `example.py` so it runs in under a second on CPU.
Prints parameter count, forward-pass shape, generated shape, and spectral radius
of the recurrent injection matrix (must be < 1 for the loop to be contractive).
"""

from __future__ import annotations

import torch

from open_mythos.main import MythosConfig, OpenMythos


def build_tiny_mla_config() -> MythosConfig:
    return MythosConfig(
        vocab_size=256,
        dim=64,
        n_heads=4,
        max_seq_len=32,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        lora_rank=4,
        attn_type="mla",
        n_kv_heads=4,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )


def main() -> None:
    torch.manual_seed(0)
    cfg = build_tiny_mla_config()
    model = OpenMythos(cfg)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[quick_demo] params: {n_params:,}")

    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits = model(ids, n_loops=2)
    print(f"[quick_demo] logits shape: {tuple(logits.shape)}")

    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=4, n_loops=2)
    print(f"[quick_demo] generated shape: {tuple(out.shape)}")

    a_max = model.recurrent.injection.get_A().max().item()
    print(f"[quick_demo] spectral radius rho(A) max: {a_max:.4f} (must be < 1)")
    assert a_max < 1.0, "Recurrent injection is not contractive"
    print("[quick_demo] OK")


if __name__ == "__main__":
    main()
