"""Tiny CPU smoke test for the OpenMythos Recurrent-Depth Transformer.

Smaller than the upstream `example.py` so it runs in under a second on CPU.
Prints parameter count, forward-pass shape, generated shape, and spectral radius
of the recurrent injection matrix (must be < 1 for the loop to be contractive).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from scripts._common import OpenMythos, build_tiny_mla_config  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    cfg = build_tiny_mla_config()
    cfg.max_loop_iters = 2
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
