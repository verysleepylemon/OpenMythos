"""Shared helpers for personal scripts.

Loads `open_mythos.main` directly via importlib to bypass the package
`__init__.py`, which eagerly imports `transformers` (a heavy dep we do not
need for the core model). Every personal script imports from here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_open_mythos_main() -> ModuleType:
    """Return the loaded `open_mythos.main` module without triggering the
    package `__init__.py`."""

    if "open_mythos_main" in sys.modules:
        return sys.modules["open_mythos_main"]

    main_path = Path(__file__).resolve().parent.parent / "open_mythos" / "main.py"
    spec = importlib.util.spec_from_file_location("open_mythos_main", main_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"could not load open_mythos.main from {main_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["open_mythos_main"] = mod
    spec.loader.exec_module(mod)
    return mod


_om = load_open_mythos_main()
MythosConfig = _om.MythosConfig
OpenMythos = _om.OpenMythos


def build_tiny_mla_config() -> "MythosConfig":
    """Tiny MLA-flavored config used by every smoke/bench script (~110k params)."""
    return MythosConfig(
        vocab_size=256,
        dim=64,
        n_heads=4,
        max_seq_len=64,
        max_loop_iters=4,
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


def build_tiny_gqa_config() -> "MythosConfig":
    """Same shape as tiny MLA but using GQA so attention variants are comparable."""
    return MythosConfig(
        vocab_size=256,
        dim=64,
        n_heads=4,
        max_seq_len=64,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        lora_rank=4,
        attn_type="gqa",
        n_kv_heads=2,
    )
