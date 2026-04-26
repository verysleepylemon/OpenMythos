"""Benchmark MLA vs GQA attention on the tiny config (CPU).

Compares forward latency, generate latency (KV-cache enabled), and parameter
count for two otherwise-identical OpenMythos configs. MLA uses low-rank Q/KV
projections + decoupled RoPE; GQA uses standard grouped-query attention.

Useful as a sanity check that both attention paths are wired correctly and
to see the parameter-count tradeoff at this scale.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from scripts._common import (  # noqa: E402
    OpenMythos,
    build_tiny_gqa_config,
    build_tiny_mla_config,
)

WARMUP = 3
TRIALS = 20
SEQ_LEN = 16
GEN_TOKENS = 16
N_LOOPS = 3


def time_forward(model: OpenMythos, x: torch.Tensor) -> float:
    for _ in range(WARMUP):
        with torch.no_grad():
            model(x, n_loops=N_LOOPS)
    t0 = time.perf_counter()
    for _ in range(TRIALS):
        with torch.no_grad():
            model(x, n_loops=N_LOOPS)
    return (time.perf_counter() - t0) / TRIALS


def time_generate(model: OpenMythos, x: torch.Tensor) -> float:
    for _ in range(WARMUP):
        with torch.no_grad():
            model.generate(x, max_new_tokens=GEN_TOKENS, n_loops=N_LOOPS, top_k=1)
    t0 = time.perf_counter()
    for _ in range(TRIALS):
        with torch.no_grad():
            model.generate(x, max_new_tokens=GEN_TOKENS, n_loops=N_LOOPS, top_k=1)
    return (time.perf_counter() - t0) / TRIALS


def bench(name: str, cfg) -> None:
    torch.manual_seed(0)
    model = OpenMythos(cfg)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN))

    fwd_ms = time_forward(model, x) * 1000
    gen_ms = time_generate(model, x) * 1000

    print(
        f"[bench] {name:>5}  params={n_params:>7,}  "
        f"forward={fwd_ms:>7.2f}ms  "
        f"generate({GEN_TOKENS}tok)={gen_ms:>7.2f}ms"
    )


def main() -> None:
    print(
        f"[bench] seq_len={SEQ_LEN}  gen_tokens={GEN_TOKENS}  "
        f"n_loops={N_LOOPS}  warmup={WARMUP}  trials={TRIALS}"
    )
    bench("MLA", build_tiny_mla_config())
    bench("GQA", build_tiny_gqa_config())


if __name__ == "__main__":
    main()
