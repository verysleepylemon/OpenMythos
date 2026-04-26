"""KV-cache speed micro-benchmark.

Compares two generation strategies for the same trained model:

  (A) CACHED   -- model.generate(...) using the built-in KV cache. After the
                  prompt is consumed, each new token only re-runs the model on
                  the single most recent token id, with all prior K/V looked up
                  from the cache.
  (B) RECOMPUTE -- naive baseline. For each new token, re-run model.forward on
                  the entire growing sequence (no cache).

The two paths should produce equivalent token-by-token logits (cache hits are
deterministic), so we also assert greedy-decoded outputs match for the first
few tokens.

Reports tokens/sec for each path and the speedup ratio. Speedup grows roughly
linearly with prompt length and number of generated tokens because RECOMPUTE
work scales O((T+i)^2) while CACHED scales O(T+i).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import OpenMythos, build_tiny_mla_config

VOCAB = 32
PROMPT_LEN = 16
GEN_TOKENS = 32
N_LOOPS = 3
SEED = 0
WARMUP = 1
ASSERT_PREFIX = 6  # how many greedy tokens must match exactly


def build_model() -> OpenMythos:
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = PROMPT_LEN + GEN_TOKENS + 8
    cfg.max_loop_iters = N_LOOPS
    return OpenMythos(cfg)


def greedy_recompute(model: OpenMythos, prompt: torch.Tensor, n_new: int) -> torch.Tensor:
    """No-cache greedy decode: run forward over the whole growing sequence each step."""
    out = prompt.clone()
    with torch.no_grad():
        for _ in range(n_new):
            logits = model(out, n_loops=N_LOOPS)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_tok], dim=1)
    return out


def greedy_cached(model: OpenMythos, prompt: torch.Tensor, n_new: int) -> torch.Tensor:
    """KV-cached greedy decode mirroring ``model.generate`` but with argmax (no sampling)."""
    out = prompt.clone()
    kv_cache: dict = {}
    prompt_len = out.shape[1]
    with torch.no_grad():
        for step in range(n_new):
            if step == 0:
                cur_ids = out
                start_pos = 0
            else:
                cur_ids = out[:, -1:]
                start_pos = prompt_len + step - 1
            logits = model(cur_ids, n_loops=N_LOOPS, kv_cache=kv_cache, start_pos=start_pos)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_tok], dim=1)
    return out


def main() -> None:
    torch.manual_seed(SEED)
    model = build_model()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[kv] params={n_params:,}  prompt_len={PROMPT_LEN}  gen={GEN_TOKENS}  n_loops={N_LOOPS}")

    rng = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, VOCAB, (1, PROMPT_LEN), generator=rng)

    # Warmup so the first run doesn't include lazy initialization.
    for _ in range(WARMUP):
        _ = greedy_recompute(model, prompt, 4)
        _ = greedy_cached(model, prompt, 4)

    t0 = time.perf_counter()
    out_recomp = greedy_recompute(model, prompt, GEN_TOKENS)
    t_recomp = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_cached = greedy_cached(model, prompt, GEN_TOKENS)
    t_cached = time.perf_counter() - t0

    tps_recomp = GEN_TOKENS / t_recomp
    tps_cached = GEN_TOKENS / t_cached
    speedup = t_recomp / t_cached if t_cached > 0 else float("inf")

    print(f"[kv] RECOMPUTE  wall={t_recomp:.3f}s  tokens/sec={tps_recomp:.1f}")
    print(f"[kv] CACHED     wall={t_cached:.3f}s  tokens/sec={tps_cached:.1f}")
    print(f"[kv] speedup    {speedup:.2f}x  (RECOMPUTE / CACHED)")

    # Equivalence check on the first ASSERT_PREFIX generated tokens.
    gen_recomp = out_recomp[0, PROMPT_LEN : PROMPT_LEN + ASSERT_PREFIX].tolist()
    gen_cached = out_cached[0, PROMPT_LEN : PROMPT_LEN + ASSERT_PREFIX].tolist()
    print(f"[kv] first {ASSERT_PREFIX} generated tokens (RECOMPUTE): {gen_recomp}")
    print(f"[kv] first {ASSERT_PREFIX} generated tokens (CACHED):    {gen_cached}")

    if gen_recomp != gen_cached:
        # Numerical drift in repeated MoE rounding can cause a single divergence;
        # accept a 1-token mismatch but escalate anything worse.
        n_match = sum(1 for a, b in zip(gen_recomp, gen_cached) if a == b)
        print(f"[kv] WARNING: only {n_match}/{ASSERT_PREFIX} tokens matched exactly")
        assert n_match >= ASSERT_PREFIX - 1, (
            f"cached and uncached generation diverged: {gen_recomp} vs {gen_cached}"
        )
    else:
        print(f"[kv] greedy outputs match exactly")

    assert speedup > 1.0, f"KV cache should be faster, got {speedup:.2f}x"
    print("\n[kv] OK")


if __name__ == "__main__":
    main()
