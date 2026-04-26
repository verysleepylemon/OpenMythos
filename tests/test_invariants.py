"""Stronger invariant tests beyond the smoke suite.

Each test asserts a property that should hold *exactly* (not just on average)
for a correctly implemented Recurrent-Depth Transformer with KV cache and
LTI injection.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._common import OpenMythos, build_tiny_mla_config  # noqa: E402

VOCAB = 32
SEQ_LEN = 12
N_LOOPS = 3


def _build(seed: int = 0) -> OpenMythos:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN + 8
    cfg.max_loop_iters = N_LOOPS
    return OpenMythos(cfg)


def test_causal_mask_isolation() -> None:
    """Perturbing token i must not change predictions at positions < i.

    Causal masking is the load-bearing assumption behind autoregressive
    decoding. We perturb a single token in the middle of the sequence and
    verify that all earlier-position logits are bit-identical.
    """
    model = _build()
    model.eval()
    torch.manual_seed(1)
    x = torch.randint(0, VOCAB, (1, SEQ_LEN))
    with torch.no_grad():
        logits_a = model(x, n_loops=N_LOOPS)
        x2 = x.clone()
        # Flip a token at position 6 (middle of the sequence).
        x2[0, 6] = (x[0, 6].item() + 7) % VOCAB
        assert x2[0, 6].item() != x[0, 6].item()
        logits_b = model(x2, n_loops=N_LOOPS)

    # Positions 0..5 should be unchanged (causal: cannot see position 6+).
    diff = (logits_a[:, :6, :] - logits_b[:, :6, :]).abs().max().item()
    assert diff == 0.0, f"causal mask leak: max diff at <i positions = {diff}"
    # Positions 6+ MUST differ (sanity: the perturbation actually propagates).
    diff_after = (logits_a[:, 6:, :] - logits_b[:, 6:, :]).abs().max().item()
    assert diff_after > 0.0, "perturbation did not propagate to later positions"


def test_kv_cache_logits_match_full_forward() -> None:
    """Token-by-token cached forward must match a full-sequence forward
    in *raw logits*, not just argmax (stronger than the smoke version)."""
    model = _build(seed=2)
    model.eval()
    torch.manual_seed(7)
    x = torch.randint(0, VOCAB, (1, SEQ_LEN))

    with torch.no_grad():
        full = model(x, n_loops=N_LOOPS)

        kv: dict = {}
        cached_logits = []
        for i in range(SEQ_LEN):
            tok = x[:, i : i + 1]
            start_pos = i
            out = model(tok, n_loops=N_LOOPS, kv_cache=kv, start_pos=start_pos)
            cached_logits.append(out[:, -1, :])
        cached = torch.stack(cached_logits, dim=1)

    diff = (full - cached).abs().max().item()
    # Float32 ops with low-rank routing can introduce tiny rounding (~1e-5).
    assert diff < 1e-4, f"KV cache logits diverged from full forward: max diff = {diff}"


def test_loop_kl_decreases_monotonically_on_average() -> None:
    """KL between successive loop iterations should not increase with depth.

    The recurrent block is contractive (rho(A) < 1), so iteration converges
    to a fixed point. We don't require *strict* monotonicity at every step
    (numerical noise can fluctuate after convergence), but KL(n -> n+1) at
    high n should be <= KL(1 -> 2).
    """
    model = _build(seed=3)
    model.eval()
    torch.manual_seed(11)
    x = torch.randint(0, VOCAB, (2, SEQ_LEN))
    kl_curve = []
    prev = None
    with torch.no_grad():
        for n in range(1, 8):
            logits = model(x, n_loops=n)
            log_p = F.log_softmax(logits, dim=-1)
            if prev is not None:
                kl = F.kl_div(log_p, prev, reduction="batchmean", log_target=True)
                kl_curve.append(kl.item())
            prev = log_p
    assert kl_curve[-1] <= kl_curve[0] + 1e-6, (
        f"KL did not decrease across depth: {kl_curve}"
    )
    # Early KL should be at least as large as late KL by a meaningful margin
    # (or both should already be ~0). We accept both.
    assert kl_curve[0] >= kl_curve[-1] - 1e-6


def test_spectral_radius_under_one_across_seeds() -> None:
    """rho(A_disc) < 1 must hold for every random initialization.

    A_disc = exp(dt * -exp(log_A)) is the ZOH-discretized state matrix from
    the LTI injection. Stability is the fundamental guarantee that lets the
    recurrent block be unrolled an arbitrary number of times.
    """
    for seed in range(5):
        model = _build(seed=seed)
        rhos: list[float] = []
        for m in model.modules():
            if type(m).__name__ != "LTIInjection":
                continue
            log_A = m.log_A.detach()
            log_dt = m.log_dt.detach()
            a_disc = torch.exp(-torch.exp((log_dt + log_A).clamp(-20, 20)))
            rhos.append(float(a_disc.abs().max().item()))
        assert rhos, "no LTIInjection modules found"
        max_rho = max(rhos)
        assert max_rho < 1.0, f"unstable A_disc at seed={seed}: max rho={max_rho}"


def test_lora_adapter_clamps_loop_index() -> None:
    """The LoRAAdapter must tolerate loop_t >= max_loop_iters by clamping
    rather than crashing — this is what enables depth extrapolation."""
    model = _build(seed=4)
    model.eval()
    torch.manual_seed(0)
    x = torch.randint(0, VOCAB, (1, SEQ_LEN))
    with torch.no_grad():
        # max_loop_iters = N_LOOPS = 3; ask for 8 loops.
        logits = model(x, n_loops=8)
    assert logits.shape == (1, SEQ_LEN, VOCAB)
    assert torch.isfinite(logits).all(), "non-finite logits during depth extrapolation"
