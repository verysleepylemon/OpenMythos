"""Pytest smoke tests for OpenMythos that do not require the optional
`transformers` / `datasets` dependencies.

These mirror the personal scripts under `scripts/` and are designed to run in
CI on a fresh CPU-only torch install in seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Make `from scripts._common import ...` work whether pytest is run from the
# repo root or from inside tests/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._common import (  # noqa: E402
    OpenMythos,
    build_tiny_gqa_config,
    build_tiny_mla_config,
)

VOCAB = 32
PROMPT_LEN = 6
SEQ_LEN = PROMPT_LEN * 2


def _tiny_train_config():
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = 3
    return cfg


def _reverse_batch(rng: torch.Generator, batch: int):
    prompt = torch.randint(0, VOCAB, (batch, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def test_mla_forward_shape():
    torch.manual_seed(0)
    cfg = build_tiny_mla_config()
    model = OpenMythos(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        logits = model(x, n_loops=2)
    assert logits.shape == (2, 8, cfg.vocab_size)


def test_gqa_forward_shape():
    torch.manual_seed(0)
    cfg = build_tiny_gqa_config()
    model = OpenMythos(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        logits = model(x, n_loops=2)
    assert logits.shape == (2, 8, cfg.vocab_size)


def test_recurrent_injection_is_contractive():
    torch.manual_seed(0)
    model = OpenMythos(build_tiny_mla_config()).eval()
    rho = model.recurrent.injection.get_A().abs().max().item()
    assert rho < 1.0, f"injection not contractive: rho={rho}"


def test_generate_extends_sequence():
    torch.manual_seed(0)
    cfg = build_tiny_mla_config()
    model = OpenMythos(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=5, n_loops=2, top_k=1)
    assert out.shape == (1, 4 + 5)
    assert torch.equal(out[:, :4], x)


def test_loop_convergence_kl_decreases():
    """Successive loop iterations should produce KL between consecutive
    distributions that trends toward zero (contractive fixed point)."""
    torch.manual_seed(0)
    cfg = build_tiny_mla_config()
    cfg.max_loop_iters = 8
    model = OpenMythos(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    prev = None
    kls = []
    for n in (1, 2, 3, 4, 6, 8):
        with torch.no_grad():
            probs = F.softmax(model(x, n_loops=n), dim=-1)
        if prev is not None:
            kls.append(F.kl_div(probs.log(), prev, reduction="batchmean").item())
        prev = probs
    assert kls[-1] <= kls[0] + 1e-6, f"KL did not converge: {kls}"
    assert kls[-1] < 1e-3, f"final KL too large: {kls[-1]}"


def test_model_learns_reverse_copy():
    """End-to-end: 200 AdamW steps must drop CE below uniform baseline and
    push reverse-half accuracy clearly above random."""
    import math

    torch.manual_seed(0)
    cfg = _tiny_train_config()
    model = OpenMythos(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = torch.Generator().manual_seed(0)

    model.train()
    for _ in range(200):
        x, y = _reverse_batch(rng, 32)
        loss = F.cross_entropy(model(x, n_loops=3).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    eval_rng = torch.Generator().manual_seed(123)
    x, y = _reverse_batch(eval_rng, 128)
    with torch.no_grad():
        logits = model(x, n_loops=3)
        eval_loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item()
        preds = logits.argmax(dim=-1)
        rev_acc = (
            (preds[:, PROMPT_LEN - 1 :] == y[:, PROMPT_LEN - 1 :]).float().mean().item()
        )

    uniform = math.log(VOCAB)
    assert eval_loss < uniform, f"loss {eval_loss} >= uniform {uniform}"
    # 5x random baseline (random = 1/VOCAB = ~3.1%)
    assert rev_acc > 5.0 / VOCAB, f"reverse acc {rev_acc} too low"


def test_kv_cache_matches_full_forward():
    """generate() with KV cache must produce identical tokens to greedy
    decoding done by repeated full-sequence forward passes."""
    torch.manual_seed(0)
    cfg = build_tiny_mla_config()
    model = OpenMythos(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 6))

    with torch.no_grad():
        cached = model.generate(prompt, max_new_tokens=4, n_loops=2, top_k=1)

    # Reference: greedy decode by repeated full forwards (no cache).
    tokens = prompt.clone()
    with torch.no_grad():
        for _ in range(4):
            logits = model(tokens, n_loops=2)
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)

    assert torch.equal(cached, tokens), (
        f"KV cache greedy decode disagrees with reference:\n"
        f"  cached={cached.tolist()}\n  ref   ={tokens.tolist()}"
    )
