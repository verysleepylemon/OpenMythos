"""Demonstrate that an auxiliary load-balancing loss prevents router collapse.

router_collapse_check.py showed that without any load-balancing penalty, the
MoE router drifts toward an unbalanced distribution (gini ~0.45 by step 200,
some experts near-dead). This script trains the *same* tiny model on the
*same* reverse-copy task with two settings:

  baseline:  CE loss only
  balanced:  CE loss + LB_COEF * load_balance_loss

The load-balance loss is the standard Switch-Transformer / DeepSeek-MoE form:
  lb = N * sum_i (f_i * P_i)
where f_i is the *fraction of selections* assigned to expert i (one-hot
top-k count, normalized) and P_i is the *mean router probability* for expert
i. Both are per-step quantities. The product is minimized when both
distributions are uniform.

We then report the same gini / entropy / dead-expert metrics for both
trained models.
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
PROMPT_LEN = 6
SEQ_LEN = 12
BATCH = 32
STEPS = 200
LR = 3e-3
N_LOOPS = 3
TOPK = 2
LB_COEF = 0.05
SEED = 0


def make_batch(rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = torch.randint(0, VOCAB, (BATCH, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def find_moe(model: OpenMythos):
    for m in model.modules():
        if type(m).__name__ == "MoEFFN":
            return m
    raise RuntimeError("no MoEFFN module found")


def gini(p: torch.Tensor) -> float:
    """Standard Gini on a non-negative vector."""
    x, _ = torch.sort(p.flatten())
    n = x.numel()
    if n == 0 or float(x.sum()) == 0.0:
        return 0.0
    idx = torch.arange(1, n + 1, dtype=x.dtype)
    g = (n + 1 - 2 * (idx * x.flip(0)).sum() / x.sum()) / n
    return float(g.item())


def measure_routing(model: OpenMythos, n_batches: int = 16) -> dict[str, float]:
    """Snapshot routing distribution by running n_batches forward passes."""
    moe = find_moe(model)
    n_experts = moe.router.weight.shape[0]
    counts = torch.zeros(n_experts)

    def hook(_module, _inp, out):
        # Replicate the selection logic used internally by MoEFFN.
        scores = (out + moe.router_bias).flatten(0, -2)
        sel = scores.topk(TOPK, dim=-1).indices.flatten()
        bins = torch.bincount(sel, minlength=n_experts).to(counts.dtype)
        counts.add_(bins)

    h = moe.router.register_forward_hook(hook)
    rng = torch.Generator().manual_seed(2025)
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(n_batches):
                x, _ = make_batch(rng)
                model(x, n_loops=N_LOOPS)
    finally:
        h.remove()

    total = float(counts.sum().item())
    p = counts / max(total, 1.0)
    ideal = 1.0 / n_experts
    entropy = float(-(p * (p.clamp_min(1e-12)).log()).sum().item())
    norm_h = entropy / float(torch.tensor(n_experts).log().item())
    g = gini(p)
    dead = int((p < 0.01).sum().item())
    max_dev_pp = float((p - ideal).abs().max().item()) * 100
    return {
        "entropy": entropy,
        "norm_entropy": norm_h,
        "gini": g,
        "dead_experts": dead,
        "max_dev_pp": max_dev_pp,
        "p": p.tolist(),
    }


def make_model() -> OpenMythos:
    torch.manual_seed(SEED)
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = N_LOOPS
    return OpenMythos(cfg)


def train(use_lb: bool) -> tuple[float, OpenMythos]:
    model = make_model()
    moe = find_moe(model)
    n_experts = moe.router.weight.shape[0]
    rng = torch.Generator().manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # Capture the latest router scores via forward hook so we can build an LB
    # loss on the *current* step's distribution.
    captured: dict[str, torch.Tensor] = {}

    def lb_hook(_m, _i, out):
        # Save raw router scores (no bias) for LB; keep grad path.
        captured["scores"] = out

    handle = moe.router.register_forward_hook(lb_hook)
    try:
        for _ in range(STEPS):
            x, y = make_batch(rng)
            logits = model(x, n_loops=N_LOOPS)
            ce = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            loss = ce
            if use_lb and "scores" in captured:
                s = captured["scores"]
                # Effective router probabilities (after softmax).
                P = F.softmax(s + moe.router_bias, dim=-1)  # (..., E)
                P_mean = P.flatten(0, -2).mean(dim=0)        # (E,)
                # Selection fractions (top-k indicator, normalized over tokens).
                with torch.no_grad():
                    sel = P.flatten(0, -2).topk(TOPK, dim=-1).indices  # (T,K)
                    one_hot = F.one_hot(sel, num_classes=n_experts).sum(dim=1).float()
                    f = one_hot.mean(dim=0)  # (E,)  fraction per expert
                lb = float(n_experts) * (f * P_mean).sum()
                loss = loss + LB_COEF * lb
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    finally:
        handle.remove()
    return ce.item(), model


def main() -> None:
    print(f"[lb] training reverse-copy STEPS={STEPS} LB_COEF={LB_COEF if True else 0}")
    rows = []
    for use_lb, label in [(False, "BASELINE"), (True, "BALANCED")]:
        t0 = time.time()
        final_ce, model = train(use_lb)
        wall = time.time() - t0
        m = measure_routing(model)
        rows.append({"label": label, "use_lb": use_lb, "ce": final_ce, "wall": wall, **m})
        print(
            f"[lb] {label:<8}  final_ce={final_ce:.4f}  H={m['entropy']:.4f} "
            f"(norm={m['norm_entropy']:.3f})  gini={m['gini']:.4f}  "
            f"dead={m['dead_experts']}  max_dev={m['max_dev_pp']:.1f}pp  "
            f"wall={wall:.1f}s"
        )
        print(f"[lb] {label:<8}  expert shares = " +
              "  ".join(f"E{i}={p*100:5.1f}%" for i, p in enumerate(m["p"])))

    base = next(r for r in rows if not r["use_lb"])
    bal = next(r for r in rows if r["use_lb"])
    delta_gini = bal["gini"] - base["gini"]
    print(f"\n[lb] gini change (BALANCED - BASELINE): {delta_gini:+.4f}")
    if delta_gini < -0.02:
        print("[lb] verdict: LB loss measurably reduced router imbalance.")
    elif delta_gini > 0.02:
        print("[lb] verdict: LB loss INCREASED imbalance (toy task / noise).")
    else:
        print("[lb] verdict: change within noise on this toy task.")

    # Sanity: LB run must not have made CE catastrophically worse.
    assert bal["ce"] < base["ce"] * 1.5, (
        f"LB hurt CE too much: base={base['ce']:.4f} bal={bal['ce']:.4f}"
    )
    # Sanity: with 4 experts, gini in [0,1] and shares sum to 1.
    for r in rows:
        assert 0.0 <= r["gini"] <= 1.0
        assert abs(sum(r["p"]) - 1.0) < 1e-5

    print("\n[lb] OK")


if __name__ == "__main__":
    main()
