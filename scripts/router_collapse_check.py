"""Quantify router health for the MoE FFN with formal metrics.

Builds the same tiny MoE model as ``expert_usage.py``, but instead of just
printing per-expert percentages, computes hard scalar metrics over the routing
distribution at three checkpoints:

  - INIT          : fresh model, no training
  - MID  (50 st.) : early in training
  - FINAL (200 st): after the toy task is partially learned

Metrics:
  - entropy H(p)         : in nats. Larger = more uniform routing.
  - normalized H(p)      : H(p) / log(N_EXPERTS), in [0,1]. 1 = perfect uniform.
  - gini coefficient     : in [0,1). 0 = uniform, ~1 = single-expert collapse.
  - dead expert ratio    : fraction of experts receiving < 1% of selections.
  - max_dev_pp           : maximum deviation from ideal share, in percentage pts.

This is a "no-bias-update" probe: load-balancing loss / bias updates are NOT
applied, so we expect the router to drift toward imbalance — the question is
*how badly*.
"""

from __future__ import annotations

import math
import sys
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
LR = 3e-3
N_LOOPS = 3
N_EXPERTS = 4
TOPK = 2
SEED = 0
CHECKPOINTS = [(0, "INIT"), (50, "MID"), (200, "FINAL")]


def make_batch(rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = torch.randint(0, VOCAB, (BATCH, PROMPT_LEN), generator=rng)
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
    counters: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if type(module).__name__ != "MoEFFN":
            continue
        c = torch.zeros(N_EXPERTS, dtype=torch.long)
        counters[name] = c

        def make_hook(counter: torch.Tensor, m: torch.nn.Module):
            def hook(_module, _inp, out):
                _, topk_idx = (out + m.router_bias).topk(TOPK, dim=-1)
                bins = torch.bincount(topk_idx.reshape(-1), minlength=N_EXPERTS).to(counter.dtype)
                counter.add_(bins)
            return hook

        module.router.register_forward_hook(make_hook(c, module))
    return counters


def reset_counters(counters: dict[str, torch.Tensor]) -> None:
    for c in counters.values():
        c.zero_()


def gini(p: torch.Tensor) -> float:
    """Gini coefficient of a probability distribution. p must sum to 1.

    g = 1 - sum_i p_i^2  is the Gini-Simpson index; the population-Gini for
    a discrete distribution with N atoms uses the formal sorted-cumulative
    formula below. Returns 0 for uniform, approaches 1 - 1/N for one-hot.
    """
    n = p.numel()
    sorted_p, _ = p.sort()
    cum = torch.cumsum(sorted_p, dim=0)
    # standard Gini for grouped data
    # G = (n + 1 - 2 * sum_i (n - i + 1) * p_i / sum p) / n   when p sorted ascending
    idx = torch.arange(1, n + 1, dtype=p.dtype)
    g = (n + 1 - 2 * ((n - idx + 1) * sorted_p).sum() / sorted_p.sum()) / n
    return float(g.item())


def metrics_from_counters(counters: dict[str, torch.Tensor]) -> dict[str, float]:
    agg = torch.zeros(N_EXPERTS, dtype=torch.long)
    for c in counters.values():
        agg += c
    total = agg.sum().item()
    if total == 0:
        return {"entropy": 0.0, "norm_entropy": 0.0, "gini": 0.0,
                "dead_frac": 0.0, "max_dev_pp": 0.0, "selections": 0}
    p = agg.float() / total
    eps = 1e-12
    H = float(-(p * (p + eps).log()).sum().item())
    H_norm = H / math.log(N_EXPERTS)
    g = gini(p)
    dead = float((p < 0.01).float().mean().item())
    ideal = 1.0 / N_EXPERTS
    max_dev = float(((p - ideal).abs() * 100).max().item())
    return {
        "entropy": H,
        "norm_entropy": H_norm,
        "gini": g,
        "dead_frac": dead,
        "max_dev_pp": max_dev,
        "selections": total,
        "shares_pct": [float(x) * 100 for x in p.tolist()],
    }


def fmt_metrics(label: str, m: dict) -> str:
    shares = " ".join(f"E{i}={s:4.1f}%" for i, s in enumerate(m["shares_pct"]))
    return (
        f"[router] {label:<6}  H={m['entropy']:.4f} (norm={m['norm_entropy']:.3f})  "
        f"gini={m['gini']:.3f}  max_dev={m['max_dev_pp']:.1f}pp  "
        f"dead<1%={m['dead_frac']*100:.0f}%  selections={m['selections']}\n"
        f"[router]         shares: {shares}"
    )


def main() -> None:
    torch.manual_seed(SEED)
    model = build_model()
    rng = torch.Generator().manual_seed(SEED)
    counters = install_router_hooks(model)
    n_moe = sum(1 for m in model.modules() if type(m).__name__ == "MoEFFN")
    print(f"[router] MoE layers={n_moe}  experts={N_EXPERTS}  topk={TOPK}  "
          f"ideal_share={100/N_EXPERTS:.1f}%/expert")
    print(f"[router] checkpoints={CHECKPOINTS}")
    print()

    snapshots: list[tuple[str, dict]] = []

    # INIT snapshot: 16 forward passes with no training.
    model.eval()
    reset_counters(counters)
    with torch.no_grad():
        for _ in range(16):
            x, _ = make_batch(rng)
            model(x, n_loops=N_LOOPS)
    snapshots.append(("INIT", metrics_from_counters(counters)))
    print(fmt_metrics("INIT", snapshots[-1][1]))

    # Train and snapshot at MID and FINAL.
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    next_check = {step: label for step, label in CHECKPOINTS if step > 0}
    final_step = max(next_check)
    for step in range(1, final_step + 1):
        x, y = make_batch(rng)
        logits = model(x, n_loops=N_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in next_check:
            label = next_check[step]
            # Snapshot eval-mode routing on a fresh batch (don't double-count training counters).
            model.eval()
            reset_counters(counters)
            with torch.no_grad():
                for _ in range(16):
                    xs, _ = make_batch(rng)
                    model(xs, n_loops=N_LOOPS)
            snapshots.append((label, metrics_from_counters(counters)))
            print(fmt_metrics(label, snapshots[-1][1]))
            model.train()

    print()
    init_g = snapshots[0][1]["gini"]
    final_g = snapshots[-1][1]["gini"]
    drift = final_g - init_g
    print(f"[router] gini drift INIT->FINAL: {init_g:.3f} -> {final_g:.3f}  "
          f"(delta={drift:+.3f})")
    if drift > 0.10:
        print(f"[router] WARNING: routing drifted significantly toward collapse.")
    elif drift < -0.05:
        print(f"[router] routing actually got MORE uniform during training.")
    else:
        print(f"[router] routing stayed roughly stable (drift in noise band).")

    # Hard sanity: the final entropy should not be ~0 (one-hot collapse).
    assert snapshots[-1][1]["entropy"] > 0.1, "router collapsed to single expert"
    print("\n[router] OK")


if __name__ == "__main__":
    main()
