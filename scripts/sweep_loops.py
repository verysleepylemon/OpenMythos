"""Sweep n_loops to observe how recurrent depth affects logit distribution.

The Recurrent-Depth Transformer applies the same Recurrent Block up to
`max_loop_iters` times. With a contractive injection (rho(A) < 1) the hidden
state should converge as n_loops grows, so the logit distribution should
stabilize. This script measures that convergence on CPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.quick_demo import build_tiny_mla_config

from open_mythos.main import OpenMythos


def main() -> None:
    torch.manual_seed(0)
    cfg = build_tiny_mla_config()
    cfg.max_loop_iters = 8
    model = OpenMythos(cfg)
    model.eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    prev: torch.Tensor | None = None

    print(f"{'n_loops':>8}  {'mean_logit':>12}  {'std_logit':>10}  {'kl_vs_prev':>12}")
    for n in (1, 2, 3, 4, 6, 8):
        with torch.no_grad():
            logits = model(ids, n_loops=n)
        probs = F.softmax(logits, dim=-1)
        mean = logits.mean().item()
        std = logits.std().item()
        kl_str = "—"
        if prev is not None:
            kl = F.kl_div(probs.log(), prev, reduction="batchmean").item()
            kl_str = f"{kl:.6f}"
        print(f"{n:>8}  {mean:>12.4f}  {std:>10.4f}  {kl_str:>12}")
        prev = probs


if __name__ == "__main__":
    main()
