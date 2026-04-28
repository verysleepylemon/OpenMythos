"""rotate_replicate.py - 6-seed audit of rotate_task_loops claims.

Original rotate_task_loops.py (3 seeds, dim=128 pl=8 k=4 STEPS=2000):
  - n=2 wins big (+6.36pp acc, 43x variance collapse)
  - n=4 collapses (-2.69pp)

After headline_replicate killed +13.17pp at 6 seeds, every 3-seed claim
on this branch needs to face the audit. This re-runs Rotate at the same
(dim, pl, k) with n in {1, 2, 4} at 6 seeds.

If n=2 survives: Rotate has its OWN sweet spot (n=2), distinct from
ReverseCopy's (n=4). The general recipe becomes
"loop count is task-specific; tune per task class."

If n=2 dies: rotate_task_loops was a 3-seed cluster artifact like
harder_n8 was. Then the SHIP_RECIPE green zone shrinks to ReverseCopy.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import OpenMythos, build_tiny_mla_config
from scripts._eval import evaluate, train_reverse_copy


@dataclass(frozen=True)
class RotateTask:
    vocab: int
    prompt_len: int
    k: int

    @property
    def seq_len(self) -> int:
        return 2 * self.prompt_len

    def make_batch(self, batch: int, rng: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt = torch.randint(0, self.vocab, (batch, self.prompt_len), generator=rng)
        rotated = torch.roll(prompt, shifts=-self.k, dims=1)
        seq = torch.cat([prompt, rotated], dim=1)
        return seq[:, :-1], seq[:, 1:]


LOOP_GRID = [1, 2, 4]
N_SEEDS = 6
STEPS = 2000
PROMPT_LEN = 8
ROT_K = 4
DIM = 128


def run(n_loops, seed):
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = RotateTask(vocab=8, prompt_len=PROMPT_LEN, k=ROT_K)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = n_loops
    cfg.prelude_layers = 1
    cfg.coda_layers = 2
    cfg.dim = DIM
    cfg.expert_dim = DIM // 2
    model = OpenMythos(cfg)
    train_reverse_copy(model, task, n_loops=n_loops, steps=STEPS, seed=seed)
    ce, acc = evaluate(model, task, n_loops=n_loops, seed=seed + 9999)
    return acc * 100, ce


def main():
    print(
        f"[rotate_replicate] dim={DIM} pl={PROMPT_LEN} k={ROT_K} "
        f"STEPS={STEPS} loops={LOOP_GRID} seeds={N_SEEDS}"
    )
    rows = {}
    for n in LOOP_GRID:
        ces, accs = [], []
        t0 = time.time()
        for seed in range(N_SEEDS):
            a, ce = run(n, seed)
            ces.append(ce)
            accs.append(a)
            print(
                f"[rotate_replicate] n={n} seed={seed} ce={ce:.4f} acc={a:.2f}%",
                flush=True,
            )
        rows[n] = {
            "ce_mean": statistics.fmean(ces),
            "ce_std": statistics.stdev(ces) if len(ces) > 1 else 0.0,
            "acc_mean": statistics.fmean(accs),
            "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "wall": time.time() - t0,
        }
        r = rows[n]
        print(
            f"[rotate_replicate] n={n} SUMMARY "
            f"ce={r['ce_mean']:.4f}+/-{r['ce_std']:.4f} "
            f"acc={r['acc_mean']:.2f}+/-{r['acc_std']:.2f} wall={r['wall']:.0f}s",
            flush=True,
        )

    base = rows[1]
    print()
    print(f"  delta vs n=1 ({N_SEEDS} seeds):")
    for n in LOOP_GRID[1:]:
        d_ce = rows[n]["ce_mean"] - base["ce_mean"]
        d_acc = rows[n]["acc_mean"] - base["acc_mean"]
        d_acc_std = rows[n]["acc_std"] - base["acc_std"]
        pooled = math.sqrt(rows[n]["ce_std"] ** 2 + base["ce_std"] ** 2)
        z = d_ce / pooled if pooled > 0 else 0.0
        verdict = "<- helps" if d_acc > 0 else "<- hurts"
        print(
            f"    n={n}: d_acc={d_acc:+.2f}pp d_std={d_acc_std:+.2f}pp "
            f"z_ce={z:+.2f} {verdict}"
        )

    chance_ce = math.log(8)
    assert base["ce_mean"] < chance_ce, (
        f"n=1 ce {base['ce_mean']} >= chance {chance_ce}"
    )

    print()
    print("[rotate_replicate] OK")


if __name__ == "__main__":
    main()
