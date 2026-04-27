"""Was the Sort null result a headroom artifact?

sort_task_loops at vocab=8 prompt_len=8 hit 98.68% with n=1 - no
room left to show a loop benefit. Re-run with vocab=32 to drop
the floor and open headroom.

dim=128 prompt_len=8 vocab=32 n_loops in {1, 2, 4} 3 seeds.
Same recipe as before, only vocab changes.
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
class SortTask:
    vocab: int
    prompt_len: int

    @property
    def seq_len(self) -> int:
        return 2 * self.prompt_len

    def make_batch(self, batch: int, rng: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt = torch.randint(0, self.vocab, (batch, self.prompt_len), generator=rng)
        sorted_prompt, _ = torch.sort(prompt, dim=1)
        seq = torch.cat([prompt, sorted_prompt], dim=1)
        return seq[:, :-1], seq[:, 1:]


VOCAB = 32
LOOP_GRID = [1, 2, 4]
N_SEEDS = 3
STEPS = 2000
PROMPT_LEN = 8
DIM = 128


def run(n_loops: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = SortTask(vocab=VOCAB, prompt_len=PROMPT_LEN)
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


def main() -> None:
    print(f"[sort_hard] dim={DIM} prompt_len={PROMPT_LEN} vocab={VOCAB} STEPS={STEPS}")
    grid = {}
    for n in LOOP_GRID:
        ces, accs = [], []
        t0 = time.time()
        for seed in range(N_SEEDS):
            a, ce = run(n, seed)
            ces.append(ce); accs.append(a)
        grid[n] = {
            "ce_mean": statistics.mean(ces),
            "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs),
            "wall": time.time() - t0,
        }
        r = grid[n]
        print(f"[sort_hard] n={n}  ce={r['ce_mean']:.4f}+/-{r['ce_std']:.4f}  "
              f"acc={r['acc_mean']:.2f}%+/-{r['acc_std']:.2f}  wall={r['wall']:.1f}s")

    print("\n  n_loops  eval_ce            acc%")
    for n in LOOP_GRID:
        r = grid[n]
        print(f"     {n}    {r['ce_mean']:.4f}+/-{r['ce_std']:.4f}  "
              f"{r['acc_mean']:5.2f}+/-{r['acc_std']:.2f}")

    print("\n  n>1 advantage vs n=1:")
    base = grid[1]
    for n in LOOP_GRID:
        if n == 1:
            continue
        r = grid[n]
        delta = r["ce_mean"] - base["ce_mean"]
        pooled = math.sqrt(r["ce_std"] ** 2 + base["ce_std"] ** 2)
        z = delta / pooled if pooled > 0 else float("inf")
        d_acc = r["acc_mean"] - base["acc_mean"]
        print(f"    n={n}: delta_ce={delta:+.4f}  z={z:+.2f}  delta_acc={d_acc:+.2f}pp")

    chance_ce = math.log(VOCAB)
    print(f"\n  chance_ce = ln({VOCAB}) = {chance_ce:.4f}")
    for r in grid.values():
        assert r["ce_mean"] < chance_ce
    print("\n[sort_hard] OK")


if __name__ == "__main__":
    main()
