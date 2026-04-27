"""Does the harder_n8 result generalize across routing tasks?

harder_n8 found n=8 gives +13.17pp at prompt_len=12 on ReverseCopy.
On Rotate at prompt_len=8, the optimum was n=2 and n=4 collapsed.
At prompt_len=12, does Rotate's optimum slide deeper too, or stay
shallow?

If task DIFFICULTY drives the optimum, Rotate at prompt_len=12
should also benefit from n=8. If task CLASS (Rotate's specific
structure) dominates, optimum stays near n=2.

dim=128 vocab=8 STEPS=2000 prompt_len=12 RotateBy(k=4) n in {1,2,4,8}
3 seeds.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import OpenMythos, build_tiny_mla_config
from scripts._eval import evaluate, train_reverse_copy

PROMPT_LEN = 12
VOCAB = 8
DIM = 128
STEPS = 2000
N_SEEDS = 3
LOOP_GRID = [1, 2, 4, 8]
ROTATE_K = 4


class RotateTask:
    """Predict input rotated left by k positions."""

    def __init__(self, vocab: int, prompt_len: int, k: int) -> None:
        self.vocab = vocab
        self.prompt_len = prompt_len
        self.k = k
        self.seq_len = 2 * prompt_len

    def make_batch(self, batch: int, rng):
        x = torch.randint(0, self.vocab, (batch, self.prompt_len), generator=rng)
        rotated = torch.roll(x, shifts=-self.k, dims=1)
        seq = torch.cat([x, rotated], dim=1)
        return seq[:, :-1], seq[:, 1:]


def run(n_loops: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = build_tiny_mla_config()
    task = RotateTask(vocab=VOCAB, prompt_len=PROMPT_LEN, k=ROTATE_K)
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
    print(f"[rotate_hard] dim={DIM} prompt_len={PROMPT_LEN} k={ROTATE_K} "
          f"STEPS={STEPS} LOOPS={LOOP_GRID} SEEDS={N_SEEDS}")
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
        print(f"[rotate_hard] n={n}  ce={r['ce_mean']:.4f}+/-{r['ce_std']:.4f}  "
              f"acc={r['acc_mean']:.2f}%+/-{r['acc_std']:.2f}  wall={r['wall']:.1f}s")

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
    for r in grid.values():
        assert r["ce_mean"] < chance_ce
    print("\n[rotate_hard] OK")


if __name__ == "__main__":
    main()
