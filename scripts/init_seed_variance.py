"""Init-seed variance probe.

Earlier `seed_robustness.py` showed that "single-seed wins are noise."
But what fraction of that noise is from model init vs from data shuffling?

This script fixes the data RNG (same training batches across all runs)
and varies ONLY the model-init seed. If init seed produces large variance
even with identical data, then init is a significant noise source on
its own, and we should report any future ablation alongside an init-only
noise floor.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import OpenMythos, build_tiny_mla_config
from scripts._eval import ReverseCopyTask, evaluate, train_reverse_copy

N_SEEDS = 5
STEPS = 200
N_LOOPS = 2
DATA_SEED = 42  # FIXED across all runs


def run(init_seed: int) -> tuple[float, float]:
    torch.manual_seed(init_seed)  # only this varies
    cfg = build_tiny_mla_config()
    task = ReverseCopyTask(vocab=8, prompt_len=4)
    cfg.vocab_size = task.vocab
    cfg.max_seq_len = task.seq_len
    cfg.max_loop_iters = N_LOOPS
    model = OpenMythos(cfg)
    train_reverse_copy(model, task, n_loops=N_LOOPS, steps=STEPS, seed=DATA_SEED)
    ce, acc = evaluate(model, task, n_loops=N_LOOPS, seed=DATA_SEED + 9999)
    return acc * 100, ce


def main() -> None:
    print(f"[init_seed] N_SEEDS={N_SEEDS} STEPS={STEPS} DATA_SEED={DATA_SEED} (fixed)")
    accs, ces = [], []
    t0 = time.time()
    for s in range(N_SEEDS):
        a, c = run(s)
        accs.append(a)
        ces.append(c)
        print(f"  init_seed={s}  ce={c:.4f}  acc={a:.2f}%")
    wall = time.time() - t0

    ce_mean = statistics.mean(ces)
    ce_std = statistics.pstdev(ces)
    acc_mean = statistics.mean(accs)
    acc_std = statistics.pstdev(accs)
    print(f"\n[init_seed] ce={ce_mean:.4f} +/- {ce_std:.4f}  acc={acc_mean:.2f} +/- {acc_std:.2f}%  wall={wall:.1f}s")

    # Compare to the seed_robustness "everything varied" baseline (both seeds varied).
    # Earlier seed_robustness on the same task gave roughly std(ce) ~ 0.018.
    # If init_only std is ~the same, init dominates the noise budget;
    # if much smaller, data shuffling dominates.
    print(f"\n[init_seed] Reference: 'all-varied' run earlier had std(ce) ~ 0.018.")
    if ce_std > 0.012:
        print(f"[init_seed] init alone explains {ce_std/0.018*100:.0f}% of variance budget — init dominates.")
    else:
        print(f"[init_seed] init alone is {ce_std/0.018*100:.0f}% of variance — data shuffling dominates.")

    assert ce_std < 0.1, f"init-only ce std {ce_std} suspiciously high"
    print("\n[init_seed] OK")


if __name__ == "__main__":
    main()
