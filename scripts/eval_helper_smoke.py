"""End-to-end smoke test that scripts/_eval.py works.

We re-implement the canonical reverse-copy training run, this time using
ReverseCopyTask + train_reverse_copy + evaluate from scripts/_eval.py.
The point is to exercise the shared helper before any other script depends
on it (so a regression here is caught in CI before propagating).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import OpenMythos, build_tiny_mla_config
from scripts._eval import ReverseCopyTask, evaluate, train_reverse_copy

VOCAB = 32
PROMPT_LEN = 6
N_LOOPS = 3
STEPS = 200
SEEDS = [0, 1, 2]


def main() -> None:
    task = ReverseCopyTask(vocab=VOCAB, prompt_len=PROMPT_LEN)
    print(f"[eval-helper] task seq_len={task.seq_len} vocab={VOCAB} steps={STEPS} seeds={SEEDS}")

    eval_ces = []
    accs = []
    t0 = time.time()
    for seed in SEEDS:
        torch.manual_seed(seed)
        cfg = build_tiny_mla_config()
        cfg.vocab_size = VOCAB
        cfg.max_seq_len = task.seq_len
        cfg.max_loop_iters = N_LOOPS
        model = OpenMythos(cfg)
        train_reverse_copy(
            model, task,
            n_loops=N_LOOPS, steps=STEPS, batch=32, lr=3e-3, clip=1.0, seed=seed,
        )
        ce, acc = evaluate(model, task, n_loops=N_LOOPS)
        eval_ces.append(ce)
        accs.append(acc * 100)
        print(f"[eval-helper] seed={seed}  ce={ce:.4f}  acc={acc*100:.2f}%")

    wall = time.time() - t0
    mean_ce = sum(eval_ces) / len(eval_ces)
    mean_acc = sum(accs) / len(accs)
    print(f"\n[eval-helper] mean_ce={mean_ce:.4f}  mean_acc={mean_acc:.2f}%  wall={wall:.1f}s")

    uniform_ce = math.log(VOCAB)
    assert mean_ce < uniform_ce, f"helper-trained model failed to learn (ce={mean_ce:.4f} >= {uniform_ce:.4f})"
    print("\n[eval-helper] OK")


if __name__ == "__main__":
    main()
