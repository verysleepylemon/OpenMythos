"""Compare flat LR vs cosine LR (with warmup) on the toy task.

Train two identical tiny models with the same seed, same total step count,
same peak LR. The only difference is the schedule:
  - flat:    LR is held at PEAK_LR the entire run.
  - cosine:  LR linearly warms up from 0 -> PEAK_LR over WARMUP steps, then
             decays cosine-shaped from PEAK_LR -> 0 over the remaining steps.

Reports per-schedule loss at milestones, final eval CE, and reverse-half
greedy accuracy.

This is one of those experiments where the toy task is *just* big enough to
see a schedule effect: cosine + warmup typically gives a marginally better
final loss, even though both eventually converge.
"""

from __future__ import annotations

import math
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
STEPS = 300
PEAK_LR = 3e-3
WARMUP = 30
N_LOOPS = 3
EVAL_BATCHES = 16
SEED = 0
MILESTONES = [50, 100, 200, 300]


def make_batch(rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = torch.randint(0, VOCAB, (BATCH, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def cosine_lr(step: int) -> float:
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, STEPS - WARMUP)
    return PEAK_LR * 0.5 * (1.0 + math.cos(math.pi * progress))


def flat_lr(_step: int) -> float:
    return PEAK_LR


def make_model() -> OpenMythos:
    torch.manual_seed(SEED)
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = N_LOOPS
    return OpenMythos(cfg)


def train(schedule, label: str) -> dict:
    model = make_model()
    rng = torch.Generator().manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
    model.train()
    milestones: dict[int, float] = {}
    final_loss = float("nan")
    t0 = time.time()
    for step in range(1, STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = schedule(step - 1)
        x, y = make_batch(rng)
        logits = model(x, n_loops=N_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in MILESTONES:
            milestones[step] = loss.item()
        final_loss = loss.item()
    wall = time.time() - t0

    # Eval
    model.eval()
    correct = total = 0
    losses = []
    eval_rng = torch.Generator().manual_seed(SEED + 9999)
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = make_batch(eval_rng)
            logits = model(x, n_loops=N_LOOPS)
            losses.append(F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item())
            preds = logits.argmax(dim=-1)
            t = y[:, PROMPT_LEN - 1 : 2 * PROMPT_LEN - 1]
            p = preds[:, PROMPT_LEN - 1 : 2 * PROMPT_LEN - 1]
            correct += (p == t).sum().item()
            total += t.numel()
    return {
        "label": label,
        "milestones": milestones,
        "final_train_loss": final_loss,
        "eval_ce": sum(losses) / len(losses),
        "rev_acc": correct / max(1, total) * 100,
        "wall": wall,
    }


def main() -> None:
    print(f"[lr] flat vs cosine+warmup  STEPS={STEPS}  WARMUP={WARMUP}  "
          f"PEAK_LR={PEAK_LR}  N_LOOPS={N_LOOPS}")
    rows = [train(flat_lr, "FLAT  "), train(cosine_lr, "COSINE")]
    for r in rows:
        ms = "  ".join(f"step{k}={v:.4f}" for k, v in r["milestones"].items())
        print(f"[lr] {r['label']}  {ms}  eval_ce={r['eval_ce']:.4f}  "
              f"rev_acc={r['rev_acc']:.2f}%  wall={r['wall']:.1f}s")

    print("\n  schedule    s50      s100     s200     s300     eval_ce   rev_acc%  wall_s")
    for r in rows:
        m = r["milestones"]
        print(f"  {r['label']}     {m[50]:.4f}  {m[100]:.4f}  {m[200]:.4f}  {m[300]:.4f}  "
              f"{r['eval_ce']:.4f}    {r['rev_acc']:6.2f}    {r['wall']:5.1f}")

    flat = next(r for r in rows if r["label"].strip() == "FLAT")
    cos = next(r for r in rows if r["label"].strip() == "COSINE")
    delta_ce = cos["eval_ce"] - flat["eval_ce"]
    print(f"\n[lr] eval_ce delta (cosine - flat): {delta_ce:+.4f}")
    if delta_ce < -0.01:
        print("[lr] verdict: cosine+warmup wins by a measurable margin.")
    elif delta_ce > 0.01:
        print("[lr] verdict: flat wins by a measurable margin (toy / noise).")
    else:
        print("[lr] verdict: schedules are within noise on this toy task.")

    # Both schedules must learn.
    uniform_ce = math.log(VOCAB)
    for r in rows:
        assert r["eval_ce"] < uniform_ce, (
            f"{r['label']} failed: ce={r['eval_ce']:.4f} >= uniform={uniform_ce:.4f}"
        )

    print("\n[lr] OK")


if __name__ == "__main__":
    main()
