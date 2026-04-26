"""Tiny scaling law: vary `dim` (and proportionally `expert_dim`) on the toy
task with multiple seeds, then plot params vs eval CE.

This is the personal branch's only "scaling law" — tiny in absolute terms but
honestly evaluated. We hold STEPS, BATCH, n_loops, n_layers fixed and vary
``cfg.dim`` over {32, 64, 128}, sweeping ``cfg.expert_dim = dim // 2``,
``cfg.n_heads`` and ``cfg.n_kv_heads`` to keep the model self-consistent.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import OpenMythos, MythosConfig

VOCAB = 32
PROMPT_LEN = 6
SEQ_LEN = 12
BATCH = 32
STEPS = 200
LR = 3e-3
N_LOOPS = 3
EVAL_BATCHES = 16
DIM_GRID = [32, 64, 128]
N_SEEDS = 3


def make_cfg(dim: int) -> MythosConfig:
    n_heads = max(2, dim // 16)
    head_dim = dim // n_heads
    return MythosConfig(
        vocab_size=VOCAB,
        dim=dim,
        n_heads=n_heads,
        max_seq_len=SEQ_LEN,
        max_loop_iters=N_LOOPS,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=max(16, dim // 2),
        lora_rank=4,
        attn_type="mla",
        n_kv_heads=n_heads,
        kv_lora_rank=max(16, dim // 4),
        q_lora_rank=max(32, dim // 2),
        qk_rope_head_dim=max(4, head_dim // 2),
        qk_nope_head_dim=max(4, head_dim // 2),
        v_head_dim=max(4, head_dim // 2),
    )


def make_batch(rng):
    prompt = torch.randint(0, VOCAB, (BATCH, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def train_one(dim: int, seed: int) -> tuple[int, float, float]:
    torch.manual_seed(seed)
    cfg = make_cfg(dim)
    model = OpenMythos(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        x, y = make_batch(rng)
        logits = model(x, n_loops=N_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    correct = total = 0
    losses = []
    eval_rng = torch.Generator().manual_seed(seed + 9999)
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
    return n_params, correct / max(1, total), sum(losses) / len(losses)


def main() -> None:
    print(f"[scale] sweeping dim={DIM_GRID} N_SEEDS={N_SEEDS}")
    rows = []
    for d in DIM_GRID:
        params_seen = []
        accs = []
        ces = []
        t0 = time.time()
        for s in range(N_SEEDS):
            p, a, ce = train_one(d, s)
            params_seen.append(p)
            accs.append(a * 100)
            ces.append(ce)
        wall = time.time() - t0
        assert len(set(params_seen)) == 1
        rows.append({
            "dim": d,
            "params": params_seen[0],
            "ce_mean": statistics.mean(ces),
            "ce_std": statistics.pstdev(ces),
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.pstdev(accs),
            "wall": wall,
        })
        print(f"[scale] dim={d:>3}  params={params_seen[0]:>7,}  "
              f"ce={rows[-1]['ce_mean']:.4f} +/- {rows[-1]['ce_std']:.4f}  "
              f"acc={rows[-1]['acc_mean']:.2f}%  wall={wall:.1f}s")

    print("\n  dim    params    eval_ce (mean +/- std)    rev_acc% (mean +/- std)")
    for r in rows:
        print(f"  {r['dim']:>3}  {r['params']:>8,}   "
              f"{r['ce_mean']:.4f} +/- {r['ce_std']:.4f}      "
              f"{r['acc_mean']:6.2f} +/- {r['acc_std']:5.2f}")

    if len(rows) >= 2:
        small = rows[0]
        big = rows[-1]
        ce_drop = small["ce_mean"] - big["ce_mean"]
        param_ratio = big["params"] / small["params"]
        pooled = math.sqrt(small["ce_std"] ** 2 + big["ce_std"] ** 2)
        z = ce_drop / pooled if pooled > 0 else float("inf")
        print(f"\n[scale] params x{param_ratio:.1f}  ce_drop={ce_drop:+.4f}  "
              f"pooled_std={pooled:.4f}  z={z:.2f}")

    uniform_ce = math.log(VOCAB)
    for r in rows:
        assert r["ce_mean"] < uniform_ce
    print("\n[scale] OK")


if __name__ == "__main__":
    main()
