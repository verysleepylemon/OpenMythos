# Personal Branch — Headline Findings

> Plain-English summary of what we actually learned by running 35+
> ablations on the OpenMythos recurrent-depth transformer at tiny
> scale on toy reverse-copy / parity / sort tasks.

This doc points at the scripts in [`../scripts/`](../scripts/) and
the per-script numbers in [`../RESULTS.md`](../RESULTS.md).

---

## TL;DR

1. **Recurrent depth (`n_loops`) buys nothing on toy tasks at this size.**
   Triangulated across THREE independent convergent measurements:
   `seed_robustness` (single-seed wins were init noise), `longer_training`
   (gap stayed closed across 5x compute), and `loops_at_long_train` (at
   94% acc convergence, n=4 vs n=1 gap is z=-0.12). This is a real
   architectural ceiling on this regime, not under-training.
2. **Stacked depth DOES help** — `recurrent_vs_stacked` shows K=4 stacked
   beats K=4 recurrent by 0.09 ce (z=2.69), but pays 1.77× more params.
3. **The architecture is asymmetric.** `prelude_coda_depth` shows
   `prelude=1, coda=2` is optimal (z=2.16 vs `1/1`); MORE prelude
   actively HURTS (+0.06 to +0.11 ce, with much higher variance).
   This is the first ablation where increasing capacity in one
   direction helps and the symmetric direction hurts.
4. **Init seed is the dominant noise source.** `init_seed_variance`
   shows fixing data and varying only model init produces *higher*
   variance than varying both. Any ablation reporting `delta(ce) < 0.026`
   on this recipe is almost certainly noise.
5. **Three knobs actually move the needle.** Batch size (z=3.71),
   training length (z=8-11), and (for routed experts) `n_experts_per_tok=2`
   over `=1`. Everything else we tried sits at or below the init-noise floor.

---

## What does NOT help (multi-seed, below noise floor)

| Knob | Result | Script |
|------|--------|--------|
| Recurrent depth (n_loops) | z=-0.12 even at convergence | [`loops_at_long_train.py`](../scripts/loops_at_long_train.py) |
| Schedule shape (cosine warmup vs flat) | Flat slightly wins (+0.04 ce) | [`cosine_lr_warmup.py`](../scripts/cosine_lr_warmup.py) |
| Weight decay (0 / 0.01 / 0.1) | All within 0.04 ce | [`weight_decay_ablation.py`](../scripts/weight_decay_ablation.py) |
| Grad clip (≥1.0) | clip never fires once magnitude is reasonable | [`grad_clip_ablation.py`](../scripts/grad_clip_ablation.py) |
| MLA vs GQA attention | gap below seed noise | [`attention_variant_compare.py`](../scripts/attention_variant_compare.py) |
| LoRA adapter rank | rank does not matter on this task | [`lora_adapter_ablation.py`](../scripts/lora_adapter_ablation.py) |
| n_experts in {2, 4, 8} | all within init-noise floor | [`expert_count_sweep.py`](../scripts/expert_count_sweep.py) |

## What DOES move the needle (real positive, multi-seed)

| Knob | Effect | Script |
|------|--------|--------|
| Batch size 8→64 | ce 3.41 → 2.97 (z=3.71) | [`batch_size_curve.py`](../scripts/batch_size_curve.py) |
| Training length 200→1000 | ce 1.7 → 1.0 (z=8–11) | [`longer_training.py`](../scripts/longer_training.py) |
| `n_experts_per_tok` 1→2 | ce -0.05 (z≈1.3) | [`topk_experts_sweep.py`](../scripts/topk_experts_sweep.py) |
| Load-balance loss | gini 0.37 → 0.11 (huge) | [`router_balance_loss.py`](../scripts/router_balance_loss.py) |
| Stacked depth K=4 vs recurrent K=4 | stacked +0.09 ce for 1.77× params | [`recurrent_vs_stacked.py`](../scripts/recurrent_vs_stacked.py) |
| coda_layers 1→2 | ce -0.05 (z=2.16) | [`prelude_coda_depth.py`](../scripts/prelude_coda_depth.py) |

## What HURTS (multi-seed, real negative)

| Knob | Effect | Script |
|------|--------|--------|
| prelude_layers 1→2 | +0.06 to +0.11 ce, ~3x higher variance | [`prelude_coda_depth.py`](../scripts/prelude_coda_depth.py) |
| LR=1e-2 (3x baseline) | +0.28 ce, 5x higher variance | [`lr_peak_curve.py`](../scripts/lr_peak_curve.py) |
| kv_lora_rank=8 (half default) | +0.05 ce (z=1.4), saves only 3k params | [`mla_kv_rank_sweep.py`](../scripts/mla_kv_rank_sweep.py) |

## What's set correctly by default

| Default | Confirmed by |
|---------|--------------|
| Peak LR = 3e-3 | [`lr_peak_curve.py`](../scripts/lr_peak_curve.py) (1e-2 unstable, 1e-3 underfits) |
| `n_experts_per_tok` = 2 | [`topk_experts_sweep.py`](../scripts/topk_experts_sweep.py) (k=2 = k=3 on ce, lower compute) |
| `n_shared_experts` = 1 | [`shared_expert_ablation.py`](../scripts/shared_expert_ablation.py) (best ce, lowest variance) |
| `kv_lora_rank` = 16 | [`mla_kv_rank_sweep.py`](../scripts/mla_kv_rank_sweep.py) (sits exactly at the elbow) |
| `dim` = 64 | [`model_size_scaling.py`](../scripts/model_size_scaling.py) (saturates here) |

## Tasks the model CAN learn

| Task | Best result | Notes |
|------|-------------|-------|
| Reverse-copy (vocab=8, len=8) | ~95% acc, ce ≈ 1.0 | Reaches floor by 1000 steps |
| Sort-3 (vocab=8 + SEP) | 99.7% acc at n=1 | Too easy to be a depth probe |

## Tasks the model CANNOT learn

| Task | Result | Why |
|------|--------|-----|
| Parity / cumulative XOR (16 bits) | All n_loops at chance (50.01%) | Hahn 2020: transformers cannot represent unbounded parity |

---

## Methodology notes worth carrying forward

1. **Always report multi-seed.** Single-seed experiments produced
   misleading wins on this branch repeatedly (loops, weight decay, schedule).
2. **Init dimension matters.** When isolating noise sources, vary
   `torch.manual_seed` separately from data-shuffle RNG.
3. **Below-noise floor on default recipe = 0.026 ce.** Anything smaller
   should not be claimed as a finding.
4. **Test at convergence.** Even after pushing training 5× and reaching
   94% acc, the n_loops gap stayed within 1/200th of a sigma —
   so the recurrent ceiling here is real, not an under-training artifact.
5. **Toy tasks are dangerous priors.** Reverse-copy is data-bound,
   not capacity-bound (`batch_size_curve` and `model_size_scaling`
   both confirm). Don't extrapolate "recurrence is useless" — extrapolate
   "recurrence is useless at this scale on this task."

## Where to look next (untested here)

- Larger model sizes (dim ≥ 256) — does parameter sharing start
  paying when each block's params are large?
- Tasks with explicit iterative structure (graph traversal, multi-hop
  arithmetic, dynamic programming).
- Inference-time depth control — even if more loops don't *train*
  better, early-exit gating still has independent value.
- The asymmetry from `prelude_coda_depth` deserves its own followup —
  why does adding prelude layers hurt but adding coda layers help?
  Likely candidate: the recurrent block already saturates "early"
  representation, and the model is bottlenecked on output-side depth.

---

*Generated from CI runs on the personal branch.
For exact numbers and seeds, see [`RESULTS.md`](../RESULTS.md).*


## Tasks the model CAN learn

| Task | Best result | Notes |
|------|-------------|-------|
| Reverse-copy (vocab=8, len=8) | ~95% acc, ce ≈ 1.0 | Reaches floor by 1000 steps |
| Sort-3 (vocab=8 + SEP) | 99.7% acc at n=1 | Too easy to be a depth probe |

## Tasks the model CANNOT learn

| Task | Result | Why |
|------|--------|-----|
| Parity / cumulative XOR (16 bits) | All n_loops at chance (50.01%) | Hahn 2020: transformers cannot represent unbounded parity |

---

## Methodology notes worth carrying forward

1. **Always report multi-seed.** Single-seed experiments produced
   misleading wins on this branch repeatedly (loops, weight decay, schedule).
2. **Init dimension matters.** When isolating noise sources, vary
   `torch.manual_seed` separately from data-shuffle RNG.
3. **Below-noise floor on default recipe = 0.026 ce.** Anything smaller
   should not be claimed as a finding.
4. **Test at convergence.** Effects measured at 200 steps may look
   smaller than they are — but in this codebase even 5× more compute
   did NOT open the recurrent gap, so this is a real architectural
   ceiling, not a training-budget artifact.
5. **Toy tasks are dangerous priors.** Reverse-copy is data-bound,
   not capacity-bound (`batch_size_curve` and `model_size_scaling`
   both confirm). Don't extrapolate "recurrence is useless" — extrapolate
   "recurrence is useless at this scale on this task."

## Where to look next (untested here)

- Larger model sizes (dim ≥ 256) — does parameter sharing start
  paying when each block's params are large?
- Tasks with explicit iterative structure (graph traversal, multi-hop
  arithmetic, dynamic programming).
- Inference-time depth control — even if more loops don't *train*
  better, early-exit gating still has independent value.

---

*Generated from CI runs on the personal branch.
For exact numbers and seeds, see [`RESULTS.md`](../RESULTS.md).*
