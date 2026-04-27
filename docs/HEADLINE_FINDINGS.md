# Personal Branch — Headline Findings

> Plain-English summary of what we actually learned by running 35+
> ablations on the OpenMythos recurrent-depth transformer at tiny
> scale on toy reverse-copy / parity / sort tasks.

This doc points at the scripts in [`../scripts/`](../scripts/) and
the per-script numbers in [`../RESULTS.md`](../RESULTS.md).

---

## TL;DR

1. **Recurrent depth (`n_loops`) buys little MEAN ce on toy at this size**
   on the default architecture (`prelude=2, coda=2`). Triangulated across
   THREE independent measurements: `seed_robustness`, `longer_training`,
   and `loops_at_long_train` (at 94% acc convergence, n=4 vs n=1 gap is
   z=-0.12). **But this was a bad measurement** — see point 2 for what
   recurrent depth actually does.
2. **At the optimal architecture (`prelude=1, coda=2`),
   recurrence has a GOLDILOCKS zone in compute budget AND
   its advantage GROWS with task difficulty.**
   `convergence_curve` maps the full STEPS × n_loops grid:
   - **STEPS=500 (under-converged):** loops HURT (ce +0.06, acc -4.7pp)
   - **STEPS=1000-2000 (Goldilocks):** loops HELP (acc +1-3pp,
     variance collapses 3-43×, z reaches -0.96)
   - **STEPS=2500+ (saturated):** all collapse to ce≈0.898

   `task_difficulty_scaling` then shows the n_loops advantage scales
   with task length: at `prompt_len=8`, n=4 vs n=1 acc gap is
   **+3.11pp** AND n=4 collapses variance **43×** (acc std 4.31pp →
   0.10pp). Recurrent depth is a real architectural property — it
   just needs the right architecture, the right compute budget,
   AND a non-trivial task to express itself.

   `loops_at_hard_task` then refines the per-task optimum:
   at `prompt_len=8`, n=2 and n=4 are STATISTICALLY TIED at 99.93%
   acc, but **n=8 REGRESSES** (-1.63pp acc, 16× variance increase).
   Practical recipe: **n_loops=2** at this scale (cheapest of the
   tied winners; 1.30× cost vs n=1 vs 1.62× for n=4).

5. **Capacity changes the optimum: optimal n_loops INCREASES with
   model dim.** `dim_x_loops_at_hard` (dim in {64, 128} ×
   n_loops in {1, 2, 4} at prompt_len=8, STEPS=2000) gives the
   first **statistically significant** recurrence result on this
   branch:
   - `dim=64`:  n=2 ≡ n=4 (tied, both 99.93%); z=-0.72 each
   - `dim=128`: n=4 strictly beats n=2 (z=-3.20 vs -2.67;
                +1.87pp vs +1.68pp acc)

   This rejects H2 ("loops just compensate for capacity"): bigger
   models still need them. It supports H1 ("loops add effective
   depth"). And it tells you the optimum is not fixed — it grows
   with capacity.

   `u_shape_x_dim` then nails the upper bound: at BOTH dim=64
   and dim=128, `n=8` regresses (and regresses HARDER at dim=128,
   acc -2.51pp with std=5.69pp). More capacity does NOT unlock
   arbitrarily deep loops — there is a sharp ceiling.

6. **Recurrence is TASK-DEPENDENT.** `sort_task_loops` (Sort
   task, same recipe as the ReverseCopy sweep that gave z=-3.20)
   produces a clean NULL: z = -0.14 to -0.22 across n_loops in
   {1,2,4}. `sort_task_loops_hard` (vocab=32 to add headroom,
   n=1 only 90.85%) shows loops actively HURT Sort
   (delta_acc=-2.37pp at n=4, z=+0.99).

   `rotate_task_loops` triangulates the picture with a third
   task (Rotate by k=4): clear benefit at n=2 (+6.36pp acc, 43×
   variance collapse from ±7.77pp to ±0.18pp), but n=4 collapses
   (-2.69pp acc). Cross-task at dim=128 prompt_len=8:
   - ReverseCopy: optimum n=4, +1.87pp acc, z=-3.20 (smooth)
   - Rotate:      optimum n=2, +6.36pp acc, z=-0.73 (sharp; n=4 hurts)
   - Sort:        no optimum; n=4 HURTS by 2.37pp at vocab=32

   Pattern: tasks that require explicit cross-positional ROUTING
   (ReverseCopy, Rotate) benefit from loops. Tasks that don't
   (Sort) do not, and may be harmed by them. The complete recipe
   is now: **right arch + right compute + hard enough task +
   task that benefits from iteration.** And the per-task optimum
   shifts (Rotate=2, ReverseCopy=4 at the same scale) so loop
   count must be tuned per task class.
3. **Stacked depth still beats recurrent depth on raw ce** —
   `recurrent_vs_stacked` shows K=4 stacked beats K=4 recurrent by
   0.09 ce (z=2.69), but pays 1.77× more params.
4. **The architecture is asymmetric.** `prelude_coda_depth` shows
   `prelude=1, coda=2` is optimal (z=2.16 vs `1/1`); MORE prelude
   actively HURTS (+0.06 to +0.11 ce, with much higher variance).
5. **Init seed is the dominant noise source.** `init_seed_variance`
   shows fixing data and varying only model init produces *higher*
   variance than varying both. Init-noise floor on default recipe = 0.026 ce.
6. **Three knobs cleanly move the needle.** Batch size (z=3.71),
   training length (z=8-11), and the asymmetric coda-depth (z=2.16).

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
