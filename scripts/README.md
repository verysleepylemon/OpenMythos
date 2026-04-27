# Personal branch — script index

All scripts here are CPU-only, no `numpy` / `transformers` / `datasets`,
runnable with `python -m scripts.X` from the repo root.
The full personal CI matrix runs in well under a minute on a free runner.

For the narrative walkthrough that ties these together, read
[`docs/PERSONAL_WALKTHROUGH.md`](../docs/PERSONAL_WALKTHROUGH.md).

For per-script results and exact log output, read [`RESULTS.md`](../RESULTS.md).

## Index

| # | Script | Category | What it shows |
|---|--------|----------|---------------|
| 1 | `quick_demo.py` | smoke | First end-to-end forward on random tokens. |
| 2 | `sweep_loops.py` | depth | Output distribution shifts with `n_loops` at inference time. |
| 3 | `bench_attention.py` | attention | MLA vs GQA wall-time comparison (forward + generate). |
| 4 | `train_copy_task.py` | training | Canonical reverse-copy training recipe (the one everyone reuses). |
| 5 | `sample.py` | sampling | Greedy / low-T / balanced / high-T / wide-top-k decoding paths. |
| 6 | `depth_extrapolation.py` | depth | Train at K=3, eval K∈[1..32]; KL collapses to 0 by K=4. |
| 7 | `expert_usage.py` | MoE | Raw routing distribution; reveals imbalance without an LB loss. |
| 8 | `save_load.py` | infra | State_dict round-trip is bit-identical (params + greedy decode). |
| 9 | `early_exit.py` | inference | Per-token KL halting saves ~60% compute with no accuracy loss. |
| 10 | `loops_grad_study.py` | depth | Single-seed sweep of `n_loops` with grad norms and ρ(A_disc). |
| 11 | `extrapolate_seq_len.py` | extrapolation | Train at PL=6, eval up to PL=12; ~14% at extrapolated lengths. |
| 12 | `router_collapse_check.py` | MoE | Formal entropy/Gini metrics at INIT/MID/FINAL — drift then collapse. |
| 13 | `lora_adapter_ablation.py` | adapter | LoRA rank sweep — negative result: rank doesn't matter on toy task. |
| 14 | `kv_cache_speed.py` | inference | Cached vs full-recompute decoding; 1.20× speedup, exact match. |
| 15 | `attention_variant_compare.py` | attention | Train MLA vs GQA matched; both learn, gap below seed noise. |
| 16 | `router_balance_loss.py` | MoE | Adding LB loss collapses gini 0.37 → 0.11 (strong positive). |
| 17 | `seed_robustness.py` | depth | Multi-seed: single-seed `n=4` win was seed noise. |
| 18 | `cosine_lr_warmup.py` | training | Flat vs cosine+warmup; flat slightly wins on this short toy budget. |
| 19 | `weight_decay_ablation.py` | training | AdamW wd in {0, 0.01, 0.1} multi-seed; below-noise effect. |
| 20 | `grad_clip_ablation.py` | training | clip in {0.5, 1.0, 2.0, no-clip}; clip=0.5 fires 99.8% and hurts. |
| 21 | `batch_size_curve.py` | training | BATCH 8->64 at fixed STEPS; ce 3.41->2.97 (z=3.71, real win). |
| 22 | `model_size_scaling.py` | scaling | dim {32, 64, 128}; saturates at dim=64 on the toy task. |
| 23 | `eval_helper_smoke.py` | infra | End-to-end smoke for the shared `_eval.py` helper. |
| 24 | `parity_loop_sweep.py` | depth | Cumulative XOR; all loop counts pinned at chance (Hahn 2020). |
| 25 | `sort3_loop_sweep.py` | depth | Sort-3 is too easy — n=1 already 99.7% acc; depth doesn't help. |
| 26 | `lr_peak_curve.py` | training | Peak LR {1e-3, 3e-3, 1e-2}; 3e-3 wins, 1e-2 unstable (3σ worse). |
| 27 | `topk_experts_sweep.py` | MoE | k in {1,2,3}; k=1 underfits (+0.05 ce, 1.3σ); k=2 = k=3 on ce. |
| 28 | `shared_expert_ablation.py` | MoE | n_shared in {0,1,2}; default n_shared=1 wins; 0 and 2 both noisier. |
| 29 | `init_seed_variance.py` | noise | Fix data, vary init only — init explains ALL of the seed-noise budget. |
| 30 | `longer_training.py` | training | 200→1000 steps drops ce 0.7 (z=8-11); n_loops gap still doesn't open. |
| 31 | `recurrent_vs_stacked.py` | depth | At K=4, stacked WINS +0.09 ce (z=2.69) for 1.77x params; recurrence buys nothing. |
| 32 | `prelude_coda_depth.py` | depth | prelude=1, coda=2 is optimal (ce 1.63 z=2.2 vs 1/1); MORE prelude actively hurts. |
| 33 | `mla_kv_rank_sweep.py` | mla | Default kv_lora_rank=16 sits at elbow; 8 underfits +0.05ce; 32 wastes 6k params. |
| 34 | `expert_count_sweep.py` | moe | n_experts in {2,4,8} all within init-noise floor; n=2 cheapest-not-worse (saves 12k params). |
| 35 | `loops_at_long_train.py` | depth | At 1000 steps (94% acc convergence) n_loops gap z=-0.16: recurrence STILL doesn't pay. |
| 36 | `loops_at_optimal_arch.py` | depth | At prelude=1+coda=2 optimal arch, recurrence shows MONOTONIC ce improvement and collapses variance 3x. |
| 37 | `loops_at_optimal_arch_long.py` | depth | At STEPS=2500 (99.6% acc), all loop counts converge to identical ce — variance reduction was a training-dynamics effect, not steady-state. |
| 38 | `convergence_curve.py` | depth | Goldilocks zone: at STEPS=500 loops HURT (z=+0.40), at STEPS=1000-2000 loops HELP (z=-0.71 to -0.96 + 3x variance drop), at STEPS=2500 all collapse. |
| 39 | `task_difficulty_scaling.py` | depth | Loop advantage GROWS with prompt_len: acc gap 0.94pp -> 2.47pp -> 3.11pp at prompt_len 4/6/8. At prompt_len=8, n=4 collapses variance 43x (acc std 4.31pp -> 0.10pp). |
| 40 | `loops_at_hard_task.py` | depth | At prompt_len=8, U-shape revealed: n=2 == n=4 (both 99.93% acc, near-zero variance), n=8 REGRESSES. n=2 is the cheapest winner (1.30x cost vs n=1). |
| 41 | `dim_x_loops_at_hard.py` | depth | FIRST z>3 recurrence result: at dim=128 + prompt_len=8, n=4 vs n=1 z=-3.20 (statistically significant). Bigger model -> recurrence MORE valuable, supports H1 (depth-amplifier). |
| 42 | `u_shape_x_dim.py` | depth | n=8 sweep at dim in {64,128}. Confirms U-shape at BOTH scales: n=8 regresses harder at dim=128 (acc -2.51pp, std 5.69pp) than at dim=64 (acc +1.48pp, std 1.56pp). More capacity does NOT unlock more loops. |
| 43 | `sort_task_loops.py` | depth | NULL: Sort task at dim=128 prompt_len=8 sees no recurrence benefit (z=-0.14 to -0.22, delta_acc ~ 0pp). Recurrence is TASK-DEPENDENT - ReverseCopy benefits, Sort does not. |
| 44 | `sort_task_loops_hard.py` | depth | Control: Sort with vocab=32 (headroom: n=1 only 90.85%). Loops actively HURT (delta_acc -2.37pp at n=4, z=+0.99). NULL was task-class, not headroom. |
| 45 | `rotate_task_loops.py` | depth | Triangulation: Rotate (cross-positional like ReverseCopy). n=2 wins big (+6.36pp acc, 43x variance collapse), n=4 collapses (-2.69pp). Confirms routing tasks benefit from loops; per-task optimum differs (ReverseCopy=4, Rotate=2). |
| 46 | `harder_task_loops.py` | depth | ReverseCopy at prompt_len=12 dim=128: non-monotonic across n_loops. n=2 HURTS (-7.18pp acc), n=4 HELPS (+5.57pp, 2.4x variance reduction). Optimum jumps with task length. |
| 47 | `harder_n8.py` | depth | n=8 at prompt_len=12 dim=128: +13.17pp acc (z=-1.30). Largest gain ever measured. The U-shape "ceiling" was a task-length artifact, not an intrinsic optimization limit. |
| 48 | `profile_trace.py` | infra | cProfile of forward+backward for hotspot inspection. |

## Tests

| File | What it asserts |
|------|-----------------|
| `tests/test_smoke.py` | Model builds, forward shapes, both attn variants produce real distributions. |
| `tests/test_invariants.py` | 5 hard invariants: causal mask, KV cache exact, KL monotone, ρ<1, LoRA clamp. |

## Common pattern

Every script:

```python
from scripts._common import OpenMythos, build_tiny_mla_config

cfg = build_tiny_mla_config()
cfg.vocab_size = VOCAB
cfg.max_seq_len = SEQ_LEN
cfg.max_loop_iters = N_LOOPS
model = OpenMythos(cfg)
# ... train / eval / measure ...
print("[name] OK")
```

The shared loader in `scripts/_common.py` bypasses the package
`__init__.py` (which eagerly imports `transformers`). It's the single
landmine that makes the entire CPU-only pipeline work.

## Headline findings (so far)

- **Architecture is stable.** ρ(A_disc) < 1 across all seeds; KL converges
  with depth; depth-extrapolation works to at least K=32.
- **MoE collapses without help.** Auxiliary load-balancing loss is mandatory
  (gini 0.37 → 0.11 with `LB_COEF=0.05`).
- **Single-seed wins are usually noise.** The toy task can validate
  abstractions but cannot discriminate hyperparameter values; always re-run
  with multiple seeds.
- **KV cache is exact, not just approximate.** The invariant test asserts
  raw-logit equality with the full forward pass.
