# Personal walkthrough — OpenMythos by example

This is the narrative tour of the `personal` branch. Each script in
`scripts/` is a self-contained experiment that targets one *architectural
claim* of the Recurrent-Depth Transformer. Read them in this order to build
intuition for what the model does, what it actually delivers, and where the
abstractions are leaky.

Everything here runs CPU-only on `torch==2.5.1+cpu`, no `numpy`, no
`transformers`, no `datasets`. The full personal CI pipeline finishes in
under a minute on a free GitHub runner.

---

## Chapter 0 — environment

The package's `open_mythos/__init__.py` eagerly imports `transformers`. That
makes the package useless on a minimal install. So every personal script
loads `open_mythos/main.py` directly via `importlib`:

```python
from scripts._common import OpenMythos, build_tiny_mla_config
```

`scripts/_common.py` builds a tiny config (4 layers, 64 dim, 4 experts top-2,
n_loops=3) and bypasses the package import entirely. This is the single
landmine that everything else assumes.

---

## Chapter 1 — does it run?

- **`tests/test_smoke.py`** — pytest sanity: model builds, forward shape is
  right, both attention variants produce real distributions.
- **`scripts/quick_demo.py`** — first end-to-end forward on random tokens.

If these don't pass, nothing else matters.

---

## Chapter 2 — does it learn?

- **`scripts/train_copy_task.py`** — the canonical reverse-copy task. Train
  on `[a, b, c, d, e, f]` → predict `[f, e, d, c, b, a]`. Tiny vocab (32),
  short sequence (12), 200 steps, AdamW @ 3e-3. This is the *one* training
  recipe everything else in this branch reuses.
- **`scripts/sample.py`** — runs the four temperature/top-k decoding modes
  to confirm sampling code paths work end-to-end.

CE drops below `log(32) ≈ 3.466` quickly. Rev accuracy ~28–30% in 200 steps
(random baseline = 3.1%).

---

## Chapter 3 — what does the recurrent block buy?

This is the *whole pitch* of the architecture. Three independent angles:

- **`scripts/sweep_loops.py`** — sweep `n_loops` at *inference time only*,
  no retraining. Output distribution shifts with depth.
- **`scripts/depth_extrapolation.py`** — train at `n_loops=3`, evaluate at
  `n_loops ∈ {1, 2, 4, 8, 16, 32}`, measure KL between successive depths.
  KL collapses to 0 by `n=4` and stays flat to `n=32`. The recurrence is
  *contractive*; you can over-iterate at inference safely.
- **`scripts/loops_grad_study.py`** — sweep `n_loops` during training,
  report grad norms and ρ(A_disc). All settings stable.
- **`scripts/seed_robustness.py`** — multi-seed reality check. The single-
  seed n=4 win in `loops_grad_study.py` was seed noise; across 4 seeds all
  loop counts are within one std. Lesson: *always* re-run with seeds before
  publishing a "best" hyperparameter.

The takeaway: the recurrent block is *stable*, *depth-extrapolates*, and
*runs cleanly across seeds* — but on a 6-token reverse task there is too
little capacity demand to differentiate loop counts. To see the loop count
matter, the task has to actually need the extra compute.

---

## Chapter 4 — sequence length extrapolation

- **`scripts/extrapolate_seq_len.py`** — train at `prompt_len=6`, evaluate at
  `prompt_len ∈ {4, 6, 8, 10, 12}` (sequences up to 24 tokens, *double* the
  training horizon). 47.7% at trained length, ~14% at extrapolated lengths.
  14% is well above the 3.1% random baseline, so the loop-indexed RoPE
  preserves *some* structure, but the model overfits to the trained horizon.
  Classic position-extrapolation pattern.

---

## Chapter 5 — attention abstraction

- **`scripts/bench_attention.py`** — wall-time MLA vs GQA forward / generate.
- **`scripts/attention_variant_compare.py`** — train both variants on the
  same task with identical hyperparameters; both learn cleanly. Eval-CE
  delta between them is below the per-seed noise.

The attention module is a true drop-in.

---

## Chapter 6 — MoE experts: collapse and rescue

This is the most interesting story in the branch.

- **`scripts/expert_usage.py`** — first-look router distribution. With no
  load-balancing loss, one expert ends up at ~45% of selections, another at
  ~6%. Imbalance is real.
- **`scripts/router_collapse_check.py`** — formal entropy / Gini / dead-
  expert metrics at INIT, MID (50 steps), and FINAL (200 steps). Mid-
  training, the router rebalances on its own (gini 0.46 → 0.29). By step
  200 it drifts back to imbalance (gini 0.45). Without intervention, the
  router does not stay healthy.
- **`scripts/router_balance_loss.py`** — apply the Switch-Transformer load-
  balancing loss `L_lb = N * sum_i(f_i * P_i)` with coefficient 0.05.
  Result: gini collapses 0.37 → 0.11, max-deviation 25pp → 7.3pp,
  normalized entropy 0.81 → 0.99. CE *also* improves slightly. This is the
  textbook fix and it works as advertised.

Read these three together; they tell the full diagnose-then-fix story.

---

## Chapter 7 — adapters and capacity

- **`scripts/lora_adapter_ablation.py`** — sweep `lora_rank ∈ {2, 4, 8, 16}`.
  Negative result on this toy task: rank doesn't matter (eval-CE all within
  0.02 nat). The bottleneck is task capacity, not adapter capacity.

---

## Chapter 8 — inference paths

- **`scripts/save_load.py`** — round-trip the state dict; assert every
  parameter is bit-identical and greedy decoding is bit-identical.
  101,490 parameters, 414 KiB on disk.
- **`scripts/early_exit.py`** — per-token KL halting. Average loops used
  drops to 3.24 against an 8-loop budget (60% compute saving) with no
  accuracy loss vs running full depth.
- **`scripts/kv_cache_speed.py`** — cached vs full-recompute generation.
  1.20× speedup at this tiny scale; first 6 tokens match exactly.
- **`scripts/profile_trace.py`** — cProfile of a forward+backward pass for
  hotspot inspection.

---

## Chapter 9 — invariants

`tests/test_invariants.py` enforces the 5 properties that the rest of this
branch implicitly assumes:

1. Causal mask is real (perturbing token i does not change logits at j < i).
2. KV cache produces *exact* logits, not just same argmax.
3. KL(n → n+1) at large n is no larger than at n=1 (contractive recurrence).
4. ρ(A_disc) < 1 across multiple seeds.
5. LoRAAdapter clamps `loop_t` (depth extrapolation does not crash).

These are the bug-killers for everything above.

---

## How to read the CI log

`.github/workflows/personal-ci.yml` runs (in order):

1. `tests/test_smoke.py`
2. `tests/test_invariants.py`
3. All scripts in `scripts/` via `python -m scripts.X` (the `-m` form is
   load-bearing — `python scripts/X.py` puts `scripts/` on sys.path on
   Linux and breaks the `from scripts._common import ...` line.)

Each script ends with a `[name] OK` line. If you see that line, the
experiment passed its embedded assertions.

---

## Lessons learned (the meta-story)

- **Multi-seed everything** before believing a hyperparameter result. (The
  single-seed n=4 win turned out to be noise.)
- **A toy task can't differentiate sophisticated architectural knobs.** It
  can still validate stability, abstraction integrity, and bug-killers — but
  treat any *quantitative* win on the toy task with extreme suspicion.
- **The MoE collapse story holds even on tiny models.** Ignoring the load-
  balancing loss is not optional.
- **Test invariants, not just outputs.** The 5 invariant tests catch entire
  classes of bugs (causal leakage, cache desync, instability) that a smoke
  test never would.
