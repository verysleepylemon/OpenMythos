# Ship Recipe — OpenMythos Recurrent-Depth (personal branch)

> One page. The smallest set of settings that delivers a real, replicated win
> over the no-recurrence baseline. Everything outside this recipe is null,
> mixed, or actively harmful at 6-seed audit.

## TL;DR

If your task and dimensions match the **green zone** below, set
`max_loop_iters = 4`. Otherwise leave `max_loop_iters = 1`.

| Knob               | Ship value             |
|--------------------|------------------------|
| `prelude_layers`   | **1**                  |
| `coda_layers`      | **2**                  |
| `dim`              | **128**                |
| `expert_dim`       | `dim // 2` (= 64)      |
| `max_loop_iters`   | **4** (only in green zone, else **1**) |
| `grad_clip`        | **1.0**                |
| training steps     | ~2000                  |

## Green zone (where `n_loops=4` ships)

`max_loop_iters = 4` is recommended **iff all** of the following hold:

1. `dim >= 128`
2. task prompt length **pl ∈ {6, 8}** — verified at 6 seeds by
   `pl_sweep_n4.py`. At pl=10 the mean still improves (+2.85 pp) but
   *variance increases* (+3.19 pp std, one seed collapsed to 72%) so
   it is **not** in the green zone. At pl≥12 loops actively hurt
   (`headline_replicate.py`).
3. task is **ReverseCopy-style position reversal**. Rotate (`rotate_replicate.py`,
   6 seeds) is NULL: n=2 +0.81pp z=+0.07, n=4 -1.31pp z=+0.22. The
   3-seed Rotate win in `rotate_task_loops.py` was a single-seed
   artifact. Sort-style tasks did not benefit either
   (`sort_task_loops.py`). Treat task-class generalization as
   **not yet established** — ship `n=1` for any task other than
   ReverseCopy until you replicate at 6 seeds.
4. baseline `n=1` accuracy is in `[94%, 99%]` — i.e. there is real
   headroom for recurrence to close. If `n=1` is already ≥ 99.5%
   (saturated) loops only add variance.

If **any** of those fail: ship `max_loop_iters = 1`.

## What you get in the green zone

From `replicate_pl8_n4.py` and `pl_sweep_n4.py` (6 seeds each, dim=128,
prelude=1, coda=2, STEPS=2000, grad_clip=1.0):

| pl | n_loops | accuracy           | per-seed range      | notes |
|----|---------|--------------------|---------------------|-------|
| 6  | 1       | 94.43 ± 5.98 %     | [85.48, 99.38]      | wide variance |
| 6  | **4**   | **98.16 ± 2.79 %** | [92.48, 99.61]      | +3.73 pp, 2.1× variance reduction |
| 8  | 1       | 98.41 ± 0.85 %     | [97.36, 99.83]      |        |
| 8  | **4**   | **99.81 ± 0.17 %** | **[99.51, 100.00]** | +1.40 pp, **5× variance collapse** |

The variance collapse is the dominant practical win. It means deployment
quality doesn't depend on getting lucky with the init seed — every seed
trained with `n_loops=4` lands in a tight band near the ceiling.

Out-of-zone reference (`pl_sweep_n4.py`):

| pl | n_loops | accuracy            | notes |
|----|---------|---------------------|-------|
| 10 | 1       | 86.20 ± 7.15 %      | model out of regime |
| 10 | 4       | 89.05 ± 10.34 %     | mean up but variance up; one seed collapsed to 72% |

## Cost

`n_loops=4` vs `n_loops=1` on the same model: ~**1.55× wall clock** training,
identical params, identical inference latency *if* you keep the loop in
serving (or 1× latency if you drop loops at serve time and accept slight
quality loss; not measured).

## What does NOT ship (killed by 6-seed audit)

- **`harder_n8`'s +13.17 pp at pl=12.** 6-seed audit
  (`headline_replicate.py`) found n=4 = -2.21 pp and n=8 = -5.86 pp.
  The original 3-seed claim sat well inside the ±12 pp seed std.
  **Recipe at pl=12 reverts to `n_loops=1`.**
- **Depth-aware clipping recipe** (`clip_universal.py`).
  `clip_recipe_pl.py` showed clip=2.0 rescue is pl=12-only:
  pl=8 −2.36 pp, pl=12 +10.12 pp, pl=14 −29.65 pp (acc std 34.6 pp).
  **Keep `grad_clip = 1.0`.**

## Methodology rule (earned the hard way)

> A 3-seed mean shift **inside** the 3-seed std band is an artifact, not
> an effect. Any shippable recurrent-depth claim on this stack requires
> **≥ 6 seeds** and either (a) `|delta| > 2 × pooled_std` or (b) clear
> variance reduction (`std_ratio ≤ 0.5`).

The recipe above passes (b): `acc_std` collapses 0.85 pp → 0.17 pp (5×).

## Reproduce

```bash
python -m scripts.replicate_pl8_n4
# expect: n=4 acc 99.81 +/- 0.17, n=1 acc 98.41 +/- 0.85
```

CI step is wired up; see `.github/workflows/personal-ci.yml`.
