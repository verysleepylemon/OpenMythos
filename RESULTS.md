# Personal smoke-test + benchmark results

Run on Windows / Python 3.12 / torch 2.5.1+cpu inside `.venv`. None of the
personal scripts or tests need `transformers`, `datasets`, or `numpy` — they
load `open_mythos/main.py` directly via `scripts/_common.py`.

## Pytest suite (`tests/test_smoke.py`)

```
collected 7 items

tests/test_smoke.py::test_mla_forward_shape PASSED                       [ 14%]
tests/test_smoke.py::test_gqa_forward_shape PASSED                       [ 28%]
tests/test_smoke.py::test_recurrent_injection_is_contractive PASSED      [ 42%]
tests/test_smoke.py::test_generate_extends_sequence PASSED               [ 57%]
tests/test_smoke.py::test_loop_convergence_kl_decreases PASSED           [ 71%]
tests/test_smoke.py::test_model_learns_reverse_copy PASSED               [ 85%]
tests/test_smoke.py::test_kv_cache_matches_full_forward PASSED           [100%]

7 passed, 1 warning in 10.23s
```

The `test_kv_cache_matches_full_forward` case asserts a non-trivial property:
greedy decode using the model's KV cache must produce the same tokens as
greedy decode performed by repeated full-sequence forwards. This catches KV
cache implementation bugs that bench tests would miss.

## quick_demo.py

```
[quick_demo] params: 113,626
[quick_demo] logits shape: (1, 8, 256)
[quick_demo] generated shape: (1, 12)
[quick_demo] spectral radius rho(A) max: 0.3679 (must be < 1)
[quick_demo] OK
```

Tiny MLA config (vocab=256, dim=64, 4 heads, prelude=1, coda=1, 4 experts) —
113k params, contractive recurrent injection (rho ≈ 0.37 < 1).

## sweep_loops.py

```
 n_loops    mean_logit   std_logit    kl_vs_prev
       1       -0.0061      0.1566             —
       2       -0.0068      0.1567      0.000176
       3       -0.0072      0.1568      0.000120
       4       -0.0072      0.1568      0.000000
       6       -0.0072      0.1568      0.000000
       8       -0.0072      0.1568      0.000000
```

Logits converge to a fixed point by `n_loops=4` — exactly the behavior the
contractive recurrent depth design predicts.

## bench_attention.py (MLA vs GQA, tiny config)

```
[bench] seq_len=16  gen_tokens=16  n_loops=3  warmup=3  trials=20
[bench]   MLA  params=113,634  forward=   7.06ms  generate(16tok)=  89.68ms
[bench]   GQA  params=124,242  forward=   4.70ms  generate(16tok)=  74.87ms
```

At this tiny scale GQA is faster (the low-rank Q/KV projections in MLA add
overhead that only pays off at much larger model dimensions where the KV
cache savings dominate). MLA still has fewer params (~10k less), reflecting
the projection compression. Both paths produce correct shapes and pass the
shape tests in `test_smoke.py`.

## train_copy_task.py (real CPU training)

Tiny model, vocab=32, seq_len=12, 200 AdamW steps on a synthetic
"reverse the prompt" copy task:

```
[train] params: 99,294  vocab=32  seq_len=12  n_loops=3
[train] uniform CE baseline: 3.4657
[train] step   1/200  loss=3.4453
[train] step  20/200  loss=3.4196
...
[train] step 200/200  loss=3.2883

[train] wall_time:        8.5s
[train] first_loss:       3.4453
[train] final_loss:       3.2883
[train] eval_loss:        3.1914  (uniform=3.4657)
[train] eval_acc_full:    17.26%
[train] eval_acc_reverse: 28.78%  (random=3.12%)
[train] rho(A) final:     0.4037  (must be < 1)
[train] OK
```

200 steps in 8.5s on CPU pushes loss below uniform (3.19 < 3.47) and reverse-half
accuracy from 3.12% (random) to 28.78% — proof that the full
Prelude → recurrent (n_loops=3) → Coda + LTI injection + MoE FFN stack
trains end-to-end on a laptop. The recurrent injection stays contractive
(rho stays well under 1) throughout training.

## sample.py (generation under several decoding configs)

```
[sample] training tiny model on reverse-copy (300 steps)...
[sample] prompt:           [15, 4, 25, 22, 3, 19]
[sample] expected reverse: [19, 3, 22, 25, 4, 15]
[sample]         greedy  T=1.0  k=  1  gen=[4, 4, 4, 4, 4]  match=1/5
[sample]       low temp  T=0.5  k= 10  gen=[13, 13, 15, 15, 13]  match=0/5
[sample]       balanced  T=1.0  k= 10  gen=[13, 13, 15, 15, 13]  match=0/5
[sample]      high temp  T=1.5  k= 20  gen=[25, 5, 15, 19, 13]  match=0/5
[sample]  uniform top-k  T=1.0  k= 32  gen=[25, 5, 15, 19, 13]  match=0/5
[sample] OK
```

300 steps isn't enough to nail a fully-correct reverse — high-temp / wide
top-k actually pulls in more tokens that appeared in the prompt (`25`, `19`),
hinting at the right behavior emerging. The point of the script is to
exercise the full sampling path end-to-end, not to win the task.

## Notes on the Windows environment

- VC++ Redistributable (2015+) is required.
- `torch==2.5.1+cpu` works cleanly. The newer `torch==2.11.0+cpu` wheel had a
  `_C` DLL load failure even with the redistributable installed.
- The `numpy not initialized` warning from torch is harmless — none of the
  personal scripts depend on numpy.
- The `open_mythos` package `__init__.py` eagerly imports `transformers`, so
  every personal script and the pytest suite uses `scripts/_common.py` to
  load `open_mythos/main.py` directly via `importlib`.
