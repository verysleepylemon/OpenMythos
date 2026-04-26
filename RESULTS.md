# Personal smoke-test results

Run on Windows / Python 3.12 / torch 2.5.1+cpu inside `.venv`.

## quick_demo.py

```
[quick_demo] params: 113,626
[quick_demo] logits shape: (1, 8, 256)
[quick_demo] generated shape: (1, 12)
[quick_demo] spectral radius rho(A) max: 0.3679 (must be < 1)
[quick_demo] OK
```

Tiny MLA config (vocab=256, dim=64, 4 heads, prelude=1, coda=1, 4 experts) — 113k params, contractive recurrent injection (rho ≈ 0.37 < 1).

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

Logits converge to a fixed point by n_loops=4 — exactly the behavior the contractive recurrent depth design predicts.

## Notes

- The `open_mythos` package `__init__.py` eagerly imports `transformers`, which is heavy and not needed for the core model. Both scripts load `open_mythos/main.py` directly via `importlib` to keep the smoke test minimal (torch only).
- VC++ Redistributable (2015+) and torch 2.5.1+cpu are required on Windows; torch 2.11.0+cpu wheel had a `_C` DLL load failure with the same redistributable installed.
