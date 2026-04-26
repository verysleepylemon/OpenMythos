# Personal Fork Notes — verysleepylemon/OpenMythos

This is my personal fork of [kyegomez/OpenMythos](https://github.com/kyegomez/OpenMythos) (10.5k★).

## Why I forked

OpenMythos implements a **Recurrent-Depth Transformer** — a clean reference architecture
for compute-adaptive reasoning (Prelude → looped Recurrent Block → Coda, switchable
MLA/GQA attention, sparse MoE FFN). I forked to:

1. Run small CPU experiments locally without depending on PyPI.
2. Add personal helper scripts that aren't worth upstreaming.
3. Track my own modifications without diverging from the upstream `main`.

## Local setup

```powershell
cd C:\Users\A\Downloads\OpenMythos
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\quick_demo.py
```

PyTorch is heavy (~800 MB on Windows). First install can take 10+ minutes on a typical
home connection. Subsequent runs are instant.

## Sync with upstream

```powershell
git fetch upstream
git merge upstream/main
git push origin main
```

`origin` = `verysleepylemon/OpenMythos` (my fork)
`upstream` = `kyegomez/OpenMythos` (canonical)

## What's mine vs upstream

Anything outside this list is upstream code, untouched:

- `PERSONAL_NOTES.md` — this file
- `scripts/quick_demo.py` — minimal CPU smoke test (smaller than `example.py`)
- `scripts/sweep_loops.py` — vary `n_loops` to see how depth affects logits

Keeping these in `scripts/` (a directory upstream doesn't use) avoids merge conflicts.
