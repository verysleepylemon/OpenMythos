"""Profile a forward pass with torch.profiler.

Runs a few warmup forwards then profiles a handful of timed forwards through
the recurrent block, prints the top operators by self-CPU time, and dumps a
Chrome trace file you can drop into ``chrome://tracing`` (or the Perfetto UI).

Run:

    python scripts/profile.py
"""

from __future__ import annotations

import os
import tempfile

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from scripts._common import OpenMythos, build_tiny_mla_config

VOCAB = 256
SEQ_LEN = 16
BATCH = 8
N_LOOPS = 4
WARMUP = 3
ACTIVE = 5
SEED = 0


def build_model(attn: str = "mla") -> OpenMythos:
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = N_LOOPS
    if attn == "gqa":
        cfg.attn_type = "gqa"
        cfg.n_kv_heads = 2
    return OpenMythos(cfg)


def main() -> None:
    torch.manual_seed(SEED)
    model = build_model("mla")
    model.eval()

    x = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))

    out_dir = tempfile.mkdtemp(prefix="openmythos_profile_")
    trace_path = os.path.join(out_dir, "trace.json")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[profile] params={n_params:,}  batch={BATCH}  seq_len={SEQ_LEN}  "
          f"n_loops={N_LOOPS}")
    print(f"[profile] trace will be written to {trace_path}")

    sched = schedule(wait=0, warmup=WARMUP, active=ACTIVE, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU],
        schedule=sched,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(WARMUP + ACTIVE):
                _ = model(x, n_loops=N_LOOPS)
                prof.step()

    prof.export_chrome_trace(trace_path)
    size_kb = os.path.getsize(trace_path) / 1024
    print(f"[profile] wrote chrome trace: {size_kb:.1f} KiB")

    print()
    print("[profile] Top 15 ops by self CPU time:")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=15,
    ))

    print("[profile] OK")


if __name__ == "__main__":
    main()
