"""Generation demo: trains the tiny model on the reverse-copy task and then
samples completions at a sweep of temperature and top-k settings.

This is the end-to-end "model actually does something" demo: prompt -> learned
behavior -> deterministic vs sampled output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts._common import OpenMythos, build_tiny_mla_config  # noqa: E402

VOCAB = 32
PROMPT_LEN = 6
SEQ_LEN = PROMPT_LEN * 2
BATCH = 32
STEPS = 300
LR = 3e-3
N_LOOPS = 3


def reverse_batch(rng: torch.Generator, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = torch.randint(0, VOCAB, (batch, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def quick_train(model: OpenMythos) -> None:
    rng = torch.Generator().manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(STEPS):
        x, y = reverse_batch(rng, BATCH)
        loss = F.cross_entropy(
            model(x, n_loops=N_LOOPS).reshape(-1, VOCAB), y.reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()


def main() -> None:
    torch.manual_seed(0)
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = N_LOOPS
    model = OpenMythos(cfg)
    print(f"[sample] training tiny model on reverse-copy ({STEPS} steps)...")
    quick_train(model)

    # Build a single deterministic prompt and sample under several configs
    prompt_rng = torch.Generator().manual_seed(7)
    prompt = torch.randint(0, VOCAB, (1, PROMPT_LEN), generator=prompt_rng)
    expected = prompt.flip(dims=[1])[0].tolist()
    print(f"[sample] prompt:           {prompt[0].tolist()}")
    print(f"[sample] expected reverse: {expected}")

    configs = [
        ("greedy",         dict(temperature=1.0, top_k=1)),
        ("low temp",       dict(temperature=0.5, top_k=10)),
        ("balanced",       dict(temperature=1.0, top_k=10)),
        ("high temp",      dict(temperature=1.5, top_k=20)),
        ("uniform top-k",  dict(temperature=1.0, top_k=VOCAB)),
    ]

    for name, kwargs in configs:
        torch.manual_seed(42)  # so each config is reproducible
        with torch.no_grad():
            out = model.generate(
                prompt,
                max_new_tokens=PROMPT_LEN - 1,
                n_loops=N_LOOPS,
                **kwargs,
            )
        gen = out[0, PROMPT_LEN:].tolist()
        match = sum(1 for a, b in zip(gen, expected[1:]) if a == b)
        print(
            f"[sample] {name:>14}  "
            f"T={kwargs['temperature']}  k={kwargs['top_k']:>3}  "
            f"gen={gen}  match={match}/{len(gen)}"
        )

    print("[sample] OK")


if __name__ == "__main__":
    main()
