"""Save/load round-trip verification.

Trains a tiny model briefly, saves its ``state_dict`` to a temp file, loads
into a fresh model, and verifies:

1. Forward logits are bit-for-bit identical between the original and the
   reloaded model.
2. Greedy generations are identical.
3. State-dict tensor shapes and dtypes match.

This is the kind of test a real research codebase desperately needs but
rarely has — silent breakage in checkpointing is one of the most expensive
bugs you can ship in an ML project.

Run:

    python scripts/save_load.py
"""

from __future__ import annotations

import os
import tempfile

import torch
import torch.nn.functional as F

from scripts._common import OpenMythos, build_tiny_mla_config

VOCAB = 32
PROMPT_LEN = 6
SEQ_LEN = 12
BATCH = 32
STEPS = 50  # short — we only need *some* learned weights
LR = 3e-3
N_LOOPS = 3
SEED = 0


def make_batch(rng: torch.Generator, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = torch.randint(0, VOCAB, (batch, PROMPT_LEN), generator=rng)
    seq = torch.cat([prompt, prompt.flip(dims=[1])], dim=1)
    return seq[:, :-1], seq[:, 1:]


def build_model() -> OpenMythos:
    cfg = build_tiny_mla_config()
    cfg.vocab_size = VOCAB
    cfg.max_seq_len = SEQ_LEN
    cfg.max_loop_iters = N_LOOPS
    cfg.expert_dim = 32
    cfg.num_experts = 4
    cfg.num_shared_experts = 1
    cfg.top_k = 2
    return OpenMythos(cfg)


def quick_train(model: OpenMythos) -> None:
    rng = torch.Generator().manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        x, y = make_batch(rng, BATCH)
        logits = model(x, n_loops=N_LOOPS)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()


def greedy(model: OpenMythos, prompt: torch.Tensor, n_new: int) -> torch.Tensor:
    model.eval()
    out = prompt.clone()
    with torch.no_grad():
        for _ in range(n_new):
            logits = model(out, n_loops=N_LOOPS)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
    return out


def main() -> None:
    torch.manual_seed(SEED)

    # -------- train --------
    original = build_model()
    quick_train(original)

    # -------- save --------
    tmp = tempfile.NamedTemporaryFile(prefix="openmythos_save_", suffix=".pt", delete=False)
    tmp.close()
    path = tmp.name
    try:
        torch.save(original.state_dict(), path)
        size_kb = os.path.getsize(path) / 1024
        print(f"[save_load] saved state_dict to {path}  ({size_kb:.1f} KiB)")

        # -------- reload into fresh model --------
        torch.manual_seed(SEED + 999)  # deliberately different init seed
        reloaded = build_model()
        sd = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = reloaded.load_state_dict(sd, strict=True)
        assert not missing, f"missing keys: {missing}"
        assert not unexpected, f"unexpected keys: {unexpected}"
        print(f"[save_load] state_dict loaded ok  "
              f"({len(sd)} tensors, "
              f"{sum(t.numel() for t in sd.values()):,} params)")

        # -------- shape + dtype audit --------
        for k, v_orig in original.state_dict().items():
            v_new = reloaded.state_dict()[k]
            assert v_orig.shape == v_new.shape, f"shape mismatch on {k}"
            assert v_orig.dtype == v_new.dtype, f"dtype mismatch on {k}"
        print("[save_load] all tensors match in shape + dtype")

        # -------- forward equality --------
        rng = torch.Generator().manual_seed(SEED + 7)
        x, _ = make_batch(rng, batch=8)
        original.eval()
        reloaded.eval()
        with torch.no_grad():
            la = original(x, n_loops=N_LOOPS)
            lb = reloaded(x, n_loops=N_LOOPS)
        max_abs_diff = (la - lb).abs().max().item()
        assert torch.equal(la, lb), (
            f"forward logits differ — max_abs_diff={max_abs_diff}"
        )
        print(f"[save_load] forward logits bit-identical  "
              f"(max_abs_diff={max_abs_diff:.2e})")

        # -------- greedy generation equality --------
        prompt = torch.randint(0, VOCAB, (2, PROMPT_LEN), generator=rng)
        ga = greedy(original, prompt, n_new=5)
        gb = greedy(reloaded, prompt, n_new=5)
        assert torch.equal(ga, gb), "greedy generations differ"
        print(f"[save_load] greedy generations bit-identical  "
              f"(prompt={prompt[0].tolist()}  out={ga[0].tolist()})")

    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    print("[save_load] OK")


if __name__ == "__main__":
    main()
