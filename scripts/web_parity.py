"""Check the browser port (docs/model.js) against PyTorch.

    python export_web.py && python scripts/web_parity.py

Feeds every model a short prompt and 200 validation characters (long enough for
the 128-character window to slide) one character at a time, as generate() does,
and compares the next-character logits after each step. Also compares the
transformer's attention weights. Needs Node.js.
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import load_splits  # noqa: E402
from models import MODELS  # noqa: E402
from train import BLOCK_SIZE, CKPT_DIR  # noqa: E402

TOLERANCE = 1e-4


def load(name, vocab_size):
    ckpt = torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu")
    model = MODELS[name](vocab_size, **ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


@torch.no_grad()
def step_logits(model, ids):
    """Next-character logits after each prefix, cropped to the window as in generate()."""
    return [
        model(ids[None, max(0, t - BLOCK_SIZE) : t])[0, -1].tolist() for t in range(1, len(ids) + 1)
    ]


@torch.no_grad()
def last_query_attention(model, ids):
    """Attention weights of the last position in every layer: [layer][head][position]."""
    captured = []

    def hook(attn, inputs, output):
        x = inputs[0]  # the block's ln1 output
        B, T, C = x.shape
        q, k, _ = attn.qkv(x).split(C, dim=2)
        q = q.view(B, T, attn.n_head, -1).transpose(1, 2)[:, :, -1:]
        k = k.view(B, T, attn.n_head, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        captured.append(att.softmax(dim=-1)[0, :, 0].tolist())

    handles = [block.attn.register_forward_hook(hook) for block in model.blocks]
    model(ids[None, -BLOCK_SIZE:])
    for h in handles:
        h.remove()
    return captured


def main():
    vocab, _, val_data = load_splits()
    sequences = {
        "prompt": torch.tensor(vocab.encode("ROMEO:\nBut soft")),
        "sliding window": val_data[:200],
    }
    reference = {"tolerance": TOLERANCE, "models": {}}
    for name in MODELS:
        model = load(name, len(vocab))
        ref = {
            "sequences": {
                k: {"ids": v.tolist(), "logits": step_logits(model, v)}
                for k, v in sequences.items()
            }
        }
        if name == "transformer":
            ref["attention"] = {
                k: {"ids": v.tolist(), "weights": last_query_attention(model, v)}
                for k, v in sequences.items()
            }
        reference["models"][name] = ref

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(reference, f)
    result = subprocess.run(["node", str(ROOT / "scripts" / "web_parity.cjs"), f.name])
    Path(f.name).unlink()
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
