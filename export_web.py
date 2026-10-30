"""Export the trained models for the browser demo in docs/.

    python export_web.py

Writes docs/assets/manifest.json (vocabulary, settings, scores, tensor layout) and
one float32 little-endian .bin file of weights per model.
"""

import json
import sys
from array import array
from pathlib import Path

import torch

from data import CharVocab, load_text
from models import MODELS
from train import BLOCK_SIZE, CKPT_DIR, RESULTS_DIR

OUT_DIR = Path(__file__).parent / "docs" / "assets"


def export_model(name, metrics):
    ckpt = torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu")
    weights, tensors = array("f"), {}
    for key, t in ckpt["state_dict"].items():
        tensors[key] = {"shape": list(t.shape), "offset": len(weights)}
        weights.extend(t.float().flatten().tolist())
    if sys.byteorder == "big":
        weights.byteswap()
    (OUT_DIR / f"{name}.bin").write_bytes(weights.tobytes())

    m = metrics[name]
    keep = ["params", "train_loss", "val_loss", "train_time_s", "steps", "device"]
    return {
        "file": f"{name}.bin",
        "hparams": ckpt["hparams"],
        "tensors": tensors,
        **{k: m[k] for k in keep},
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text())
    manifest = {
        "block_size": BLOCK_SIZE,
        "vocab": CharVocab(load_text()).chars,
        "models": {name: export_model(name, metrics) for name in MODELS},
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest))
    total = sum(f.stat().st_size for f in OUT_DIR.iterdir())
    print(f"Wrote {OUT_DIR} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
