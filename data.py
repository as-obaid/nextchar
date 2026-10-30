"""Tiny Shakespeare: download, character vocabulary, train/val split, and batching."""

import urllib.request
from pathlib import Path

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path(__file__).parent / "data" / "input.txt"


def load_text():
    """The corpus ships in data/input.txt; it is downloaded only if that file is missing."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        print(f"Downloading Tiny Shakespeare to {DATA_PATH}")
        with urllib.request.urlopen(URL) as response:
            DATA_PATH.write_bytes(response.read())  # written only once the download is complete
    return DATA_PATH.read_text(encoding="utf-8")


class CharVocab:
    """One id per distinct character in the corpus."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def load_splits(val_frac=0.1):
    """Cut the encoded text into two contiguous blocks *before* any windowing,
    so no training window can contain validation characters."""
    text = load_text()
    vocab = CharVocab(text)
    data = torch.tensor(vocab.encode(text), dtype=torch.long)
    n_train = int(len(data) * (1 - val_frac))
    return vocab, data[:n_train], data[n_train:]


def get_batch(data, batch_size, block_size, generator):
    """Random training windows; targets are the inputs shifted one character left."""
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def eval_windows(data, block_size, max_windows=None, stride=None):
    """Fixed windows, so every model is scored on the same characters. They do not
    overlap unless stride < block_size. If there are more than max_windows, keep an
    evenly spaced subset."""
    starts = torch.arange(0, len(data) - block_size, stride or block_size)
    if max_windows is not None and len(starts) > max_windows:
        starts = starts[torch.linspace(0, len(starts) - 1, max_windows).long()]
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y
