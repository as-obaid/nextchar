"""Build the comparison from the saved checkpoints and metrics.

    python report.py
    python report.py --temperature 1.0 --length 800

Writes results/results.md (table), results/loss_curves.png, results/context_curve.png,
and results/samples.md.
"""

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from data import eval_windows, load_splits  # noqa: E402
from models import MODELS, generate  # noqa: E402
from train import BLOCK_SIZE, CKPT_DIR, RESULTS_DIR, SEED  # noqa: E402

LABELS = {"bigram": "Bigram", "lstm": "LSTM", "transformer": "Transformer"}
COLORS = {"bigram": "#2a78d6", "lstm": "#eb6834", "transformer": "#1baf7a"}
INK, GRID, AXIS, SURFACE = "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
PROMPTS = ["ROMEO:\n", "KING RICHARD III:\n", "Thou art"]


def load_model(name, vocab_size):
    ckpt = torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu")
    model = MODELS[name](vocab_size, **ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def count_bigram_val_loss(train_data, val_data, vocab_size, alpha=0.1):
    """Validation loss of a bigram estimated by counting next-character frequencies
    in the training set (add-alpha smoothing), on the same windows as the models."""
    counts = torch.full((vocab_size, vocab_size), alpha)
    pairs = (train_data[:-1], train_data[1:])
    counts.index_put_(pairs, torch.ones(len(train_data) - 1), accumulate=True)
    logp = (counts / counts.sum(1, keepdim=True)).log()
    x, y = eval_windows(val_data, BLOCK_SIZE)
    return -logp[x, y].mean().item()


@torch.no_grad()
def loss_by_position(model, x, y, batch_size=256):
    """Mean loss at each window position; position t has t earlier characters of context."""
    total = torch.zeros(x.size(1))
    for i in range(0, len(x), batch_size):
        logits = model(x[i : i + batch_size])
        total += F.cross_entropy(
            logits.transpose(1, 2), y[i : i + batch_size], reduction="none"
        ).sum(0)
    return total / len(x)


def write_table(metrics, names, count_bigram_loss):
    rows = [
        "| Model | Parameters | Steps | Best step | Train loss | Val loss | Val perplexity "
        "| Training time | Device |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in names:
        m = metrics[name]
        rows.append(
            f"| {LABELS[name]} | {m['params']:,} | {m['steps']:,} | {m['best_step']:,} | "
            f"{m['train_loss']:.4f} | {m['val_loss']:.4f} | {math.exp(m['val_loss']):.2f} | "
            f"{m['train_time_s']:.1f} s | {m['device']} |"
        )
    note = (
        "\nLoss is mean cross-entropy in nats per character, measured with dropout off on the "
        "same fixed windows for every model. Perplexity is exp(loss). The checkpoint kept is "
        "the evaluation with the lowest validation loss (best step). Training time is for all "
        "steps and excludes evaluation.\n\n"
        f"Reference: a bigram built by counting character pairs in the training set "
        f"(add-0.1 smoothing) has validation loss {count_bigram_loss:.4f}.\n"
    )
    (RESULTS_DIR / "results.md").write_text("\n".join(rows) + "\n" + note)


def style(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set(title=title, xlabel=xlabel, ylabel="cross-entropy (nats per char)")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK)
    for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
        text.set_color(INK)


def plot_curves(metrics, names):
    fig, (ax_step, ax_time) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for name in names:
        h = metrics[name]["history"]
        steps = [r["step"] for r in h]
        val = [r["val_loss"] for r in h]
        ax_step.plot(steps, [r["train_loss"] for r in h], color=COLORS[name], lw=2, ls="--")
        ax_step.plot(steps, val, color=COLORS[name], lw=2)
        ax_time.plot([r["time_s"] for r in h], val, color=COLORS[name], lw=2)

    handles = [Line2D([], [], color=COLORS[n], lw=2, label=LABELS[n]) for n in names]
    handles += [
        Line2D([], [], color=INK, lw=2, label="validation"),
        Line2D([], [], color=INK, lw=2, ls="--", label="train"),
    ]
    ax_step.legend(handles=handles, frameon=False, labelcolor=INK)
    style(ax_step, "Loss by training step", "step")
    style(ax_time, "Validation loss by training time", "seconds (evaluation excluded)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)


def plot_context(models, x, y):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    visible = range(1, x.size(1) + 1)  # characters the model can see, including the current one
    for name, model in models.items():
        ax.plot(
            visible, loss_by_position(model, x, y), color=COLORS[name], lw=2, label=LABELS[name]
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(
        [1, 2, 4, 8, 16, 32, 64, 128], labels=["1", "2", "4", "8", "16", "32", "64", "128"]
    )
    ax.legend(frameon=False, labelcolor=INK)
    style(ax, "Validation loss by available context", "characters of context (log scale)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "context_curve.png", dpi=150)


def write_samples(vocab, models, temperature, length):
    out = [
        "# Generated samples\n",
        f"Every model gets the same prompts, temperature ({temperature}), length ({length} new "
        f"characters), and sampling seed ({SEED}), and sees at most the last {BLOCK_SIZE} "
        "characters.\n",
    ]
    for name, model in models.items():
        out.append(f"## {LABELS[name]}\n")
        for prompt in PROMPTS:
            torch.manual_seed(SEED)
            idx = torch.tensor([vocab.encode(prompt)])
            text = vocab.decode(generate(model, idx, length, BLOCK_SIZE, temperature)[0].tolist())
            out.append(f"Prompt: `{prompt!r}`\n\n```text\n{text}\n```\n")
    (RESULTS_DIR / "samples.md").write_text("\n".join(out))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--length", type=int, default=500, help="new characters per sample")
    args = parser.parse_args()

    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text())
    names = [n for n in MODELS if n in metrics]
    vocab, train_data, val_data = load_splits()
    models = {name: load_model(name, len(vocab)) for name in names}

    write_table(metrics, names, count_bigram_val_loss(train_data, val_data, len(vocab)))
    plot_curves(metrics, names)
    # Overlapping windows from every start offset, so each position averages over
    # different characters rather than the same ones.
    plot_context(models, *eval_windows(val_data, BLOCK_SIZE, max_windows=16384, stride=1))
    write_samples(vocab, models, args.temperature, args.length)
    print(f"Wrote results.md, loss_curves.png, context_curve.png, and samples.md to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
