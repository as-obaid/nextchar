"""Tune the two neural models with one shared search procedure.

    python tune.py                      # both models, one after the other
    python tune.py --model transformer  # one model (two processes can run side by side)

A coordinate search in three rounds: learning rate, optimizer, then shape. Each round
tries its changes on top of the best settings so far and keeps the best one only if it
lowers validation loss. Both models get the same rounds and the same number of runs.
Every run is 3,000 steps on the same batches; checkpoints are not touched. Writes
results/tuning.json and results/tuning.md.
"""

import argparse
import json

from data import load_splits
from train import BLOCK_SIZE, RESULTS_DIR, eval_sets, pick_device, train_one

TUNE_STEPS = 3000
LABELS = {"lstm": "LSTM", "transformer": "Transformer"}

# Where the search starts: the settings of the original comparison (AdamW defaults).
BASELINE = {
    "lstm": dict(
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        hparams=dict(n_embd=128, hidden_size=160, n_layer=2, dropout=0.1),
    ),
    "transformer": dict(
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        hparams=dict(
            n_embd=128, n_head=4, n_layer=2, ff_dim=512, block_size=BLOCK_SIZE, dropout=0.1
        ),
    ),
}

# (label, change) pairs per round. The shape round trades width for depth and keeps
# every LSTM/transformer pairing within 5% in parameter count (399k-430k).
ROUNDS = [
    (
        "learning rate",
        {m: [("lr 4e-3", dict(lr=4e-3)), ("lr 8e-3", dict(lr=8e-3))] for m in BASELINE},
    ),
    (
        "optimizer",
        {
            m: [("β2 0.99, weight decay 0.1", dict(betas=(0.9, 0.99), weight_decay=0.1))]
            for m in BASELINE
        },
    ),
    (
        "shape",
        {
            "lstm": [
                (
                    "3 layers × 128",
                    dict(hparams=dict(n_embd=128, hidden_size=128, n_layer=3, dropout=0.1)),
                )
            ],
            "transformer": [
                (
                    "4 layers × 88",
                    dict(
                        hparams=dict(
                            n_embd=88,
                            n_head=4,
                            n_layer=4,
                            ff_dim=352,
                            block_size=BLOCK_SIZE,
                            dropout=0.1,
                        )
                    ),
                )
            ],
        },
    ),
]


def search(name, vocab_size, train_data, evals, device):
    runs = []

    def run(round_name, setting, config):
        m = train_one(
            name, vocab_size, train_data, evals, TUNE_STEPS, device, config=config, save=False
        )
        val_loss = m["history"][-1]["val_loss"]  # loss at the end of the run
        runs.append(dict(round=round_name, setting=setting, params=m["params"], val_loss=val_loss))
        return val_loss

    best = BASELINE[name]
    best_loss = run("start", "original settings", best)
    best_run = runs[-1]
    for round_name, candidates in ROUNDS:
        start = best
        for setting, change in candidates[name]:
            loss = run(round_name, setting, {**start, **change})
            if loss < best_loss:
                best, best_loss, best_run = {**start, **change}, loss, runs[-1]
        best_run["kept"] = True
    for r in runs:
        r.setdefault("kept", False)
    return {"runs": runs, "best": best, "best_val_loss": best_loss}


def write_table(results):
    lines = []
    for name in BASELINE:
        if name not in results:
            continue
        lines += [
            f"## {LABELS[name]}",
            "",
            "| Round | Setting | Parameters | Val loss | Kept |",
            "| --- | --- | ---: | ---: | :---: |",
        ]
        for r in results[name]["runs"]:
            kept = "✓" if r["kept"] else ""
            lines.append(
                f"| {r['round']} | {r['setting']} | {r['params']:,} | "
                f"{r['val_loss']:.4f} | {kept} |"
            )
        lines.append("")
    lines.append(
        f"{TUNE_STEPS:,} steps per run on the same batches. Validation loss in nats per "
        "character. A ✓ marks the setting carried into the next round."
    )
    (RESULTS_DIR / "tuning.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=[*BASELINE, "all"], default="all")
    args = parser.parse_args()

    device = pick_device("auto")
    vocab, train_data, val_data = load_splits()
    evals = eval_sets(train_data, val_data)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "tuning.json"

    names = list(BASELINE) if args.model == "all" else [args.model]
    for name in names:
        result = search(name, len(vocab), train_data, evals, device)
        results = json.loads(path.read_text()) if path.exists() else {}
        results[name] = result
        path.write_text(json.dumps(results, indent=2))
        write_table(results)
        print(f"\n{name} best: {result['best']}  val {result['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
