"""Check whether the LSTM-vs-transformer ranking depends on the shared learning rate.

    python sweep.py

Trains both neural models at each learning rate under the same budget as
train.py and writes results/lr_sweep.md. Checkpoints are not touched.
"""

from data import load_splits
from train import RESULTS_DIR, STEPS, eval_sets, pick_device, train_one

LRS = [1e-3, 2e-3, 4e-3]


def main():
    device = pick_device("auto")
    vocab, train_data, val_data = load_splits()
    evals = eval_sets(train_data, val_data)

    rows = [
        "| Peak learning rate | LSTM val loss | Transformer val loss |",
        "| ---: | ---: | ---: |",
    ]
    for lr in LRS:
        lstm, transformer = (
            train_one(name, len(vocab), train_data, evals, STEPS, device, lr=lr, save=False)
            for name in ("lstm", "transformer")
        )
        rows.append(f"| {lr:g} | {lstm['val_loss']:.4f} | {transformer['val_loss']:.4f} |")

    note = f"\n{STEPS:,} steps per run, otherwise identical to train.py.\n"
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "lr_sweep.md").write_text("\n".join(rows) + "\n" + note)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
