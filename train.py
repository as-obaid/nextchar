"""Train the bigram, LSTM, and transformer models under one shared budget.

    python train.py                     # all three, one after another
    python train.py --model lstm        # just one
    python train.py --steps 500         # quick run

Writes checkpoints/<model>.pt and results/metrics.json.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import eval_windows, get_batch, load_splits
from models import MODELS

ROOT = Path(__file__).parent
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

# Shared by every model, so each one trains on the same batches, in the same
# order, for the same number of target characters.
BLOCK_SIZE = 128  # context length
BATCH_SIZE = 64
STEPS = 7500
WARMUP = 100
GRAD_CLIP = 1.0
EVAL_INTERVAL = 250
EVAL_WINDOWS = 1000  # cap on evaluation windows per split
SEED = 1337

# Per-model optimizer settings and shape. The two neural models use the winners of
# tune.py (results/tuning.md), which gave both the same search. The LSTM hidden size
# is 160 rather than 128 so its parameter count lands close to the transformer's
# (~410k vs ~399k). The bigram table is at its counting optimum already; at lr 2e-3
# it ended ~0.08 nats short of it in 3,000 steps, so it keeps a larger step.
CONFIG = {
    "bigram": dict(lr=1e-2, betas=(0.9, 0.999), weight_decay=0.01, hparams={}),
    "lstm": dict(
        lr=8e-3,
        betas=(0.9, 0.99),
        weight_decay=0.1,
        hparams=dict(n_embd=128, hidden_size=160, n_layer=2, dropout=0.1),
    ),
    "transformer": dict(
        lr=8e-3,
        betas=(0.9, 0.99),
        weight_decay=0.1,
        hparams=dict(
            n_embd=88, n_head=4, n_layer=4, ff_dim=352, block_size=BLOCK_SIZE, dropout=0.1
        ),
    ),
}


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    # GPU work runs asynchronously; wait for it before reading the clock.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def lr_scale(step, steps):
    """Linear warmup, then cosine decay to 10% of the peak learning rate."""
    if step < WARMUP:
        return (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, steps - WARMUP)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, x, y, device, batch_size=256):
    """Mean cross-entropy in nats per character over fixed windows, dropout off."""
    model.eval()
    total = 0.0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i : i + batch_size].to(device), y[i : i + batch_size].to(device)
        logits = model(xb)
        total += F.cross_entropy(logits.flatten(0, 1), yb.flatten(), reduction="sum").item()
    model.train()
    return total / y.numel()


def eval_sets(train_data, val_data):
    return {
        "train": eval_windows(train_data, BLOCK_SIZE, EVAL_WINDOWS),
        "val": eval_windows(val_data, BLOCK_SIZE, EVAL_WINDOWS),
    }


def make_optimizer(model, config):
    """AdamW with weight decay on weight matrices only, not on biases or LayerNorm gains."""
    params = list(model.parameters())
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": config["weight_decay"]},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, lr=config["lr"], betas=config["betas"])


def train_one(name, vocab_size, train_data, evals, steps, device, config=None, save=True):
    config = CONFIG[name] if config is None else config
    torch.manual_seed(SEED)  # same initialisation and dropout seed for every model
    model = MODELS[name](vocab_size, **config["hparams"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{name}: {n_params:,} parameters, {config}")

    opt = make_optimizer(model, config)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_scale(s, steps))
    batches = torch.Generator().manual_seed(SEED)  # identical batch order for every model

    history = []
    best = {"val_loss": math.inf}
    train_time = 0.0

    def log(step):
        tr = evaluate(model, *evals["train"], device)
        va = evaluate(model, *evals["val"], device)
        history.append({"step": step, "train_loss": tr, "val_loss": va, "time_s": train_time})
        print(f"  step {step:5d}  train {tr:.4f}  val {va:.4f}  {train_time:7.1f}s")
        if va < best["val_loss"]:
            # Early stopping: keep the weights from the lowest validation loss seen.
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best.update(step=step, train_loss=tr, val_loss=va, state_dict=state)

    log(0)
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, batches)
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.flatten(0, 1), yb.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()

        if step % EVAL_INTERVAL == 0 or step == steps:
            sync(device)
            train_time += time.perf_counter() - t0  # evaluation time is not counted
            log(step)
            t0 = time.perf_counter()

    if save:
        ckpt = {"hparams": config["hparams"], "state_dict": best["state_dict"]}
        torch.save(ckpt, CKPT_DIR / f"{name}.pt")
    return {
        "params": n_params,
        "train_loss": best["train_loss"],
        "val_loss": best["val_loss"],
        "best_step": best["step"],
        "train_time_s": train_time,
        "device": str(device),
        "lr": config["lr"],
        "betas": list(config["betas"]),
        "weight_decay": config["weight_decay"],
        "steps": steps,
        "target_chars": steps * BATCH_SIZE * BLOCK_SIZE,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = parser.parse_args()

    device = pick_device(args.device)
    vocab, train_data, val_data = load_splits()
    print(
        f"vocab {len(vocab)} chars | train {len(train_data):,} chars | "
        f"val {len(val_data):,} chars | device {device}"
    )
    evals = eval_sets(train_data, val_data)

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        metrics[name] = train_one(name, len(vocab), train_data, evals, args.steps, device)
        metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {metrics_path}. Next: python report.py")


if __name__ == "__main__":
    main()
