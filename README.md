# NextChar

Three ways to predict the next character of Shakespeare: a **bigram** table, an **LSTM**,
and a small **transformer**, trained under one controlled budget so the comparison is fair.

[![ci](https://github.com/as-obaid/nextchar/actions/workflows/ci.yml/badge.svg)](https://github.com/as-obaid/nextchar/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[Live demo](https://as-obaid.github.io/nextchar/):** type any text and watch all three models
predict the next character side by side, sample continuations, and inspect what the transformer
attends to. Everything runs in your browser.

[![NextChar demo](docs/demo.png)](https://as-obaid.github.io/nextchar/)

## Results

| Model | Parameters | Train loss | Val loss | Val perplexity | Training time | Sample quality |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bigram | 4,225 | 2.457 | 2.483 | 11.98 | 23 s | Low |
| LSTM | 410,465 | 1.172 | **1.453** | **4.28** | 220 s | Medium |
| Transformer | 399,233 | 1.252 | 1.469 | 4.34 | 458 s | Medium |

Cross-entropy in nats per character on held-out text; 7,500 steps per model on an Apple M1 Pro
(MPS), keeping each model's checkpoint with the lowest validation loss. Both neural models use
the winners of the same tuning search ([Tuning](#tuning)). Full numbers:
[`results/results.md`](results/results.md). Samples: [`results/samples.md`](results/samples.md).

- **The LSTM still wins, narrowly.** After the same tuning and a longer run, its lead is 0.015
  nats (0.094 at the original shared settings), and it trains in half the time.
- **Tuning helped the transformer most.** A higher learning rate, AdamW with β2 0.99 and weight
  decay, and four narrow layers instead of two wide ones cut its loss by 0.115. The same search
  gained the LSTM 0.067, and the deeper shape made the LSTM worse.
- **The bigram is at its ceiling.** The trained table (2.483) matches a bigram built by counting
  character pairs (2.484). More training cannot help it.
- **The last 8 characters do most of the work.** Both neural models get most of their gain from
  them, then improve slowly out to about 32–64. The transformer is better with ≤ 6 characters of
  context and the LSTM from 8 on.

![Training curves](results/loss_curves.png)
![Loss by available context](results/context_curve.png)

## Quickstart

```bash
pip install -r requirements.txt
python train.py     # trains all three models (~12 min on an M1 Pro)
python report.py    # results table, figures, and samples in results/
pytest              # model and data checks
```

Trained weights are in `checkpoints/`, so `report.py` and the demo work without retraining. On
Google Colab: `!git clone https://github.com/as-obaid/nextchar`, `%cd nextchar`, then the same
commands with a leading `!`.

To run the demo locally, serve `docs/` over HTTP. After retraining, re-export its weights and check
that the browser models still match PyTorch (needs Node):

```bash
python -m http.server -d docs 8000   # http://localhost:8000
python export_web.py                 # checkpoints/ -> docs/assets/
python scripts/web_parity.py         # JS vs PyTorch logits, tolerance 1e-4
```

## Models

All three map a window of characters to a next-character distribution at every position, and
share one training loop, loss, and sampler ([`models.py`](models.py)).

**Bigram.** A 65 × 65 table: the row for the current character is the distribution over the next
one. It uses exactly one character and ignores everything before it, so it learns pair statistics
(`q` → `u`, a capital after a newline) but has no notion of words, speakers, or lines. Its samples
are pronounceable nonsense, and two prompts that end in the same character produce the same
continuation.

**LSTM.** An embedding, two LSTM layers (hidden size 160), and a linear output. It reads left to
right, folding every earlier character into a fixed-size state through input, forget, and output
gates, which is enough to spell words, close lines, and write speaker tags. Everything it
remembers must pass through that state one step at a time, so older information is gradually
overwritten and gradients through long spans fade.

**Transformer.** Token and learned position embeddings, four pre-norm blocks, a final LayerNorm,
and a linear output. Each block applies 4-head causal self-attention (width 88, 22 dimensions per
head) and then a feedforward layer (88 → 352 → 88), each behind a LayerNorm and inside a residual
connection. Each position compares its query with the keys of every earlier position and takes a
weighted mix of their values, so it can read any earlier character directly instead of through a
summary. Attention is written out by hand in `CausalSelfAttention`.

## Method

Everything that is not the architecture is held fixed or tuned the same way:

- **Data:** [Tiny Shakespeare](https://github.com/karpathy/char-rnn) (1.1M characters, 65
  symbols), included as `data/input.txt` and re-downloaded only if missing. It is split 90/10
  into contiguous blocks before windowing, so no window crosses the split.
- **Budget:** the same 7,500 batches of 64 × 128 characters, in the same order, for every model.
- **Optimiser:** AdamW with weight decay on weight matrices only, 100-step warmup, cosine decay
  to 10%, gradient clipping at 1.0. Both neural models use peak learning rate 8e-3, β2 0.99, and
  weight decay 0.1, chosen by the search below. The bigram keeps AdamW defaults at 1e-2, which
  reaches its counting optimum.
- **Size and context:** LSTM hidden size 160, so its parameter count is within 3% of the
  transformer's. Both see 128 characters in training and generation, and the LSTM state is reset
  for every window.
- **Evaluation:** fixed windows, identical for every model, with dropout off, every 250 steps.
  Each model keeps the checkpoint with the lowest validation loss (early stopping; both neural
  models bottomed out at step 6,750). Training time excludes evaluation.
- **Sampling:** same prompts, temperature 0.8, length, and seed.

### Tuning

`python tune.py` gives both neural models the same coordinate search: three rounds, each tried
on top of the best settings so far, 3,000 steps per run. Validation loss, with kept changes in
bold:

| Round | Setting | LSTM | Transformer |
| --- | --- | ---: | ---: |
| Start | lr 2e-3, β2 0.999, weight decay 0.01, 2 layers | 1.552 | 1.646 |
| Learning rate | 4e-3 | 1.519 | 1.606 |
| | 8e-3 | **1.490** | **1.594** |
| Optimiser | β2 0.99, weight decay 0.1 | **1.485** | **1.565** |
| Shape | deeper, same size: LSTM 3 × 128, transformer 4 × 88 | 1.551 | **1.531** |

Depth is the change the two models disagree on. Two attention layers is the minimum for one layer
to build on what another found, and four give the transformer room to do it. For the LSTM, a third
layer costs width in the recurrent state that carries its memory (160 → 128). Settings were
chosen on the validation split with one seed, so differences under ~0.01 should not be
over-read. Per-model tables: [`results/tuning.md`](results/tuning.md).

## Analysis

**Which model performed best?** The LSTM, narrowly: validation perplexity 4.28, against 4.34 for
the transformer and 11.98 for the bigram.

**Which trained fastest?** The bigram, by an order of magnitude, but it converges to a much worse
loss. Of the neural models, the LSTM: 220 s against 458 s for the same 7,500 steps. The
four-layer transformer takes twice as long per step as the two-layer one did, despite having
fewer parameters. At this size, the cost of each layer, not the arithmetic, sets the speed, so
processing positions in parallel does not pay off on this hardware.

**Which generated the most realistic text?** The LSTM and the transformer are close: both write
speaker tags (`CAPULET:`, `BUCKINGHAM:`), verse-length lines, punctuation, and mostly real words,
without lasting meaning. The bigram does not produce words.

**What did the transformer improve?** Over the bigram, everything: it cuts perplexity by almost
two-thirds. Over the LSTM, it makes better predictions from very short context (≤ 6 characters),
overfits less (a train-to-validation gap of 0.22 against 0.28), and gained the most from tuning.
It does not beat the LSTM overall.

**Did the models overfit?** Mildly, and late. Validation loss bottomed out at step 6,750 and rose
by about 0.005 afterwards, so early stopping kept the step-6,750 weights. Most of the gap between
train and validation loss is not memorisation: it is already 0.16 at step 500, because the
validation split is the end of the file, with different plays from the training text.

**Was it worth the complexity?** Not at this scale. The transformer needed more tuning (a deeper
shape, a higher learning rate, weight decay) to come within 0.015 of the LSTM, and twice the
training time. The trend favours it, though: tuning and longer training closed most of the gap,
and it overfits less, so more data, depth, or training is where it should overtake.

**What did I learn about attention?** Attention makes every earlier character reachable, but using
that reach has to be learned. The context curve shows the transformer gets most of its gain from
the last 8 characters, like the LSTM. In the samples, the tuned models' `ROMEO:` and
`KING RICHARD III:` continuations split early and never rejoin, where the untuned models'
converged on identical text within a few lines, so their predictions now depend on more than the
last few characters.

## Reflection

The most useful decisions were about the comparison, not the models: fixing the batches, matching
parameter counts, tuning both models with the same search, and checking the bigram against its
counting optimum made the numbers interpretable. Tuning mattered more than I expected. The same
search moved the transformer almost twice as far as the LSTM, and the first comparison, at shared
default settings and 3,000 steps, put the LSTM's lead at six times what it is now. The
per-position loss and the samples explained what the totals could not. Next steps would be
several seeds per model for error bars, a separate test split (settings were chosen on
validation), and a larger budget to find where the ranking flips.

## Repository

| Path | Contents |
| --- | --- |
| `data/input.txt` | Tiny Shakespeare, 1.1 MB, public domain |
| `data.py` | Vocabulary, contiguous split, batching, fixed evaluation windows |
| `models.py` | `BigramLM`, `LSTMLM`, `TransformerLM`, and the shared sampler |
| `train.py` | Shared budget and per-model settings; writes `checkpoints/` and `results/metrics.json` |
| `tune.py` | The same tuning search for both neural models; writes `results/tuning.md` |
| `report.py` | Results table, figures, and samples |
| `export_web.py` | Exports the checkpoints to `docs/assets/` for the demo |
| `docs/` | The browser demo: plain-JS ports of the three models, no build step |
| `scripts/` | Parity check between the JS models and PyTorch |
| `tests/` | Shape, causality, and data-pipeline checks |

## References

- D. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed. draft:
  "N-gram Language Models", "RNNs and LSTMs", "Transformers and Pretraining".
  https://web.stanford.edu/~jurafsky/slp3/
- S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 1997.
- A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017.
- A. Karpathy, [char-rnn](https://github.com/karpathy/char-rnn), source of the Tiny Shakespeare text.

## License

[MIT](LICENSE). Tiny Shakespeare is public domain.
