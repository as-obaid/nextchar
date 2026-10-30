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
| Bigram | 4,225 | 2.463 | 2.488 | 12.04 | 9 s | Low |
| LSTM | 410,465 | 1.353 | **1.552** | **4.72** | 88 s | Medium |
| Transformer | 429,889 | 1.448 | 1.647 | 5.19 | 87 s | Medium |

Cross-entropy in nats per character on held-out text; 3,000 steps per model on an Apple M1 Pro
(MPS). Full numbers: [`results/results.md`](results/results.md). Samples:
[`results/samples.md`](results/samples.md).

- **The LSTM wins** at equal size, equal data, and equal wall-clock time, and keeps its lead at
  every learning rate tried ([sweep](#learning-rate-check)).
- **The bigram is at its ceiling.** Counting character pairs directly gives 2.484; the trained
  table reaches 2.488. More training cannot help it.
- **Both neural models use only a short window.** Their loss stops improving after about 8–16
  characters, although all 128 are available. The transformer is better with ≤ 4 characters of
  context and the LSTM is better from there on.

![Training curves](results/loss_curves.png)
![Loss by available context](results/context_curve.png)

## Quickstart

```bash
pip install -r requirements.txt
python train.py     # trains all three models (~3 min on an M1 Pro)
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

**Transformer.** Token and learned position embeddings, two pre-norm blocks of 4-head causal
self-attention and a feedforward layer (128 → 512 → 128), a final LayerNorm, and a linear output.
Each position compares its query with the keys of every earlier position and takes a weighted mix
of their values, so it can read any earlier character directly instead of through a summary.
Attention is written out by hand in `CausalSelfAttention`.

## Method

Everything that is not the architecture is held fixed:

- **Data:** [Tiny Shakespeare](https://github.com/karpathy/char-rnn) (1.1M characters, 65
  symbols), split 90/10 into contiguous blocks before windowing, so no window crosses the split.
- **Budget:** the same 3,000 batches of 64 × 128 characters, in the same order, for every model.
- **Optimiser:** AdamW, 100-step warmup, cosine decay, gradient clipping at 1.0. The LSTM and
  transformer share a peak learning rate of 2e-3. The bigram uses 1e-2, because at 2e-3 it stopped
  short of its counting optimum.
- **Size and context:** LSTM hidden size 160, so its parameter count is within 5% of the
  transformer's. Both see 128 characters in training and generation, and the LSTM state is reset
  for every window.
- **Evaluation:** fixed windows, identical for every model, with dropout off. Training time
  excludes evaluation.
- **Sampling:** same prompts, temperature 0.8, length, and seed.

### Learning-rate check

`python sweep.py` retrains both neural models at three peak learning rates:

| Peak learning rate | LSTM val loss | Transformer val loss |
| ---: | ---: | ---: |
| 1e-3 | 1.609 | 1.729 |
| 2e-3 (main runs) | 1.552 | 1.647 |
| 4e-3 | **1.519** | **1.607** |

Both models do better at 4e-3, so the shared rate is not optimal for either, but the LSTM leads by
0.09–0.12 at every rate.

## Analysis

**Which model performed best?** The LSTM: validation perplexity 4.72, against 5.19 for the
transformer and 12.04 for the bigram.

**Which trained fastest?** The bigram, by an order of magnitude, but it converges to a much worse
loss. The LSTM and the transformer cost the same per step on this hardware, so the transformer's
ability to process positions in parallel did not translate into a speed advantage at this size.

**Which generated the most realistic text?** The LSTM and the transformer are close: both produce
speaker tags, verse-length lines, punctuation, and mostly real words, without lasting meaning. The
bigram does not produce words.

**What did the transformer improve?** Over the bigram, everything: it cuts perplexity by more than
half. Over the LSTM, it makes better predictions from very short context (≤ 4 characters) and
reaches its plateau sooner, at the same cost per step. It does not beat the LSTM overall.

**Was it worth the complexity?** Not at this scale. With two layers, 430k parameters, and 25M
training characters, the LSTM's recurrence is the better inductive bias for local character
patterns. The transformer's advantages (direct access to distant positions and parallel training)
pay off with more layers, more data, and longer training.

**What did I learn about attention?** Attention makes every earlier character reachable, but it
does not force the model to use them. The context curve shows the transformer stops improving
after about 8 characters. In the samples, the `ROMEO:` and `KING RICHARD III:` prompts converge on
identical text within a few lines for every model, so the prompt has stopped mattering. Given
little data and depth, both neural models learned mostly local patterns. Reaching far back is
something attention has to be trained into.

## Reflection

The most useful decisions were about the comparison, not the models: fixing the batches, matching
parameter counts, and checking the bigram against its counting optimum made the numbers
interpretable. Without those checks the transformer could have looked better or worse for reasons
that had nothing to do with attention. The result also contradicted my expectation that the
transformer would win, and the per-position loss was what explained why. Next steps would be
several seeds per model, a longer training budget, and a deeper transformer, to find where the
ranking flips.

## Repository

| Path | Contents |
| --- | --- |
| `data.py` | Download, vocabulary, contiguous split, batching, fixed evaluation windows |
| `models.py` | `BigramLM`, `LSTMLM`, `TransformerLM`, and the shared sampler |
| `train.py` | Shared training budget; writes `checkpoints/` and `results/metrics.json` |
| `report.py` | Results table, figures, and samples |
| `sweep.py` | Learning-rate check for the two neural models |
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
