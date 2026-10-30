| Model | Parameters | Steps | Train loss | Val loss | Val perplexity | Training time | Device |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Bigram | 4,225 | 3,000 | 2.4631 | 2.4884 | 12.04 | 8.9 s | mps |
| LSTM | 410,465 | 3,000 | 1.3534 | 1.5516 | 4.72 | 87.6 s | mps |
| Transformer | 429,889 | 3,000 | 1.4481 | 1.6465 | 5.19 | 86.5 s | mps |

Loss is mean cross-entropy in nats per character, measured with dropout off on the same fixed windows for every model. Perplexity is exp(loss). Training time excludes evaluation.

Reference: a bigram built by counting character pairs in the training set (add-0.1 smoothing) has validation loss 2.4838.
