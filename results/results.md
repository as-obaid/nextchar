| Model | Parameters | Steps | Best step | Train loss | Val loss | Val perplexity | Training time | Device |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Bigram | 4,225 | 7,500 | 5,250 | 2.4569 | 2.4828 | 11.98 | 23.2 s | mps |
| LSTM | 410,465 | 7,500 | 6,750 | 1.1715 | 1.4534 | 4.28 | 220.1 s | mps |
| Transformer | 399,233 | 7,500 | 6,750 | 1.2516 | 1.4687 | 4.34 | 458.1 s | mps |

Loss is mean cross-entropy in nats per character, measured with dropout off on the same fixed windows for every model. Perplexity is exp(loss). The checkpoint kept is the evaluation with the lowest validation loss (best step). Training time is for all steps and excludes evaluation.

Reference: a bigram built by counting character pairs in the training set (add-0.1 smoothing) has validation loss 2.4838.
