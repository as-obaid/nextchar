## LSTM

| Round | Setting | Parameters | Val loss | Kept |
| --- | --- | ---: | ---: | :---: |
| start | original settings | 410,465 | 1.5515 |  |
| learning rate | lr 4e-3 | 410,465 | 1.5186 |  |
| learning rate | lr 8e-3 | 410,465 | 1.4896 | ✓ |
| optimizer | β2 0.99, weight decay 0.1 | 410,465 | 1.4850 | ✓ |
| shape | 3 layers × 128 | 412,993 | 1.5510 |  |

## Transformer

| Round | Setting | Parameters | Val loss | Kept |
| --- | --- | ---: | ---: | :---: |
| start | original settings | 429,889 | 1.6459 |  |
| learning rate | lr 4e-3 | 429,889 | 1.6061 |  |
| learning rate | lr 8e-3 | 429,889 | 1.5942 | ✓ |
| optimizer | β2 0.99, weight decay 0.1 | 429,889 | 1.5646 | ✓ |
| shape | 4 layers × 88 | 399,233 | 1.5313 | ✓ |

3,000 steps per run on the same batches. Validation loss in nats per character. A ✓ marks the setting carried into the next round.
