# EfficientNet-B0 vs ResNet-18 (draft)

Both trained on the identical split with a matched 40-epoch budget and
evaluated on the identical test partition. Comparison is paired per image.

## Per seed

| seed | EfficientNet-B0 acc | ResNet-18 (draft) acc | Δ | A✓B✗ | A✗B✓ | McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 98.53% | 97.07% | +1.47 | 13 | 5 | 0.0963 |
| 1 | 98.72% | 96.70% | +2.01 | 13 | 2 | 0.0074 |

## Pooled over seeds

- EfficientNet-B0: **98.63%**
- ResNet-18 (draft): **96.89%**
- Difference: **+1.74 pp**, bootstrap 95% CI [+0.73, +2.75]
- Discordant pairs: EfficientNet-B0 correct only **26**, ResNet-18 (draft) correct only **7**
- McNemar exact p = **0.0013**

The difference is statistically significant. EfficientNet-B0 may be reported as the stronger model on this dataset.

Note that seeds are not independent replicates of the test set - the same
546 images recur - so the pooled McNemar count is optimistic about sample
size. The per-seed rows are the conservative reading.