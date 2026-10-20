# EfficientNet-B0 vs ResNet-50

Both trained on the identical split with a matched 40-epoch budget and
evaluated on the identical test partition. Comparison is paired per image.

## Per seed

| seed | EfficientNet-B0 acc | ResNet-50 acc | Δ | A✓B✗ | A✗B✓ | McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 98.53% | 98.35% | +0.18 | 7 | 6 | 1.0000 |
| 1 | 98.72% | 98.53% | +0.18 | 5 | 4 | 1.0000 |
| 2 | 99.45% | 97.62% | +1.83 | 11 | 1 | 0.0063 |

## Pooled over seeds

- EfficientNet-B0: **98.90%**
- ResNet-50: **98.17%**
- Difference: **+0.73 pp**, bootstrap 95% CI [+0.06, +1.47]
- Discordant pairs: EfficientNet-B0 correct only **23**, ResNet-50 correct only **11**
- McNemar exact p = **0.0576**

**The difference is not statistically significant.** The two architectures cannot be separated on this test set, and the paper should say so rather than presenting EfficientNet-B0 as superior. Either would be a defensible choice; report the comparison as a tie and pick on secondary grounds such as parameter count or latency.

Note that seeds are not independent replicates of the test set - the same
546 images recur - so the pooled McNemar count is optimistic about sample
size. The per-seed rows are the conservative reading.