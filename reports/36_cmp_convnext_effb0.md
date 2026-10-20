# ConvNeXt-Tiny vs EfficientNet-B0

Both trained on the identical split with a matched 40-epoch budget and
evaluated on the identical test partition. Comparison is paired per image.

## Per seed

| seed | ConvNeXt-Tiny acc | EfficientNet-B0 acc | Δ | A✓B✗ | A✗B✓ | McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 92.15% | 92.15% | +0.00 | 6 | 6 | 1.0000 |
| 1 | 94.24% | 93.19% | +1.05 | 8 | 6 | 0.7905 |

## Pooled over seeds

- ConvNeXt-Tiny: **93.19%**
- EfficientNet-B0: **92.67%**
- Difference: **+0.52 pp**, bootstrap 95% CI [-2.09, +3.14]
- Discordant pairs: ConvNeXt-Tiny correct only **14**, EfficientNet-B0 correct only **12**
- McNemar exact p = **0.8450**

**The difference is not statistically significant.** The two architectures cannot be separated on this test set, and the paper should say so rather than presenting ConvNeXt-Tiny as superior. Either would be a defensible choice; report the comparison as a tie and pick on secondary grounds such as parameter count or latency.

Note that seeds are not independent replicates of the test set - the same
546 images recur - so the pooled McNemar count is optimistic about sample
size. The per-seed rows are the conservative reading.