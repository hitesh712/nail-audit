# EfficientNet-B0 vs ResNet-18 (draft)

Both trained on the identical split with a matched 40-epoch budget and
evaluated on the identical test partition. Comparison is paired per image.

## Per seed

| seed | EfficientNet-B0 acc | ResNet-18 (draft) acc | Δ | A✓B✗ | A✗B✓ | McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 92.15% | 92.67% | -0.52 | 8 | 9 | 1.0000 |
| 1 | 93.19% | 90.58% | +2.62 | 12 | 7 | 0.3593 |

## Pooled over seeds

- EfficientNet-B0: **92.67%**
- ResNet-18 (draft): **91.62%**
- Difference: **+1.05 pp**, bootstrap 95% CI [-2.09, +4.19]
- Discordant pairs: EfficientNet-B0 correct only **20**, ResNet-18 (draft) correct only **16**
- McNemar exact p = **0.6177**

**The difference is not statistically significant.** The two architectures cannot be separated on this test set, and the paper should say so rather than presenting EfficientNet-B0 as superior. Either would be a defensible choice; report the comparison as a tie and pick on secondary grounds such as parameter count or latency.

Note that seeds are not independent replicates of the test set - the same
191 items recur - so the pooled McNemar count is optimistic about
sample size. The per-seed rows are the conservative reading.