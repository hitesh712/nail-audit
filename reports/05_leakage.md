# Leakage experiment

Model `resnet18_paper`, seeds [0, 1, 2], max 20 epochs.

## Condition A - the draft's own split, partitioned by contamination

One model, one held-out set. The only difference between the two rows is
whether a near-duplicate of the image was in the training data.

| subset | n | accuracy |
| --- | --- | --- |
| overall | 300 | 97.22% ± 0.69 |
| contaminated | 58 | 98.85% ± 1.99 |
| clean | 242 | 96.83% ± 0.63 |

**Memorisation gap: +2.02 percentage points.**

The contaminated images are not harder or easier by any clinical
measure - they were drawn from the same distribution. The gap is the
model recognising pictures it has already seen.

## Condition B - group-aware split, held-out test set

| seed | n | accuracy [95% CI] | balanced accuracy [95% CI] | macro-F1 |
| --- | --- | --- | --- | --- |
| 0 | 693 | 0.962 [0.948, 0.977] | 0.957 [0.941, 0.973] | 0.958 |
| 1 | 685 | 0.965 [0.952, 0.978] | 0.965 [0.951, 0.978] | 0.963 |
| 2 | 694 | 0.955 [0.939, 0.970] | 0.955 [0.938, 0.971] | 0.953 |

## Against the published claim

| | accuracy |
| --- | --- |
| Published (draft, Table V) | 97.67% |
| Reproduced on the draft's own split | 97.22% |
| Same model, uncontaminated subset only | 96.83% |
| Group-aware split, held-out test | 96.09% |