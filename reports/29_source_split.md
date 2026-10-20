# Source-aware partition

- Images: **3835** from **1337** source photographs
- Source photographs straddling a partition boundary: **0**
- Test images whose source appears in training: **0** (0.0%)

## Source photographs per partition

| class                      | test | train | val |
|----------------------------|------|-------|-----|
| Acral_Lentiginous_Melanoma | 37   | 179   | 35  |
| Blue_Finger                | 30   | 145   | 29  |
| Clubbing                   | 35   | 186   | 40  |
| Healthy_Nail               | 25   | 123   | 24  |
| Onychogryphosis            | 33   | 165   | 32  |
| Pitting                    | 31   | 157   | 31  |

## Images per partition (training uses every copy)

| class                      | test | train | val |
|----------------------------|------|-------|-----|
| Acral_Lentiginous_Melanoma | 111  | 537   | 105 |
| Blue_Finger                | 90   | 435   | 87  |
| Clubbing                   | 105  | 558   | 120 |
| Healthy_Nail               | 50   | 246   | 47  |
| Onychogryphosis            | 99   | 495   | 95  |
| Pitting                    | 92   | 470   | 93  |

## Evaluation protocol

- Validation and test scored on **one representative copy per source photograph** (382 images total), selected as the copy with the least rotation fill.
- Scoring every copy would count the same photograph up to three times and report an independent sample size that does not exist.

## Comparison with the previous partition

| | previous (image-level) | this (source-level) |
| --- | --- | --- |
| Test images sharing a source with training | 95.8% | 0.0% |
| Independent test photographs | ~23 | 191 |