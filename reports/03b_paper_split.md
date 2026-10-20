# The draft's split, audited

## Reproduction of Table II

| class | ours (train/val) | published (train/val) | match |
| --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 735 / 18 | 735 / 18 | YES |
| Blue_Finger | 603 / 9 | 603 / 9 | YES |
| Clubbing | 767 / 16 | 767 / 16 | YES |
| Healthy_Nail | 571 / 82 | 571 / 82 | YES |
| Onychogryphosis | 1252 / 159 | 1252 / 159 | YES |
| Pitting | 639 / 16 | 639 / 16 | YES |
| **Total** | **4567 / 300** | **4567 / 300** | **YES** |

Exact reproduction: **YES**.

The draft did not construct a split. Every number in Table II is the
folder structure the two Kaggle datasets shipped with, concatenated.
That is why the per-class validation share ranges from 1.5% to 12.6%
while the text states twice that stratified sampling was used.

## Duplicate contamination of the reported validation set

- Validation images: **300**
- Sharing a duplicate group with a training image: **58**
- **Contamination rate: 19.3%**

Those images are near-duplicates of training data. Performance measured
on them is partly recall of memorised examples, not generalisation.

### Per class

| class                      | val_n | contaminated | rate_pct |
|----------------------------|-------|--------------|----------|
| Acral_Lentiginous_Melanoma | 18    | 0            | 0        |
| Blue_Finger                | 9     | 0            | 0        |
| Clubbing                   | 16    | 0            | 0        |
| Healthy_Nail               | 82    | 1            | 1.200    |
| Onychogryphosis            | 159   | 57           | 35.800   |
| Pitting                    | 16    | 0            | 0        |

## Label noise

- Duplicate groups spanning more than one class: **12**

Each is the same image filed under two different diagnoses.
