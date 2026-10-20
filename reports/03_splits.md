# Splits

## Group-aware split (use this one)

## Split composition

### Images per class per split

| class                      | test | train | val |
|----------------------------|------|-------|-----|
| Acral_Lentiginous_Melanoma | 107  | 539   | 107 |
| Blue_Finger                | 89   | 434   | 89  |
| Clubbing                   | 111  | 560   | 112 |
| Healthy_Nail               | 95   | 465   | 93  |
| Onychogryphosis            | 97   | 493   | 99  |
| Onychomycosis              | 94   | 521   | 107 |
| Pitting                    | 95   | 466   | 94  |
| Psoriasis                  | 67   | 303   | 63  |

### Split sizes

- **test**: 755 images (14.2%), 705 groups
- **train**: 3781 images (71.3%), 3523 groups
- **val**: 764 images (14.4%), 705 groups

### Per-split class share (%)

| class                      | test   | train  | val    |
|----------------------------|--------|--------|--------|
| Acral_Lentiginous_Melanoma | 14.200 | 14.300 | 14     |
| Blue_Finger                | 11.800 | 11.500 | 11.600 |
| Clubbing                   | 14.700 | 14.800 | 14.700 |
| Healthy_Nail               | 12.600 | 12.300 | 12.200 |
| Onychogryphosis            | 12.800 | 13     | 13     |
| Onychomycosis              | 12.500 | 13.800 | 14     |
| Pitting                    | 12.600 | 12.300 | 12.300 |
| Psoriasis                  | 8.900  | 8      | 8.200  |

Verified: no duplicate group straddles a split boundary.

## Naive per-image split (control condition only)

## Split composition

### Images per class per split

| class                      | test | train | val |
|----------------------------|------|-------|-----|
| Acral_Lentiginous_Melanoma | 113  | 527   | 113 |
| Blue_Finger                | 92   | 428   | 92  |
| Clubbing                   | 118  | 548   | 117 |
| Healthy_Nail               | 98   | 457   | 98  |
| Onychogryphosis            | 103  | 482   | 104 |
| Onychomycosis              | 108  | 506   | 108 |
| Pitting                    | 98   | 459   | 98  |
| Psoriasis                  | 65   | 303   | 65  |

### Split sizes

- **test**: 795 images (15.0%), 780 groups
- **train**: 3710 images (70.0%), 3516 groups
- **val**: 795 images (15.0%), 780 groups

### Per-split class share (%)

| class                      | test   | train  | val    |
|----------------------------|--------|--------|--------|
| Acral_Lentiginous_Melanoma | 14.200 | 14.200 | 14.200 |
| Blue_Finger                | 11.600 | 11.500 | 11.600 |
| Clubbing                   | 14.800 | 14.800 | 14.700 |
| Healthy_Nail               | 12.300 | 12.300 | 12.300 |
| Onychogryphosis            | 13     | 13     | 13.100 |
| Onychomycosis              | 13.600 | 13.600 | 13.600 |
| Pitting                    | 12.300 | 12.400 | 12.300 |
| Psoriasis                  | 8.200  | 8.200  | 8.200  |

## The split the draft actually used (shipped source folders)

## Split composition

### Images per class per split

| class                      | train | val |
|----------------------------|-------|-----|
| Acral_Lentiginous_Melanoma | 735   | 18  |
| Blue_Finger                | 603   | 9   |
| Clubbing                   | 767   | 16  |
| Healthy_Nail               | 571   | 82  |
| Onychogryphosis            | 677   | 12  |
| Onychomycosis              | 575   | 147 |
| Pitting                    | 639   | 16  |
| Psoriasis                  | 342   | 91  |

### Split sizes

- **train**: 4909 images (92.6%), 4634 groups
- **val**: 391 images (7.4%), 352 groups

### Per-split class share (%)

| class                      | train  | val    |
|----------------------------|--------|--------|
| Acral_Lentiginous_Melanoma | 15     | 4.600  |
| Blue_Finger                | 12.300 | 2.300  |
| Clubbing                   | 15.600 | 4.100  |
| Healthy_Nail               | 11.600 | 21     |
| Onychogryphosis            | 13.800 | 3.100  |
| Onychomycosis              | 11.700 | 37.600 |
| Pitting                    | 13     | 4.100  |
| Psoriasis                  | 7      | 23.300 |

## Contamination, all three splits

| split | held-out images | sharing a group with train | rate |
| --- | --- | --- | --- |
| shipped (the draft's), held-out = `val` | 391 | 73 | **18.7%** |
| naive random | 795 | 86 | **10.8%** |
| group-aware | see above | 0 | **0.0%** |

The shipped row is the one that bears on the reported 97.67%: it is the
split the draft inherited and measured on. The naive row is what a standard
`train_test_split(stratify=y)` over the merged pool would have produced.

## Cross-validation

- 5-fold group-stratified folds assigned over train+val (column `fold`).
- Use these for the seed/fold variance the draft currently reports none of.