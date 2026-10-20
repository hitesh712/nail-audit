# Final dataset

Source: **gurav** only. Dropped 1465 images from other
sources and unmapped folders.

- Images: **3835**
- Duplicate groups (orientation-aware): **3790** (45 redundant, 1.2%)
- Classes: **6**
- Licence: CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike)

## Class distribution

| class                      | images | groups |
|----------------------------|--------|--------|
| Acral_Lentiginous_Melanoma | 753    | 751    |
| Blue_Finger                | 612    | 608    |
| Clubbing                   | 783    | 778    |
| Healthy_Nail               | 343    | 342    |
| Onychogryphosis            | 689    | 687    |
| Pitting                    | 655    | 651    |

## Split composition

### Images per class per split

| class                      | test | train | val |
|----------------------------|------|-------|-----|
| Acral_Lentiginous_Melanoma | 108  | 536   | 109 |
| Blue_Finger                | 85   | 439   | 88  |
| Clubbing                   | 113  | 557   | 113 |
| Healthy_Nail               | 48   | 245   | 50  |
| Onychogryphosis            | 98   | 493   | 98  |
| Pitting                    | 94   | 465   | 96  |

### Split sizes

- **test**: 546 images (14.2%), 542 groups
- **train**: 2735 images (71.3%), 2706 groups
- **val**: 554 images (14.4%), 542 groups

### Per-split class share (%)

| class                      | test   | train  | val    |
|----------------------------|--------|--------|--------|
| Acral_Lentiginous_Melanoma | 19.800 | 19.600 | 19.700 |
| Blue_Finger                | 15.600 | 16.100 | 15.900 |
| Clubbing                   | 20.700 | 20.400 | 20.400 |
| Healthy_Nail               | 8.800  | 9      | 9      |
| Onychogryphosis            | 17.900 | 18     | 17.700 |
| Pitting                    | 17.200 | 17     | 17.300 |

## Verification

- Groups straddling a split boundary: **0** (asserted by `make_splits`)
- Test images sharing a duplicate group with training: **0 / 546 (0.0%)**

## What dropping rasanjana removed

| problem | before | after |
| --- | --- | --- |
| Onychomycosis filed as Onychogryphosis | 722 images | none |
| Psoriasis silently dropped | 433 images | not applicable |
| Source confound (metadata-only accuracy) | 65% vs 53% baseline | none |
| Licence | one CC BY-NC-SA, one Unknown | single CC BY-NC-SA |
| Duplication | 6.9% pooled, 24.2% in rasanjana | 1.2% |

The cost is 1,032 images and a smaller Healthy_Nail class (343 rather than
653). Measured accuracy rose rather than fell.