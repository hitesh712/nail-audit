# Duplicate and near-duplicate audit

Settings: pHash Hamming <= 6, embedding cosine >= 0.98

## Deduplication report

- Images scanned: **5300**
- Distinct groups after union: **4933**
- Redundant images (image count - group count): **367** (6.9% of the pool)
- Images sitting in a group of size > 1: **623**

### Pairs contributed by each signal (pre-union)
- sha256 exact: 4
- pHash near-duplicate: 566
- embedding near-duplicate: 41

### Contamination indicators
- Groups spanning **more than one source dataset**: **2**
- Groups spanning **more than one class label**: **16**

Cross-label groups are label noise: the same image filed under two different diagnoses. Examples:

- group 49: ['Acral_Lentiginous_Melanoma', 'Onychomycosis'] (3 images)
- group 146: ['Acral_Lentiginous_Melanoma', 'Healthy_Nail'] (2 images)
- group 397: ['Acral_Lentiginous_Melanoma', 'Pitting'] (2 images)
- group 782: ['Blue_Finger', 'Healthy_Nail'] (2 images)
- group 900: ['Blue_Finger', 'Onychogryphosis'] (2 images)
- group 1209: ['Blue_Finger', 'Healthy_Nail'] (2 images)
- group 1325: ['Blue_Finger', 'Healthy_Nail'] (3 images)
- group 1799: ['Clubbing', 'Pitting'] (2 images)
- group 1893: ['Clubbing', 'Pitting'] (2 images)
- group 1903: ['Clubbing', 'Pitting'] (2 images)

### Group size distribution

- size 1: 4677 groups
- size 2: 209 groups
- size 3: 21 groups
- size 4: 13 groups
- size 5: 1 groups
- size 6: 6 groups
- size 7: 3 groups
- size 8: 2 groups
- size 12: 1 groups

## Threshold sensitivity

Argue the operating point from this table rather than asserting it.
Look for the elbow: below it, unrelated images collapse together and the
group count falls off a cliff; above it, genuine re-encodes stop being
caught. Over-merging is the safer error - it only makes the split more
conservative - but gross over-merging destroys the dataset.

| signal    | threshold | n_pairs | n_groups | pct_redundant |
|-----------|-----------|---------|----------|---------------|
| phash     | 2         | 380     | 5023     | 5.226         |
| phash     | 4         | 457     | 4984     | 5.962         |
| phash     | 5         | 457     | 4984     | 5.962         |
| phash     | 6         | 566     | 4933     | 6.925         |
| phash     | 8         | 815     | 4800     | 9.434         |
| embedding | 0.930     | 200     | 5137     | 3.075         |
| embedding | 0.950     | 110     | 5193     | 2.019         |
| embedding | 0.970     | 64      | 5236     | 1.208         |
| embedding | 0.980     | 41      | 5259     | 0.774         |
| embedding | 0.990     | 17      | 5283     | 0.321         |
| embedding | 0.995     | 8       | 5292     | 0.151         |

## What this means for the split

`group_id` is the indivisible unit for train/val/test assignment. A random
per-image split places near-duplicates of the same photograph on both sides
of the boundary, so the test set is partially memorised rather than held out.
This is the same failure mode as splitting mammograms without respecting
patient identity.

- Per-image pool: **5300** images
- Group-collapsed pool: **4933** images (one representative per group)

Report both numbers in the paper's dataset section.