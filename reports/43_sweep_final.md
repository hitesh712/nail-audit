# Duplicate and near-duplicate audit

Settings: pHash Hamming <= 6, embedding cosine >= 0.98

## Deduplication report

- Images scanned: **3835**
- Distinct groups after union: **3824**
- Redundant images (image count - group count): **11** (0.3% of the pool)
- Images sitting in a group of size > 1: **21**

### Pairs contributed by each signal (pre-union)
- sha256 exact: 0
- pHash near-duplicate: 11
- embedding near-duplicate: 0

### Contamination indicators
- Groups spanning **more than one source dataset**: **0**
- Groups spanning **more than one class label**: **10**

Cross-label groups are label noise: the same image filed under two different diagnoses. Examples:

- group 397: ['Acral_Lentiginous_Melanoma', 'Pitting'] (2 images)
- group 782: ['Blue_Finger', 'Healthy_Nail'] (2 images)
- group 900: ['Blue_Finger', 'Onychogryphosis'] (2 images)
- group 1209: ['Blue_Finger', 'Healthy_Nail'] (2 images)
- group 1325: ['Blue_Finger', 'Healthy_Nail'] (3 images)
- group 1799: ['Clubbing', 'Pitting'] (2 images)
- group 1893: ['Clubbing', 'Pitting'] (2 images)
- group 1903: ['Clubbing', 'Pitting'] (2 images)
- group 2097: ['Clubbing', 'Onychogryphosis'] (2 images)
- group 2732: ['Onychogryphosis', 'Pitting'] (2 images)

### Group size distribution

- size 1: 3814 groups
- size 2: 9 groups
- size 3: 1 groups

## Threshold sensitivity

Argue the operating point from this table rather than asserting it.
Look for the elbow: below it, unrelated images collapse together and the
group count falls off a cliff; above it, genuine re-encodes stop being
caught. Over-merging is the safer error - it only makes the split more
conservative - but gross over-merging destroys the dataset.

| signal    | threshold | n_pairs | n_groups | pct_redundant |
|-----------|-----------|---------|----------|---------------|
| phash     | 2         | 0       | 3835     | 0             |
| phash     | 4         | 0       | 3835     | 0             |
| phash     | 5         | 0       | 3835     | 0             |
| phash     | 6         | 11      | 3824     | 0.287         |
| phash     | 8         | 77      | 3760     | 1.956         |
| embedding | 0.930     | 0       | 3835     | 0             |
| embedding | 0.950     | 0       | 3835     | 0             |
| embedding | 0.970     | 0       | 3835     | 0             |
| embedding | 0.980     | 0       | 3835     | 0             |
| embedding | 0.990     | 0       | 3835     | 0             |
| embedding | 0.995     | 0       | 3835     | 0             |

## What this means for the split

`group_id` is the indivisible unit for train/val/test assignment. A random
per-image split places near-duplicates of the same photograph on both sides
of the boundary, so the test set is partially memorised rather than held out.
This is the same failure mode as splitting mammograms without respecting
patient identity.

- Per-image pool: **3835** images
- Group-collapsed pool: **3824** images (one representative per group)

Report both numbers in the paper's dataset section.