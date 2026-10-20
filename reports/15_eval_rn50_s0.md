# Test evaluation - `single_resnet50_s0`

- Model: `resnet50`
- Split column: `split`, split: `test`
- Augmentation: True, balancing: `sampler`, layer4 unfrozen: True

## Headline metrics (no TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.983 [0.970, 0.993] |
| macro_f1 | 0.984 [0.972, 0.993] |
| accuracy | 0.984 [0.973, 0.993] |
| weighted_f1 | 0.984 [0.972, 0.993] |
| macro_auc | 1.000 [1.000, 1.000] |

## Headline metrics (with TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.988 [0.976, 0.997] |
| macro_f1 | 0.989 [0.979, 0.997] |
| accuracy | 0.989 [0.980, 0.996] |
| weighted_f1 | 0.989 [0.980, 0.996] |
| macro_auc | 1.000 [1.000, 1.000] |

## Per-class performance (no TTA)

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 0.973 | 0.991 | 0.982 | 108 |
| Blue_Finger | 0.965 | 0.965 | 0.965 | 85 |
| Clubbing | 0.982 | 0.973 | 0.978 | 113 |
| Healthy_Nail | 1.000 | 0.979 | 0.989 | 48 |
| Onychogryphosis | 1.000 | 0.990 | 0.995 | 98 |
| Pitting | 0.989 | 1.000 | 0.995 | 94 |

Any class with support in single digits cannot carry a claim. State the support next to every per-class number in the paper.

## Calibration

- Temperature fitted on validation: **T = 0.582**
- ECE before scaling: **0.0843**
- ECE after scaling: **0.0118**

The draft treats softmax outputs as confidence without ever checking whether they are calibrated. If T departs materially from 1.0, the raw confidences were mis-scaled and the 0.7 threshold was arbitrary.

## Risk-coverage (selective prediction)

- AURC (area under risk-coverage curve, lower is better): **0.0003**

| confidence threshold | coverage | selective accuracy |
| --- | --- | --- |
| 0.50 | 100.0% | 98.35% |
| 0.70 | 99.6% | 98.53% |
| 0.90 | 98.5% | 99.26% |

Reporting accuracy after a confidence filter without stating coverage overstates performance. The pair must always be quoted together.

## Confusion matrix (rows = true, cols = predicted)

| | Acral_Lentig | Blue_Finger | Clubbing | Healthy_Nail | Onychogrypho | Pitting |
| --- | --- | --- | --- | --- | --- | --- |
| **Acral_Lentiginou** | 107 | 1 | 0 | 0 | 0 | 0 |
| **Blue_Finger** | 2 | 82 | 1 | 0 | 0 | 0 |
| **Clubbing** | 1 | 2 | 110 | 0 | 0 | 0 |
| **Healthy_Nail** | 0 | 0 | 1 | 47 | 0 | 0 |
| **Onychogryphosis** | 0 | 0 | 0 | 0 | 97 | 1 |
| **Pitting** | 0 | 0 | 0 | 0 | 0 | 94 |
