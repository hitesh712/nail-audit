# Test evaluation - `single_efficientnet_b0_s1`

- Model: `efficientnet_b0`
- Split column: `split`, split: `test`
- Augmentation: True, balancing: `sampler`, layer4 unfrozen: True

## Headline metrics (no TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.987 [0.976, 0.996] |
| macro_f1 | 0.987 [0.977, 0.995] |
| accuracy | 0.987 [0.978, 0.996] |
| weighted_f1 | 0.987 [0.978, 0.996] |
| macro_auc | 1.000 [1.000, 1.000] |

## Headline metrics (with TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.989 [0.978, 0.997] |
| macro_f1 | 0.989 [0.980, 0.997] |
| accuracy | 0.989 [0.980, 0.996] |
| weighted_f1 | 0.989 [0.980, 0.996] |
| macro_auc | 1.000 [1.000, 1.000] |

## Per-class performance (no TTA)

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 0.991 | 0.991 | 0.991 | 108 |
| Blue_Finger | 0.955 | 0.988 | 0.971 | 85 |
| Clubbing | 0.991 | 0.965 | 0.978 | 113 |
| Healthy_Nail | 1.000 | 0.979 | 0.989 | 48 |
| Onychogryphosis | 0.990 | 1.000 | 0.995 | 98 |
| Pitting | 1.000 | 1.000 | 1.000 | 94 |

Any class with support in single digits cannot carry a claim. State the support next to every per-class number in the paper.

## Calibration

- Temperature fitted on validation: **T = 0.486**
- ECE before scaling: **0.0924**
- ECE after scaling: **0.0113**

The draft treats softmax outputs as confidence without ever checking whether they are calibrated. If T departs materially from 1.0, the raw confidences were mis-scaled and the 0.7 threshold was arbitrary.

## Risk-coverage (selective prediction)

- AURC (area under risk-coverage curve, lower is better): **0.0003**

| confidence threshold | coverage | selective accuracy |
| --- | --- | --- |
| 0.50 | 100.0% | 98.72% |
| 0.70 | 98.2% | 99.81% |
| 0.90 | 95.8% | 99.81% |

Reporting accuracy after a confidence filter without stating coverage overstates performance. The pair must always be quoted together.

## Confusion matrix (rows = true, cols = predicted)

| | Acral_Lentig | Blue_Finger | Clubbing | Healthy_Nail | Onychogrypho | Pitting |
| --- | --- | --- | --- | --- | --- | --- |
| **Acral_Lentiginou** | 107 | 1 | 0 | 0 | 0 | 0 |
| **Blue_Finger** | 0 | 84 | 1 | 0 | 0 | 0 |
| **Clubbing** | 1 | 2 | 109 | 0 | 1 | 0 |
| **Healthy_Nail** | 0 | 1 | 0 | 47 | 0 | 0 |
| **Onychogryphosis** | 0 | 0 | 0 | 0 | 98 | 0 |
| **Pitting** | 0 | 0 | 0 | 0 | 0 | 94 |
