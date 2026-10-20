# Test evaluation - `single_efficientnet_b0_s0`

- Model: `efficientnet_b0`
- Split column: `split`, split: `test`
- Augmentation: True, balancing: `sampler`, layer4 unfrozen: True

## Headline metrics (no TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.985 [0.972, 0.994] |
| macro_f1 | 0.986 [0.974, 0.994] |
| accuracy | 0.985 [0.974, 0.995] |
| weighted_f1 | 0.985 [0.974, 0.995] |
| macro_auc | 1.000 [1.000, 1.000] |

## Headline metrics (with TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.990 [0.979, 0.998] |
| macro_f1 | 0.991 [0.981, 0.998] |
| accuracy | 0.991 [0.982, 0.998] |
| weighted_f1 | 0.991 [0.982, 0.998] |
| macro_auc | 1.000 [1.000, 1.000] |

## Per-class performance (no TTA)

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 1.000 | 0.972 | 0.986 | 108 |
| Blue_Finger | 0.965 | 0.965 | 0.965 | 85 |
| Clubbing | 0.957 | 0.991 | 0.974 | 113 |
| Healthy_Nail | 1.000 | 0.979 | 0.989 | 48 |
| Onychogryphosis | 1.000 | 1.000 | 1.000 | 98 |
| Pitting | 1.000 | 1.000 | 1.000 | 94 |

Any class with support in single digits cannot carry a claim. State the support next to every per-class number in the paper.

## Calibration

- Temperature fitted on validation: **T = 0.548**
- ECE before scaling: **0.0895**
- ECE after scaling: **0.0132**

The draft treats softmax outputs as confidence without ever checking whether they are calibrated. If T departs materially from 1.0, the raw confidences were mis-scaled and the 0.7 threshold was arbitrary.

## Risk-coverage (selective prediction)

- AURC (area under risk-coverage curve, lower is better): **0.0002**

| confidence threshold | coverage | selective accuracy |
| --- | --- | --- |
| 0.50 | 99.5% | 98.71% |
| 0.70 | 98.4% | 99.63% |
| 0.90 | 97.1% | 99.81% |

Reporting accuracy after a confidence filter without stating coverage overstates performance. The pair must always be quoted together.

## Confusion matrix (rows = true, cols = predicted)

| | Acral_Lentig | Blue_Finger | Clubbing | Healthy_Nail | Onychogrypho | Pitting |
| --- | --- | --- | --- | --- | --- | --- |
| **Acral_Lentiginou** | 105 | 2 | 1 | 0 | 0 | 0 |
| **Blue_Finger** | 0 | 82 | 3 | 0 | 0 | 0 |
| **Clubbing** | 0 | 1 | 112 | 0 | 0 | 0 |
| **Healthy_Nail** | 0 | 0 | 1 | 47 | 0 | 0 |
| **Onychogryphosis** | 0 | 0 | 0 | 0 | 98 | 0 |
| **Pitting** | 0 | 0 | 0 | 0 | 0 | 94 |
