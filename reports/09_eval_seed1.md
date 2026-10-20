# Test evaluation - `single_resnet50_s1`

- Model: `resnet50`
- Split column: `split`, split: `test`
- Augmentation: True, balancing: `sampler`, layer4 unfrozen: True

## Headline metrics (no TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.984 [0.973, 0.994] |
| macro_f1 | 0.985 [0.974, 0.994] |
| accuracy | 0.984 [0.972, 0.993] |
| weighted_f1 | 0.983 [0.972, 0.993] |
| macro_auc | 0.999 [0.997, 1.000] |

## Headline metrics (with TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.986 [0.973, 0.995] |
| macro_f1 | 0.987 [0.977, 0.996] |
| accuracy | 0.987 [0.976, 0.996] |
| weighted_f1 | 0.987 [0.976, 0.996] |
| macro_auc | 1.000 [0.999, 1.000] |

## Per-class performance (no TTA)

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 0.973 | 1.000 | 0.986 | 108 |
| Blue_Finger | 0.988 | 0.941 | 0.964 | 85 |
| Clubbing | 0.965 | 0.965 | 0.965 | 113 |
| Healthy_Nail | 1.000 | 1.000 | 1.000 | 48 |
| Onychogryphosis | 0.990 | 1.000 | 0.995 | 98 |
| Pitting | 1.000 | 1.000 | 1.000 | 94 |

Any class with support in single digits cannot carry a claim. State the support next to every per-class number in the paper.

## Calibration

- Temperature fitted on validation: **T = 0.630**
- ECE before scaling: **0.0824**
- ECE after scaling: **0.0125**

The draft treats softmax outputs as confidence without ever checking whether they are calibrated. If T departs materially from 1.0, the raw confidences were mis-scaled and the 0.7 threshold was arbitrary.

## Risk-coverage (selective prediction)

- AURC (area under risk-coverage curve, lower is better): **0.0013**

| confidence threshold | coverage | selective accuracy |
| --- | --- | --- |
| 0.50 | 99.8% | 98.53% |
| 0.70 | 98.5% | 99.07% |
| 0.90 | 96.2% | 99.24% |

Reporting accuracy after a confidence filter without stating coverage overstates performance. The pair must always be quoted together.

## Confusion matrix (rows = true, cols = predicted)

| | Acral_Lentig | Blue_Finger | Clubbing | Healthy_Nail | Onychogrypho | Pitting |
| --- | --- | --- | --- | --- | --- | --- |
| **Acral_Lentiginou** | 108 | 0 | 0 | 0 | 0 | 0 |
| **Blue_Finger** | 1 | 80 | 4 | 0 | 0 | 0 |
| **Clubbing** | 2 | 1 | 109 | 0 | 1 | 0 |
| **Healthy_Nail** | 0 | 0 | 0 | 48 | 0 | 0 |
| **Onychogryphosis** | 0 | 0 | 0 | 0 | 98 | 0 |
| **Pitting** | 0 | 0 | 0 | 0 | 0 | 94 |
