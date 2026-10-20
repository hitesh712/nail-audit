# Test evaluation - `single_efficientnet_b0_s2`

- Model: `efficientnet_b0`
- Split column: `split`, split: `test`
- Augmentation: True, balancing: `sampler`, layer4 unfrozen: True

## Headline metrics (no TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.959 [0.926, 0.984] |
| macro_f1 | 0.959 [0.927, 0.983] |
| accuracy | 0.958 [0.927, 0.984] |
| weighted_f1 | 0.959 [0.928, 0.984] |
| macro_auc | 0.992 [0.982, 0.999] |

## Headline metrics (with TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.963 [0.932, 0.987] |
| macro_f1 | 0.964 [0.932, 0.987] |
| accuracy | 0.963 [0.932, 0.984] |
| weighted_f1 | 0.964 [0.934, 0.985] |
| macro_auc | 0.996 [0.991, 1.000] |

## Per-class performance (no TTA)

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 0.972 | 0.946 | 0.959 | 37 |
| Blue_Finger | 0.853 | 0.967 | 0.906 | 30 |
| Clubbing | 0.971 | 0.943 | 0.957 | 35 |
| Healthy_Nail | 1.000 | 0.960 | 0.980 | 25 |
| Onychogryphosis | 1.000 | 0.970 | 0.985 | 33 |
| Pitting | 0.968 | 0.968 | 0.968 | 31 |

Any class with support in single digits cannot carry a claim. State the support next to every per-class number in the paper.

## Calibration

- Temperature fitted on validation: **T = 0.696**
- ECE before scaling: **0.0935**
- ECE after scaling: **0.0210**

The draft treats softmax outputs as confidence without ever checking whether they are calibrated. If T departs materially from 1.0, the raw confidences were mis-scaled and the 0.7 threshold was arbitrary.

## Risk-coverage (selective prediction)

- AURC (area under risk-coverage curve, lower is better): **0.0087**

| confidence threshold | coverage | selective accuracy |
| --- | --- | --- |
| 0.50 | 97.9% | 96.79% |
| 0.70 | 95.3% | 97.25% |
| 0.90 | 85.9% | 98.78% |

Reporting accuracy after a confidence filter without stating coverage overstates performance. The pair must always be quoted together.

## Confusion matrix (rows = true, cols = predicted)

| | Acral_Lentig | Blue_Finger | Clubbing | Healthy_Nail | Onychogrypho | Pitting |
| --- | --- | --- | --- | --- | --- | --- |
| **Acral_Lentiginou** | 35 | 2 | 0 | 0 | 0 | 0 |
| **Blue_Finger** | 0 | 29 | 1 | 0 | 0 | 0 |
| **Clubbing** | 0 | 2 | 33 | 0 | 0 | 0 |
| **Healthy_Nail** | 0 | 1 | 0 | 24 | 0 | 0 |
| **Onychogryphosis** | 0 | 0 | 0 | 0 | 32 | 1 |
| **Pitting** | 1 | 0 | 0 | 0 | 0 | 30 |
