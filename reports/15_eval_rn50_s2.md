# Test evaluation - `single_resnet50_s2`

- Model: `resnet50`
- Split column: `split`, split: `test`
- Augmentation: True, balancing: `sampler`, layer4 unfrozen: True

## Headline metrics (no TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.976 [0.962, 0.988] |
| macro_f1 | 0.977 [0.964, 0.989] |
| accuracy | 0.976 [0.963, 0.989] |
| weighted_f1 | 0.976 [0.963, 0.989] |
| macro_auc | 0.999 [0.998, 1.000] |

## Headline metrics (with TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.982 [0.969, 0.993] |
| macro_f1 | 0.984 [0.972, 0.994] |
| accuracy | 0.984 [0.973, 0.993] |
| weighted_f1 | 0.984 [0.972, 0.993] |
| macro_auc | 0.999 [0.998, 1.000] |

## Per-class performance (no TTA)

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 0.972 | 0.972 | 0.972 | 108 |
| Blue_Finger | 0.964 | 0.941 | 0.952 | 85 |
| Clubbing | 0.948 | 0.973 | 0.961 | 113 |
| Healthy_Nail | 1.000 | 0.979 | 0.989 | 48 |
| Onychogryphosis | 1.000 | 1.000 | 1.000 | 98 |
| Pitting | 0.989 | 0.989 | 0.989 | 94 |

Any class with support in single digits cannot carry a claim. State the support next to every per-class number in the paper.

## Calibration

- Temperature fitted on validation: **T = 0.620**
- ECE before scaling: **0.0890**
- ECE after scaling: **0.0096**

The draft treats softmax outputs as confidence without ever checking whether they are calibrated. If T departs materially from 1.0, the raw confidences were mis-scaled and the 0.7 threshold was arbitrary.

## Risk-coverage (selective prediction)

- AURC (area under risk-coverage curve, lower is better): **0.0015**

| confidence threshold | coverage | selective accuracy |
| --- | --- | --- |
| 0.50 | 99.6% | 97.98% |
| 0.70 | 97.1% | 98.68% |
| 0.90 | 93.4% | 99.22% |

Reporting accuracy after a confidence filter without stating coverage overstates performance. The pair must always be quoted together.

## Confusion matrix (rows = true, cols = predicted)

| | Acral_Lentig | Blue_Finger | Clubbing | Healthy_Nail | Onychogrypho | Pitting |
| --- | --- | --- | --- | --- | --- | --- |
| **Acral_Lentiginou** | 105 | 1 | 1 | 0 | 0 | 1 |
| **Blue_Finger** | 1 | 80 | 4 | 0 | 0 | 0 |
| **Clubbing** | 1 | 2 | 110 | 0 | 0 | 0 |
| **Healthy_Nail** | 1 | 0 | 0 | 47 | 0 | 0 |
| **Onychogryphosis** | 0 | 0 | 0 | 0 | 98 | 0 |
| **Pitting** | 0 | 0 | 1 | 0 | 0 | 93 |
