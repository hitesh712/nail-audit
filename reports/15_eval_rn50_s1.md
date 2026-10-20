# Test evaluation - `single_resnet50_s1`

- Model: `resnet50`
- Split column: `split`, split: `test`
- Augmentation: True, balancing: `sampler`, layer4 unfrozen: True

## Headline metrics (no TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.985 [0.973, 0.995] |
| macro_f1 | 0.986 [0.974, 0.995] |
| accuracy | 0.985 [0.974, 0.995] |
| weighted_f1 | 0.985 [0.974, 0.995] |
| macro_auc | 1.000 [0.999, 1.000] |

## Headline metrics (with TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.993 [0.984, 1.000] |
| macro_f1 | 0.994 [0.986, 1.000] |
| accuracy | 0.995 [0.987, 1.000] |
| weighted_f1 | 0.995 [0.987, 1.000] |
| macro_auc | 1.000 [1.000, 1.000] |

## Per-class performance (no TTA)

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 0.973 | 1.000 | 0.986 | 108 |
| Blue_Finger | 0.976 | 0.976 | 0.976 | 85 |
| Clubbing | 0.991 | 0.965 | 0.978 | 113 |
| Healthy_Nail | 1.000 | 0.979 | 0.989 | 48 |
| Onychogryphosis | 0.990 | 0.990 | 0.990 | 98 |
| Pitting | 0.989 | 1.000 | 0.995 | 94 |

Any class with support in single digits cannot carry a claim. State the support next to every per-class number in the paper.

## Calibration

- Temperature fitted on validation: **T = 0.648**
- ECE before scaling: **0.0760**
- ECE after scaling: **0.0151**

The draft treats softmax outputs as confidence without ever checking whether they are calibrated. If T departs materially from 1.0, the raw confidences were mis-scaled and the 0.7 threshold was arbitrary.

## Risk-coverage (selective prediction)

- AURC (area under risk-coverage curve, lower is better): **0.0006**

| confidence threshold | coverage | selective accuracy |
| --- | --- | --- |
| 0.50 | 100.0% | 98.53% |
| 0.70 | 98.7% | 98.70% |
| 0.90 | 97.3% | 99.44% |

Reporting accuracy after a confidence filter without stating coverage overstates performance. The pair must always be quoted together.

## Confusion matrix (rows = true, cols = predicted)

| | Acral_Lentig | Blue_Finger | Clubbing | Healthy_Nail | Onychogrypho | Pitting |
| --- | --- | --- | --- | --- | --- | --- |
| **Acral_Lentiginou** | 108 | 0 | 0 | 0 | 0 | 0 |
| **Blue_Finger** | 1 | 83 | 1 | 0 | 0 | 0 |
| **Clubbing** | 1 | 2 | 109 | 0 | 1 | 0 |
| **Healthy_Nail** | 1 | 0 | 0 | 47 | 0 | 0 |
| **Onychogryphosis** | 0 | 0 | 0 | 0 | 97 | 1 |
| **Pitting** | 0 | 0 | 0 | 0 | 0 | 94 |
