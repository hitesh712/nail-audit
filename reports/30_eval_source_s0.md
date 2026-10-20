# Test evaluation - `single_efficientnet_b0_s0`

- Model: `efficientnet_b0`
- Split column: `split`, split: `test`
- Augmentation: True, balancing: `sampler`, layer4 unfrozen: True

## Headline metrics (no TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.921 [0.880, 0.958] |
| macro_f1 | 0.924 [0.883, 0.959] |
| accuracy | 0.921 [0.880, 0.958] |
| weighted_f1 | 0.923 [0.883, 0.959] |
| macro_auc | 0.994 [0.987, 0.999] |

## Headline metrics (with TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.931 [0.890, 0.965] |
| macro_f1 | 0.934 [0.895, 0.966] |
| accuracy | 0.932 [0.895, 0.963] |
| weighted_f1 | 0.933 [0.896, 0.964] |
| macro_auc | 0.996 [0.991, 0.999] |

## Per-class performance (no TTA)

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 0.971 | 0.919 | 0.944 | 37 |
| Blue_Finger | 0.833 | 0.833 | 0.833 | 30 |
| Clubbing | 0.805 | 0.943 | 0.868 | 35 |
| Healthy_Nail | 1.000 | 0.960 | 0.980 | 25 |
| Onychogryphosis | 0.970 | 0.970 | 0.970 | 33 |
| Pitting | 1.000 | 0.903 | 0.949 | 31 |

Any class with support in single digits cannot carry a claim. State the support next to every per-class number in the paper.

## Calibration

- Temperature fitted on validation: **T = 0.549**
- ECE before scaling: **0.0872**
- ECE after scaling: **0.0450**

The draft treats softmax outputs as confidence without ever checking whether they are calibrated. If T departs materially from 1.0, the raw confidences were mis-scaled and the 0.7 threshold was arbitrary.

## Risk-coverage (selective prediction)

- AURC (area under risk-coverage curve, lower is better): **0.0076**

| confidence threshold | coverage | selective accuracy |
| --- | --- | --- |
| 0.50 | 98.4% | 93.09% |
| 0.70 | 95.8% | 95.08% |
| 0.90 | 90.1% | 97.67% |

Reporting accuracy after a confidence filter without stating coverage overstates performance. The pair must always be quoted together.

## Confusion matrix (rows = true, cols = predicted)

| | Acral_Lentig | Blue_Finger | Clubbing | Healthy_Nail | Onychogrypho | Pitting |
| --- | --- | --- | --- | --- | --- | --- |
| **Acral_Lentiginou** | 34 | 2 | 0 | 0 | 1 | 0 |
| **Blue_Finger** | 0 | 25 | 5 | 0 | 0 | 0 |
| **Clubbing** | 0 | 2 | 33 | 0 | 0 | 0 |
| **Healthy_Nail** | 0 | 1 | 0 | 24 | 0 | 0 |
| **Onychogryphosis** | 0 | 0 | 1 | 0 | 32 | 0 |
| **Pitting** | 1 | 0 | 2 | 0 | 0 | 28 |
