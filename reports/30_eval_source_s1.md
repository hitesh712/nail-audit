# Test evaluation - `single_efficientnet_b0_s1`

- Model: `efficientnet_b0`
- Split column: `split`, split: `test`
- Augmentation: True, balancing: `sampler`, layer4 unfrozen: True

## Headline metrics (no TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.934 [0.896, 0.966] |
| macro_f1 | 0.933 [0.893, 0.965] |
| accuracy | 0.932 [0.895, 0.963] |
| weighted_f1 | 0.932 [0.894, 0.964] |
| macro_auc | 0.995 [0.988, 0.999] |

## Headline metrics (with TTA)

| metric | value [95% CI] |
| --- | --- |
| balanced_accuracy | 0.947 [0.910, 0.977] |
| macro_f1 | 0.948 [0.911, 0.977] |
| accuracy | 0.948 [0.911, 0.979] |
| weighted_f1 | 0.948 [0.914, 0.979] |
| macro_auc | 0.996 [0.991, 1.000] |

## Per-class performance (no TTA)

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 0.944 | 0.919 | 0.932 | 37 |
| Blue_Finger | 0.853 | 0.967 | 0.906 | 30 |
| Clubbing | 0.939 | 0.886 | 0.912 | 35 |
| Healthy_Nail | 1.000 | 0.960 | 0.980 | 25 |
| Onychogryphosis | 0.917 | 1.000 | 0.957 | 33 |
| Pitting | 0.964 | 0.871 | 0.915 | 31 |

Any class with support in single digits cannot carry a claim. State the support next to every per-class number in the paper.

## Calibration

- Temperature fitted on validation: **T = 0.682**
- ECE before scaling: **0.0754**
- ECE after scaling: **0.0329**

The draft treats softmax outputs as confidence without ever checking whether they are calibrated. If T departs materially from 1.0, the raw confidences were mis-scaled and the 0.7 threshold was arbitrary.

## Risk-coverage (selective prediction)

- AURC (area under risk-coverage curve, lower is better): **0.0081**

| confidence threshold | coverage | selective accuracy |
| --- | --- | --- |
| 0.50 | 99.0% | 94.18% |
| 0.70 | 92.1% | 97.73% |
| 0.90 | 83.8% | 99.38% |

Reporting accuracy after a confidence filter without stating coverage overstates performance. The pair must always be quoted together.

## Confusion matrix (rows = true, cols = predicted)

| | Acral_Lentig | Blue_Finger | Clubbing | Healthy_Nail | Onychogrypho | Pitting |
| --- | --- | --- | --- | --- | --- | --- |
| **Acral_Lentiginou** | 34 | 1 | 1 | 0 | 1 | 0 |
| **Blue_Finger** | 1 | 29 | 0 | 0 | 0 | 0 |
| **Clubbing** | 1 | 2 | 31 | 0 | 0 | 1 |
| **Healthy_Nail** | 0 | 1 | 0 | 24 | 0 | 0 |
| **Onychogryphosis** | 0 | 0 | 0 | 0 | 33 | 0 |
| **Pitting** | 0 | 1 | 1 | 0 | 2 | 27 |
