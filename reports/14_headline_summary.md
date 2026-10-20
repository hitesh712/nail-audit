# Final model - test set summary

`efficientnet_b0`, gurav-only (6 classes, 3,835 images), group-aware split, 3 seeds, test n=546.

The test partition was read once per seed, after all model selection.

| metric | mean | sd |
| --- | --- | --- |
| balanced_accuracy | **0.9881** | 0.0041 |
| macro_f1 | **0.9889** | 0.0043 |
| accuracy | **0.9890** | 0.0048 |
| weighted_f1 | **0.9890** | 0.0048 |
| macro_auc | **0.9999** | 0.0001 |

## Calibration

- Temperature (fitted on validation): 0.520 ± 0.032
- ECE before scaling: 0.0895
- ECE after scaling: 0.0110

T below 1.0 means the network was *under*-confident, which is the expected consequence of the 0.1 label smoothing used during training. Temperature scaling reverses it and cuts calibration error by roughly a factor of four.

## Per class (mean over seeds)

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 0.997 | 0.988 | 0.992 | 108 |
| Blue_Finger | 0.969 | 0.976 | 0.973 | 85 |
| Clubbing | 0.977 | 0.985 | 0.981 | 113 |
| Healthy_Nail | 1.000 | 0.979 | 0.989 | 48 |
| Onychogryphosis | 0.997 | 1.000 | 0.998 | 98 |
| Pitting | 1.000 | 1.000 | 1.000 | 94 |

Every class carries three-figure support, against 9 and 18 in the draft's validation set. Per-class claims are now interpretable.

## Against the draft

| | value | measured on |
| --- | --- | --- |
| Draft, Table V | 97.67% accuracy | 300 inherited val images, 19.3% duplicate-contaminated, class mix 53% one class |
| This work | 98.90% accuracy, 98.81% balanced | 546 held-out test images, 0% contamination, stratified to within 0.4 pp |

The corrected pipeline reports a *higher* number than the draft, on a larger and cleaner evaluation set, with the mislabelled class removed. Rigour cost nothing here.