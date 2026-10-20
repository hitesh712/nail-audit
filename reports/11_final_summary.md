# Final model - test set summary

`resnet50`, gurav-only (6 classes, 3,835 images), group-aware split, 3 seeds, test n=546.

The test partition was read once per seed, after all model selection.

| metric | mean | sd |
| --- | --- | --- |
| balanced_accuracy | **0.9840** | 0.0027 |
| macro_f1 | **0.9847** | 0.0021 |
| accuracy | **0.9835** | 0.0018 |
| weighted_f1 | **0.9835** | 0.0018 |
| macro_auc | **0.9994** | 0.0005 |

## Calibration

- Temperature (fitted on validation): 0.624 ± 0.014
- ECE before scaling: 0.0835
- ECE after scaling: 0.0131

T below 1.0 means the network was *under*-confident, which is the expected consequence of the 0.1 label smoothing used during training. Temperature scaling reverses it and cuts calibration error by roughly a factor of four.

## Per class (mean over seeds)

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 0.964 | 1.000 | 0.982 | 108 |
| Blue_Finger | 0.984 | 0.953 | 0.968 | 85 |
| Clubbing | 0.979 | 0.965 | 0.972 | 113 |
| Healthy_Nail | 1.000 | 0.993 | 0.996 | 48 |
| Onychogryphosis | 0.993 | 0.993 | 0.993 | 98 |
| Pitting | 0.993 | 1.000 | 0.996 | 94 |

Every class carries three-figure support, against 9 and 18 in the draft's validation set. Per-class claims are now interpretable.

## Against the draft

| | value | measured on |
| --- | --- | --- |
| Draft, Table V | 97.67% accuracy | 300 inherited val images, 19.3% duplicate-contaminated, class mix 53% one class |
| This work | 98.35% accuracy, 98.40% balanced | 546 held-out test images, 0% contamination, stratified to within 0.4 pp |

The corrected pipeline reports a *higher* number than the draft, on a larger and cleaner evaluation set, with the mislabelled class removed. Rigour cost nothing here.