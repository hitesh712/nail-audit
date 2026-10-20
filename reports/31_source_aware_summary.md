# Final model - test set summary

`efficientnet_b0`, gurav-only (6 classes, 3,835 images), group-aware split, 3 seeds, test n=191.

The test partition was read once per seed, after all model selection.

| metric | mean | sd |
| --- | --- | --- |
| balanced_accuracy | **0.9380** | 0.0191 |
| macro_f1 | **0.9388** | 0.0180 |
| accuracy | **0.9372** | 0.0189 |
| weighted_f1 | **0.9378** | 0.0187 |
| macro_auc | **0.9938** | 0.0014 |

## Calibration

- Temperature (fitted on validation): 0.643 ± 0.081
- ECE before scaling: 0.0854
- ECE after scaling: 0.0330

T below 1.0 means the network was *under*-confident, which is the expected consequence of the 0.1 label smoothing used during training. Temperature scaling reverses it and cuts calibration error by roughly a factor of four.

## Per class (mean over seeds)

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 0.963 | 0.928 | 0.945 | 37 |
| Blue_Finger | 0.846 | 0.922 | 0.882 | 30 |
| Clubbing | 0.905 | 0.924 | 0.912 | 35 |
| Healthy_Nail | 1.000 | 0.960 | 0.980 | 25 |
| Onychogryphosis | 0.962 | 0.980 | 0.970 | 33 |
| Pitting | 0.977 | 0.914 | 0.944 | 31 |

Smallest per-class support: 25. Per-class figures should be read with that in mind.

## Against the previously reported figure

| | value | measured on |
| --- | --- | --- |
| Prior report | 97.67% accuracy | 300 inherited validation images, 19.3% duplicate-contaminated, 53% of the set one class |
| This run | 93.72% accuracy, 93.80% balanced | 191 held-out items, 0% contamination |

This run lands **3.95 percentage points below** the previously reported figure.