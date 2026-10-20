# Manuscript tables (clean test subset)

EfficientNet-B0, 3 seeds, n = 191 (0 rotation-contaminated images excluded).

## Table 2 - Held-out test performance

| Metric | Mean | SD |
|---|---|---|
| Balanced accuracy | 0.9380 | 0.0191 |
| Macro-F1 | 0.9388 | 0.0180 |
| Accuracy | 0.9372 | 0.0189 |
| Macro AUC | 0.9938 | 0.0014 |

## Table 3 - Per-class performance, mean over seeds

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Acral lentiginous melanoma | 0.963 | 0.928 | 0.945 | 37 |
| Blue finger | 0.846 | 0.922 | 0.882 | 30 |
| Clubbing | 0.905 | 0.924 | 0.912 | 35 |
| Healthy nail | 1.000 | 0.960 | 0.980 | 25 |
| Onychogryphosis | 0.962 | 0.980 | 0.970 | 33 |
| Pitting | 0.977 | 0.914 | 0.944 | 31 |
| **Total** | | | | **191** |

Support total 191 matches the reported n = 191.

## Table 5 - Coverage and selective accuracy (mean over seeds)

| tau | Coverage | Selective accuracy |
|---|---|---|
| 0.50 | 98.4% | 94.69% |
| 0.70 | 94.4% | 96.69% |
| 0.90 | 86.6% | 98.61% |