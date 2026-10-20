# Manuscript tables (clean test subset)

EfficientNet-B0, 3 seeds, n = 539 (7 rotation-contaminated images excluded).

## Table 2 - Held-out test performance

| Metric | Mean | SD |
|---|---|---|
| Balanced accuracy | 0.9880 | 0.0042 |
| Macro-F1 | 0.9888 | 0.0044 |
| Accuracy | 0.9889 | 0.0049 |
| Macro AUC | 0.9999 | 0.0001 |

## Table 3 - Per-class performance, mean over seeds

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Acral lentiginous melanoma | 0.997 | 0.988 | 0.992 | 107 |
| Blue finger | 0.969 | 0.976 | 0.973 | 85 |
| Clubbing | 0.977 | 0.985 | 0.981 | 111 |
| Healthy nail | 1.000 | 0.979 | 0.989 | 47 |
| Onychogryphosis | 0.997 | 1.000 | 0.998 | 97 |
| Pitting | 1.000 | 1.000 | 1.000 | 92 |
| **Total** | | | | **539** |

Support total 539 matches the reported n = 539.

## Table 5 - Coverage and selective accuracy (mean over seeds)

| tau | Coverage | Selective accuracy |
|---|---|---|
| 0.50 | 99.8% | 98.95% |
| 0.70 | 98.5% | 99.62% |
| 0.90 | 96.8% | 99.87% |