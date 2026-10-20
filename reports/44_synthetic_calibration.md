# Threshold calibration on a synthetic pool

Pool: **192 images** generated from **60 distinct originals**. Each original contributes a downscale-and-restore copy, a brightness-perturbed copy, and for every fifth original a byte-identical copy, placed across two nominal sources. The duplicate structure is therefore known rather than inferred.

`impure` counts predicted groups that span more than one original, which is over-merging. `family recall` is the share of originals whose copies all landed in one predicted group, which is what prevents leakage. Each signal is evaluated alone, not unioned.

| signal | threshold | pairs | groups | impure groups | % impure | family recall % |
|---|---|---|---|---|---|---|
| phash | 2 | 131 | 104 | 0 | 0.0 | 55.0 |
| phash | 4 | 168 | 82 | 0 | 0.0 | 76.7 |
| phash | 5 | 168 | 82 | 0 | 0.0 | 76.7 |
| phash | 6 | 195 | 70 | 1 | 1.4 | 86.7 |
| phash | 8 | 229 | 58 | 3 | 5.2 | 96.7 |
| phash | 10 | 248 | 58 | 3 | 5.2 | 96.7 |
| phash | 12 | 261 | 55 | 4 | 7.3 | 98.3 |
| embedding | 0.900 | 276 | 67 | 12 | 17.9 | 61.7 |
| embedding | 0.930 | 153 | 103 | 5 | 4.9 | 48.3 |
| embedding | 0.950 | 107 | 123 | 4 | 3.3 | 35.0 |
| embedding | 0.970 | 61 | 146 | 1 | 0.7 | 20.0 |
| embedding | 0.980 | 43 | 156 | 0 | 0.0 | 15.0 |
| embedding | 0.990 | 24 | 168 | 0 | 0.0 | 5.0 |
| embedding | 0.995 | 13 | 179 | 0 | 0.0 | 0.0 |

## Reading this

The embedding signal merges no distinct originals at cosine 0.98 and above.

The operating point used on the real collection (perceptual Hamming <= 6, cosine >= 0.98) is chosen from this table, favouring recall over precision.
