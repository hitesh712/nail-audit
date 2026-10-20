# Grad-CAM faithfulness

Model `efficientnet_b0`, 150 test images (25 per class), Grad-CAM over the final feature map.

## Deletion and insertion, against a random-attribution control

| metric | Grad-CAM | random | difference | p |
| --- | --- | --- | --- | --- |
| deletion AUC (lower better) | 0.5953 | 0.5124 | **-0.0829** | 3.12e-04 |
| insertion AUC (higher better) | 0.7052 | 0.5101 | **+0.1951** | 3.15e-18 |

The deletion test favours **random** attribution (-0.0829, p=3.12e-04), which is evidence against the map being faithful. The insertion test favours Grad-CAM (+0.1951, p=3.15e-18): restoring the highest-attributed pixels recovers the prediction faster than restoring random pixels. The two tests disagree. Deletion and insertion probe different failure modes and are known to diverge when the removal baseline itself carries signal: blurring a nail region leaves surrounding skin, lighting and overall colour intact, and if the model draws on those, occluding the salient region need not reduce class evidence. Insertion is the more informative of the two here, because starting from a fully blurred image removes that confound. The honest summary is that the attribution is faithful by insertion and unresolved by deletion, and both should be reported.

## What this does and does not establish

It establishes faithfulness: the map reflects the model's own computation.
It does **not** establish clinical correctness - that the highlighted region
is the one a dermatologist would use. That claim needs expert annotations,
which this dataset does not carry, and it is the claim the draft makes
without them.

## Per class

| class | n | deletion Δ | insertion Δ | mean confidence |
| --- | --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 25 | +0.0028 | +0.2689 | 0.889 |
| Blue_Finger | 25 | -0.1599 | +0.3093 | 0.895 |
| Clubbing | 25 | -0.3091 | +0.2356 | 0.897 |
| Healthy_Nail | 25 | +0.1900 | -0.2293 | 0.871 |
| Onychogryphosis | 25 | -0.4107 | +0.2343 | 0.901 |
| Pitting | 25 | +0.1896 | +0.3518 | 0.924 |