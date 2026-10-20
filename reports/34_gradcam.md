# Grad-CAM faithfulness

Model `efficientnet_b0`, 150 test images (25 per class), Grad-CAM over the final feature map.

## Deletion and insertion, against a random-attribution control

| metric | Grad-CAM | random | difference | p |
| --- | --- | --- | --- | --- |
| deletion AUC (lower better) | 0.4527 | 0.4942 | **+0.0415** | 2.17e-02 |
| insertion AUC (higher better) | 0.7820 | 0.4956 | **+0.2864** | 7.46e-41 |

The deletion test favours Grad-CAM (+0.0415, p=2.17e-02): removing the highest-attributed pixels collapses the prediction faster than removing random pixels. The insertion test favours Grad-CAM (+0.2864, p=7.46e-41): restoring the highest-attributed pixels recovers the prediction faster than restoring random pixels.

## What this does and does not establish

It establishes faithfulness: the map reflects the model's own computation.
It does **not** establish clinical correctness - that the highlighted region
is the one a dermatologist would use. That claim needs expert annotations,
which this dataset does not carry, and it is the claim the draft makes
without them.

## Per class

| class | n | deletion Δ | insertion Δ | mean confidence |
| --- | --- | --- | --- | --- |
| Acral_Lentiginous_Melanoma | 25 | +0.1037 | +0.3180 | 0.847 |
| Blue_Finger | 25 | +0.0401 | +0.3452 | 0.854 |
| Clubbing | 25 | -0.1442 | +0.2136 | 0.921 |
| Healthy_Nail | 25 | +0.1347 | +0.0386 | 0.867 |
| Onychogryphosis | 25 | -0.1150 | +0.4653 | 0.880 |
| Pitting | 25 | +0.2296 | +0.3379 | 0.881 |