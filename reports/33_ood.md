# Out-of-distribution behaviour

Model `efficientnet_b0` (seed 0). In-distribution: 191 held-out
test images. OOD: 1200 nail photographs from an independent public
collection (nail segmentation dataset, CC0), after removing 0
images that perceptually matched our training data.

## Does confidence fall on unfamiliar data?

| | in-distribution | OOD |
| --- | --- | --- |
| mean max-softmax | 0.873 | 0.525 |
| median max-softmax | 0.919 | 0.494 |
| mean entropy (nats) | 0.469 | 1.251 |

- OOD detection AUROC, max-softmax: **0.930**
- OOD detection AUROC, entropy: **0.942**

## What the abstention threshold actually does

| threshold | in-distribution retained | OOD wrongly accepted |
| --- | --- | --- |
| 0.50 | 96.3% | 48.7% |
| 0.70 | 88.0% | 16.5% |
| 0.90 | 58.6% | 3.2% |
| 0.95 | 33.5% | 0.9% |

## What the model predicts on OOD input

| class | share of OOD predictions |
| --- | --- |
| Onychogryphosis | 30.1% |
| Healthy_Nail | 28.7% |
| Acral_Lentiginous_Melanoma | 18.2% |
| Clubbing | 17.8% |
| Pitting | 2.7% |
| Blue_Finger | 2.5% |

## Reading this

The model separates familiar from unfamiliar input well (AUROC 0.930 from max-softmax alone). A confidence threshold set for in-distribution abstention will admit the OOD share shown above; at the draft's 0.7 it admits 16.5% of them.

Two caveats. These OOD images are unlabelled, so no accuracy can be computed
on them - only confidence behaviour. And they are still nails, making this a
*near*-OOD test; a deployed system would also meet non-nail input, which
would be a harder test than this one.

This is not a substitute for labelled external validation. It is the
strongest evidence available given that no public dataset covers these six
classes.