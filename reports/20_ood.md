# Out-of-distribution behaviour

Model `efficientnet_b0` (seed 0). In-distribution: 546 held-out
test images. OOD: 1199 nail photographs from an independent public
collection (nail segmentation dataset, CC0), after removing 1
images that perceptually matched our training data.

## Does confidence fall on unfamiliar data?

| | in-distribution | OOD |
| --- | --- | --- |
| mean max-softmax | 0.901 | 0.500 |
| median max-softmax | 0.921 | 0.464 |
| mean entropy (nats) | 0.442 | 1.315 |

- OOD detection AUROC, max-softmax: **0.975**
- OOD detection AUROC, entropy: **0.978**

## What the abstention threshold actually does

| threshold | in-distribution retained | OOD wrongly accepted |
| --- | --- | --- |
| 0.50 | 98.4% | 42.5% |
| 0.70 | 96.9% | 12.8% |
| 0.90 | 71.6% | 1.6% |
| 0.95 | 10.1% | 0.3% |

## What the model predicts on OOD input

| class | share of OOD predictions |
| --- | --- |
| Onychogryphosis | 26.7% |
| Healthy_Nail | 25.7% |
| Acral_Lentiginous_Melanoma | 22.5% |
| Clubbing | 17.2% |
| Blue_Finger | 4.3% |
| Pitting | 3.7% |

## Reading this

The model separates familiar from unfamiliar input well (AUROC 0.975 from max-softmax alone). A confidence threshold set for in-distribution abstention will admit the OOD share shown above; at the draft's 0.7 it admits 12.8% of them.

Two caveats. These OOD images are unlabelled, so no accuracy can be computed
on them - only confidence behaviour. And they are still nails, making this a
*near*-OOD test; a deployed system would also meet non-nail input, which
would be a harder test than this one.

This is not a substitute for labelled external validation. It is the
strongest evidence available given that no public dataset covers these six
classes.