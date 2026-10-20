# Ablation - source-aware split, EfficientNet-B0, 12 epochs

## Main effects (paired, percentage points of macro-F1)

| factor | change | effect | 95% CI | p | pairs |
| --- | --- | --- | --- | --- | --- |
| `cfg_unfreeze_layer4` | False → True | **+26.59** | [+23.64, +29.54] | 0.0000 | 12 |
| `cfg_augment` | False → True | **-1.05** | [-4.15, +2.04] | 0.4697 | 12 |
| `cfg_balancing` | none → sampler | **+0.58** | [-0.63, +1.79] | 0.3162 | 12 |

## Augmentation effect, split by whether the trunk is trainable

| trunk | augmentation effect | 95% CI | p | pairs |
| --- | --- | --- | --- | --- |
| trainable | **+3.00** | [+0.55, +5.44] | 0.0253 | 6 |
| frozen | **-5.10** | [-7.96, -2.25] | 0.0058 | 6 |

**Interaction**: the augmentation effect is +8.10 pp larger when the trunk is trainable, 95% CI [+5.69, +10.51], p=0.0003 (6 paired).

## Reading this

Augmentation and trunk adaptation are not independent. With the trunk frozen,
the classifier can only fit whatever ImageNet features survive, and augmentation
adds variance it has no capacity to absorb - so it hurts. With the trunk
trainable, the network can actually learn the invariances augmentation supplies,
and it helps. Reporting a single main effect for augmentation therefore averages
over two opposite regimes and is misleading in either direction.

This also explains why the draft's Table VI is not reproducible: a +11.6 point
augmentation gain is only reachable if the comparison baseline differed in
something other than augmentation.

## Config ranking

| augment | balancing | trunk | macro-F1 |
| --- | --- | --- | --- |
| True | sampler | trainable | 0.9374 +/- 0.0027 |
| True | none | trainable | 0.9338 +/- 0.0154 |
| False | sampler | trainable | 0.9160 +/- 0.0081 |
| False | none | trainable | 0.8952 +/- 0.0227 |
| False | sampler | frozen | 0.6804 +/- 0.0225 |
| False | none | frozen | 0.6800 +/- 0.0201 |
| True | none | frozen | 0.6299 +/- 0.0184 |
| True | sampler | frozen | 0.6284 +/- 0.0038 |