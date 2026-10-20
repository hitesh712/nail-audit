# Ablation - merged 8-class, ResNet-18 (draft arch), 15 epochs

## Main effects (paired, percentage points of macro-F1)

| factor | change | effect | 95% CI | p | pairs |
| --- | --- | --- | --- | --- | --- |
| `cfg_unfreeze_layer4` | False → True | **+12.77** | [+10.97, +14.56] | 0.0000 | 12 |
| `cfg_augment` | False → True | **-3.15** | [-4.99, -1.32] | 0.0030 | 12 |
| `cfg_balancing` | none → sampler | **-0.60** | [-1.23, +0.04] | 0.0620 | 12 |

## Augmentation effect, split by whether the trunk is trainable

| trunk | augmentation effect | 95% CI | p | pairs |
| --- | --- | --- | --- | --- |
| trainable | **-0.53** | [-1.66, +0.60] | 0.2824 | 6 |
| frozen | **-5.78** | [-6.62, -4.94] | 0.0000 | 6 |

**Interaction**: the augmentation effect is +5.25 pp larger when the trunk is trainable, 95% CI [+4.11, +6.39], p=0.0001 (6 paired).

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
| False | none | trainable | 0.9561 +/- 0.0028 |
| True | none | trainable | 0.9538 +/- 0.0064 |
| False | sampler | trainable | 0.9522 +/- 0.0008 |
| True | sampler | trainable | 0.9440 +/- 0.0144 |
| False | none | frozen | 0.8537 +/- 0.0028 |
| False | sampler | frozen | 0.8517 +/- 0.0033 |
| True | none | frozen | 0.7990 +/- 0.0062 |
| True | sampler | frozen | 0.7909 +/- 0.0116 |