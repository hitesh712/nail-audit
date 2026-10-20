# Ablation - gurav-only, ResNet-50, 12 epochs

## Main effects (paired, percentage points of macro-F1)

| factor | change | effect | 95% CI | p | pairs |
| --- | --- | --- | --- | --- | --- |
| `cfg_unfreeze_layer4` | False → True | **+23.17** | [+21.11, +25.23] | 0.0000 | 12 |
| `cfg_augment` | False → True | **-1.35** | [-3.44, +0.73] | 0.1807 | 12 |
| `cfg_balancing` | none → sampler | **-0.25** | [-0.76, +0.26] | 0.3058 | 12 |

## Augmentation effect, split by whether the trunk is trainable

| trunk | augmentation effect | 95% CI | p | pairs |
| --- | --- | --- | --- | --- |
| trainable | **+1.62** | [+0.46, +2.78] | 0.0159 | 6 |
| frozen | **-4.33** | [-5.48, -3.17] | 0.0002 | 6 |

**Interaction**: the augmentation effect is +5.95 pp larger when the trunk is trainable, 95% CI [+4.29, +7.60], p=0.0002 (6 paired).

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
| True | sampler | trainable | 0.9675 ± 0.0053 |
| True | none | trainable | 0.9625 ± 0.0011 |
| False | none | trainable | 0.9508 ± 0.0040 |
| False | sampler | trainable | 0.9468 ± 0.0100 |
| False | none | frozen | 0.7475 ± 0.0029 |
| False | sampler | frozen | 0.7461 ± 0.0031 |
| True | none | frozen | 0.7083 ± 0.0148 |
| True | sampler | frozen | 0.6988 ± 0.0042 |