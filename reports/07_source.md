# Source-controlled evaluation

Model `resnet18_paper`, group-aware splits, seeds [0, 1, 2].

Each row trains and tests within a single source, so class cannot be
inferred from provenance.

| condition | images | classes | test n | accuracy [95% CI] | balanced acc [95% CI] |
| --- | --- | --- | --- | --- | --- |
| gurav_only (seed 0) | 3835 | 6 | 550 | 0.967 [0.951, 0.980] | 0.966 [0.949, 0.981] |
| rasanjana_only (seed 0) | 1465 | 3 | 217 | 0.853 [0.802, 0.903] | 0.878 [0.836, 0.919] |
| gurav_only (seed 1) | 3835 | 6 | 548 | 0.967 [0.951, 0.980] | 0.968 [0.951, 0.982] |
| rasanjana_only (seed 1) | 1465 | 3 | 217 | 0.931 [0.899, 0.959] | 0.946 [0.919, 0.969] |
| gurav_only (seed 2) | 3835 | 6 | 547 | 0.954 [0.938, 0.971] | 0.951 [0.932, 0.969] |
| rasanjana_only (seed 2) | 1465 | 3 | 219 | 0.936 [0.900, 0.968] | 0.944 [0.912, 0.971] |

## Per class

### gurav_only (seed 0)

| class                      | precision | recall | f1    | support |
|----------------------------|-----------|--------|-------|---------|
| Acral_Lentiginous_Melanoma | 0.963     | 0.972  | 0.967 | 107     |
| Blue_Finger                | 0.976     | 0.943  | 0.960 | 88      |
| Clubbing                   | 0.922     | 0.964  | 0.943 | 111     |
| Healthy_Nail               | 1         | 0.960  | 0.980 | 50      |
| Onychogryphosis            | 0.990     | 0.980  | 0.985 | 99      |
| Pitting                    | 0.979     | 0.979  | 0.979 | 95      |

### rasanjana_only (seed 0)

| class         | precision | recall | f1    | support |
|---------------|-----------|--------|-------|---------|
| Healthy_Nail  | 1         | 0.953  | 0.976 | 43      |
| Onychomycosis | 0.924     | 0.787  | 0.850 | 108     |
| Psoriasis     | 0.702     | 0.894  | 0.787 | 66      |

### gurav_only (seed 1)

| class                      | precision | recall | f1    | support |
|----------------------------|-----------|--------|-------|---------|
| Acral_Lentiginous_Melanoma | 0.946     | 0.981  | 0.963 | 107     |
| Blue_Finger                | 0.966     | 0.955  | 0.960 | 88      |
| Clubbing                   | 0.964     | 0.973  | 0.969 | 111     |
| Healthy_Nail               | 1         | 0.980  | 0.990 | 49      |
| Onychogryphosis            | 0.990     | 0.960  | 0.975 | 100     |
| Pitting                    | 0.957     | 0.957  | 0.957 | 93      |

### rasanjana_only (seed 1)

| class         | precision | recall | f1    | support |
|---------------|-----------|--------|-------|---------|
| Healthy_Nail  | 0.978     | 1      | 0.989 | 44      |
| Onychomycosis | 0.960     | 0.898  | 0.928 | 108     |
| Psoriasis     | 0.859     | 0.938  | 0.897 | 65      |

### gurav_only (seed 2)

| class                      | precision | recall | f1    | support |
|----------------------------|-----------|--------|-------|---------|
| Acral_Lentiginous_Melanoma | 0.946     | 0.981  | 0.964 | 108     |
| Blue_Finger                | 0.937     | 0.841  | 0.886 | 88      |
| Clubbing                   | 0.921     | 0.955  | 0.938 | 110     |
| Healthy_Nail               | 1         | 0.940  | 0.969 | 50      |
| Onychogryphosis            | 0.990     | 0.990  | 0.990 | 98      |
| Pitting                    | 0.959     | 1      | 0.979 | 93      |

### rasanjana_only (seed 2)

| class         | precision | recall | f1    | support |
|---------------|-----------|--------|-------|---------|
| Healthy_Nail  | 1         | 1      | 1     | 43      |
| Onychomycosis | 0.935     | 0.935  | 0.935 | 108     |
| Psoriasis     | 0.897     | 0.897  | 0.897 | 68      |

## Reading this

Compare against the pooled group-aware number in `05_leakage.md`. If the
single-source accuracy is comparable, the model is separating classes on
image content rather than on which dataset the file came from, and the
provenance confound - though real and worth disclosing - is not carrying
the result. If it drops sharply, the pooled figure was substantially
measuring provenance and the paper's central claim does not hold.

Note that gurav-only is the stronger of the two tests: 3,835 images across
6 classes, all identical 416x416 JPEGs, so there is no technical signature
distinguishing any class from any other.