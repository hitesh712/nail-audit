# Source-photograph provenance from filenames

- Images: **3835**
- Filenames matching the Roboflow augmentation pattern: **3835** (100.0%)
- Distinct source photographs: **1337**
- Mean augmented copies per source: **2.87**

| grouping | groups | implied redundancy |
| --- | --- | --- |
| perceptual + embedding + dihedral | 3790 | 1.2% |
| + rotation-invariant | 3789 | 1.2% |
| **source photograph (ground truth)** | **1337** | **65.1%** |

## Does the published partition respect source identity?

- Source photographs split across partitions: **879**
- Test images whose source also appears in training: **523 / 546 (95.8%)**
- Validation images whose source also appears in training: **524 / 554 (94.6%)**

## Copies per source photograph

| copies | sources |
|--------|---------|
| 1      | 1       |
| 2      | 174     |
| 3      | 1162    |

## Verdict

**The partition is contaminated.** 95.8% of the test set consists of augmented copies of training photographs. Perceptual and embedding similarity did not detect them because Roboflow applies rotation, cropping, brightness, noise and patch compositing simultaneously, which moves an image far enough in both pixel and feature space to defeat those signals while preserving class evidence.

Every accuracy reported on this partition is inflated by an unknown amount. Splits must be rebuilt over `source_stem` and all experiments rerun. The effective sample size is **1337 photographs, not 3835 images**.