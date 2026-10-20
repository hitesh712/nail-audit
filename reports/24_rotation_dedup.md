# Rotation-invariant duplicate audit

3835 images matched under 24 rotations of a centre crop, pHash Hamming <= 6.

| grouping | groups | redundant | % |
| --- | --- | --- | --- |
| dihedral (previous) | 3790 | 45 | 1.2% |
| **rotation-invariant** | **3789** | **46** | **1.2%** |

## Effect on the existing split

- Rotation groups straddling a split boundary: **15**
- Test images sharing a rotation group with training data: **7 / 546 (1.3%)**

## Group size distribution

| group size | groups |
|------------|--------|
| 1          | 3759   |
| 2          | 24     |
| 4          | 3      |
| 5          | 2      |
| 6          | 1      |

## Verdict

**The split is contaminated under the stricter grouping.** 7 test images are rotated copies of training images that the dihedral audit could not see. Splits must be rebuilt over `group_id_rot` and every accuracy restated.