# Metadata-only shortcut test

Features: width, height, aspect ratio, byte size, pixel count, bytes per
pixel, file extension. **No image content whatsoever.**

| model | accuracy | balanced accuracy | macro-F1 |
| --- | --- | --- | --- |
| Majority-class baseline | 53.00% | - | - |
| **Random forest on metadata only** | **65.00%** | 44.27% | 0.444 |

## Per class

| class                      | n   | correct | recall_pct |
|----------------------------|-----|---------|------------|
| Acral_Lentiginous_Melanoma | 18  | 4       | 22.200     |
| Blue_Finger                | 9   | 4       | 44.400     |
| Clubbing                   | 16  | 4       | 25         |
| Healthy_Nail               | 82  | 33      | 40.200     |
| Onychogryphosis            | 159 | 143     | 89.900     |
| Pitting                    | 16  | 7       | 43.800     |

## Feature importance

| feature         | importance |
|-----------------|------------|
| bytes           | 0.433      |
| bytes_per_pixel | 0.402      |
| pixels          | 0.052      |
| width           | 0.050      |
| height          | 0.044      |
| aspect          | 0.014      |
| ext.jpeg        | 0.003      |
| ext.jpg         | 0.000      |
| ext.png         | 0          |
| ext.webp        | 0          |

## Why this matters

Every image in the gurav source is exactly 416x416 JPEG at a median 13.5 KB.
The rasanjana source has 735 distinct resolutions at a median 46.7 KB. Most
classes are drawn from exactly one of the two sources, so resolution and
compression alone carry class information that has nothing to do with nail
pathology.

Accuracy reached here is a floor on the shortcut available to a CNN, which
also sees JPEG artefacts, sharpness and colour statistics that this test
cannot use. It does not prove the CNN takes the shortcut - but it does show
the shortcut exists, and the draft's design cannot rule it out.

The fix is a source-disjoint or source-balanced evaluation: a held-out set
in which class cannot be inferred from provenance.