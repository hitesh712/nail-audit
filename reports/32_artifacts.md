# Reliance on augmentation artefacts

Model `efficientnet_b0`, test set n=191. Unoccluded accuracy **92.15%**.

## Are the source images synthetically augmented?

- Mean share of near-black pixels per image: **7.5%**
- Images with >5% near-black pixels: **32.5%** of the test set
- Images with >15% near-black pixels: **20.9%**

Large black regions are the fill left by rotating an image without cropping.
Their prevalence indicates the distributed images are augmented derivatives
rather than raw photographs.

## Equal-area occlusion test

| area occluded | border occluded | centre occluded | border drop | centre drop |
| --- | --- | --- | --- | --- |
| 19% | 92.67% | 77.49% | -0.52 pp | 14.66 pp |
| 35% | 93.72% | 60.73% | -1.57 pp | 31.41 pp |
| 52% | 91.10% | 49.21% | 1.05 pp | 42.93 pp |

## Verdict

Occluding the centre is far more damaging than occluding an equal area at the border, so the model is predominantly reading the nail region. Border attention in Grad-CAM overlays reflects the coarse resolution of the final feature map rather than genuine reliance.

Any occlusion damages accuracy, so the two conditions are matched on
occluded area. The comparison between them, not either number alone,
carries the evidence.