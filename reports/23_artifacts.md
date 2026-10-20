# Reliance on augmentation artefacts

Model `efficientnet_b0`, test set n=546. Unoccluded accuracy **98.53%**.

## Are the source images synthetically augmented?

- Mean share of near-black pixels per image: **18.9%**
- Images with >5% near-black pixels: **70.5%** of the test set
- Images with >15% near-black pixels: **57.9%**

Large black regions are the fill left by rotating an image without cropping.
Their prevalence indicates the distributed images are augmented derivatives
rather than raw photographs.

## Equal-area occlusion test

| area occluded | border occluded | centre occluded | border drop | centre drop |
| --- | --- | --- | --- | --- |
| 19% | 98.53% | 86.45% | 0.00 pp | 12.09 pp |
| 35% | 98.17% | 68.32% | 0.37 pp | 30.22 pp |
| 52% | 97.80% | 47.25% | 0.73 pp | 51.28 pp |

## Verdict

Occluding the centre is far more damaging than occluding an equal area at the border, so the model is predominantly reading the nail region. Border attention in Grad-CAM overlays reflects the coarse resolution of the final feature map rather than genuine reliance.

Any occlusion damages accuracy, so the two conditions are matched on
occluded area. The comparison between them, not either number alone,
carries the evidence.