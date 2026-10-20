# External validation: what is actually available

Searched the Kaggle catalogue via the API across seven query terms
(`nail disease`, `nail`, `onychomycosis`, `nail fungus`, `subungual melanoma`,
`clubbing finger`, `dermatology clinical images`), excluding the two sources
already used. Reproduce with `scripts/10_find_external.py`.

## Finding

**No public dataset covers this study's six classes.** The catalogue is dominated
by skin-lesion collections (ISIC 2017/2019, HAM10000 derivatives, DermNet, PH2,
HIBA), which contain dermoscopic or clinical images of skin lesions rather than
nail pathology.

The only nail-specific results are:

| Dataset | Licence | Why it does not serve |
| --- | --- | --- |
| Nail Segmentation Dataset V2 | CC0 | Nail photographs with segmentation masks, no disease labels |
| DermNet skin disease images | Unknown | One combined "Nail Fungus and other Nail Disease" category — maps to onychomycosis, the class excluded from this study |
| Various psoriasis sets | Unknown / mixed | Skin psoriasis, not nail psoriasis |

Nothing covers acral lentiginous melanoma, blue finger, clubbing, onychogryphosis
or pitting as labelled nail classes.

## Consequence

Labelled external validation is not achievable from public data for this class
set. Two honest responses, both taken:

1. **State the limitation with evidence** rather than as a generic "future work"
   line. The absence is a property of the field's data landscape, not an oversight.
2. **Run the strongest available substitute** — an out-of-distribution check using
   the CC0 nail segmentation photographs as an independent, unlabelled source. That
   cannot measure accuracy, but it can measure whether the model recognises
   unfamiliar input. See `20_ood.md`.

## For the paper

> No publicly available dataset provides labelled examples of the six conditions
> studied here, so external validation on independent labelled data was not
> possible. We therefore report an out-of-distribution analysis on 1,199 nail
> photographs from an independent public collection, and we identify prospective
> collection with dermatologist labelling as the necessary next step.

## If external validation becomes a requirement

The realistic routes, in ascending cost:

- **DermNet nail category** — would validate onychomycosis only, a class this study
  excludes. Viable if onychomycosis is added back as a seventh class.
- **PAD-UFES-20** — clinical smartphone images, Brazilian cohort, genuine
  distribution shift; nail coverage would need checking image by image.
- **Prospective collection** with a partnering dermatology department under
  institutional ethics approval. 150–300 images suffice for validation only. This
  is the route that also supplies the dermatologist co-author that higher-tier
  venues expect.
