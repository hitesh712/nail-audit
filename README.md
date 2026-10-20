# Auditing public biomedical image collections

Code and split indices for *Similarity-Based Deduplication Fails on Composed
Augmentation: A Provenance-First Curation Protocol for Public Image Collections*,
submitted to IEEE Transactions on Big Data.

**No images are redistributed.** Download the source dataset and join on filename.

## What this is

Reported accuracies on public biomedical image collections are often near ceiling
and hard to interpret, because the collections carry duplication, provenance
confounding and synthetic augmentation that standard evaluation does not detect.
This repository implements an audit protocol for those three failure modes, plus
the full experimental pipeline for the nail disease case study.

**The headline finding is a negative one.** Four perceptual deduplication signals —
exact hashing, dihedral perceptual hashing, ImageNet embedding similarity, and
24-angle rotation-invariant hashing — each reported 1.2% redundancy in this
collection. The true figure is **65%**: 3,835 images derive from 1,337 source
photographs, recoverable only from the filenames. A partition built on the best
perceptual grouping left 95.8% of held-out images sharing a photograph with
training data.

| | Image-level partition | Source-aware partition |
|---|---|---|
| Balanced accuracy (3 seeds) | 98.80 ± 0.42% | **93.80 ± 1.91%** |
| Independent test items | 23 | 191 |
| Architecture comparisons significant | yes | **none** |

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Unix
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Download the source collection, then:

```bash
python scripts/00_smoke_test.py --with-embeddings   # verifies the audit logic
python scripts/02_build_manifest.py
python scripts/03_dedup_audit.py --sweep
python scripts/03b_dihedral_audit.py
python scripts/17_rotation_dedup.py
python scripts/07_final_dataset.py
python scripts/05_train.py --sweep single --model efficientnet_b0 --seeds 0 1 2 --epochs 40
python scripts/06_evaluate.py --run results/runs_headline/single_efficientnet_b0_s0
```

To use our exact partitions instead of regenerating them, join `splits/splits.csv`
on the `filename` column.

## The audit

| Check | Script | What it catches |
|---|---|---|
| Exact and perceptual duplicates | `03_dedup_audit.py` | Re-encodes, rescales |
| Orientation-invariant duplicates | `03b_dihedral_audit.py` | Flipped and 90°-rotated copies |
| Rotation-invariant duplicates | `17_rotation_dedup.py` | Arbitrary-angle rotated copies |
| Provenance shortcut | `05c_metadata_shortcut.py` | Class predictable from file metadata |
| Shortcut learning | `16_artifact_reliance.py` | Reliance on framing over lesion |
| Leakage effect | `05b_leakage_experiment.py` | Whether contamination inflates accuracy |

`00_smoke_test.py` validates the audit logic against synthetic data with known
duplicate structure. Run it before trusting any result.

## Layout

```
splits/      partition and duplicate-group indices (no images)
src/nail/    library: dedup, splits, models, training, metrics
scripts/     numbered pipeline stages
reports/     generated markdown reports for every experiment
figures/     manuscript figures
```

## Licences

Code: MIT (`LICENSE-CODE`). Derived data files: CC BY-NC-SA 4.0, matching the
source dataset (`LICENSE-DATA.md`).

## Citation

See `CITATION.cff`.
