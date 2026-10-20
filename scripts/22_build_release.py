"""Assemble the public release package.

What goes in: code, split indices, duplicate-group assignments, generated reports
and figures. What stays out: the images themselves. Collection A is CC BY-NC-SA,
which would permit redistribution under the same terms, but the paper states that
no images are redistributed and keeping that promise is simpler than managing a
share-alike obligation on 3,835 files. Readers reproduce by downloading the source
collection and joining on filename.

Absolute paths are stripped: a released index full of D:\\nail_disease_paper\\...
is useless to anyone else and leaks the author's directory layout.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CODE_LICENSE = """MIT License

Copyright (c) {year} Himanshu N. Saha, Rudransh R. Bhatt, Hitesh Kag

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

DATA_NOTICE = """# Licence of the released data files

The CSV files under `splits/` are derived metadata: filenames, class labels,
duplicate-group identifiers and partition assignments. They contain no image
data.

They describe the *Nail Disease Detection* dataset by N. Gurav, which is
distributed on Kaggle under Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International (CC BY-NC-SA 4.0):

  https://www.kaggle.com/datasets/nikhilgurav21/nail-disease-detection-dataset

Because these files are derived from that dataset, they are released under the
same licence: CC BY-NC-SA 4.0. You must attribute the original dataset, may not
use these materials commercially, and must distribute any derivative under the
same terms.

No images from the source dataset are redistributed here. To reproduce our
experiments, download the dataset from the link above and join on the `filename`
column.

The source code in this repository is released separately under the MIT licence;
see LICENSE-CODE.
"""

CITATION = """cff-version: 1.2.0
title: >-
  Auditing public biomedical image collections for duplication, provenance
  confounding and shortcut learning
message: If you use this software or the released split indices, please cite it.
type: software
authors:
  - family-names: Saha
    given-names: Himanshu N.
    affiliation: Marwadi University
  - family-names: Bhatt
    given-names: Rudransh R.
    affiliation: Marwadi University
  - family-names: Kag
    given-names: Hitesh
    affiliation: Marwadi University
repository-code: 'https://github.com/REPLACE-ME/nail-audit'
license: MIT
date-released: '{today}'
"""

README = """# Auditing public biomedical image collections

Code and split indices for *Beyond accuracy: auditing public biomedical image
collections for duplication, provenance confounding and shortcut learning, with a
case study in nail disease classification*.

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
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "release")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_rotdedup.csv")
    args = ap.parse_args()

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    for sub in ("splits", "src", "scripts", "reports", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ---- split index, with absolute paths stripped
    df = pd.read_csv(args.manifest)
    cols = {
        "filename": df["path"].map(lambda p: Path(p).name),
        "source_relpath": df["rel_path"],
        "class": df["label"],
        # The splitting unit. Everything else is secondary.
        "source_photograph": df["source_stem"],
        "split": df["split_source"],
        "eval_representative": df["eval_representative"],
        "width": df["width"],
        "height": df["height"],
        "bytes": df["bytes"],
        "sha256": df["sha256"],
    }
    # Superseded groupings, kept so the failure is reproducible by others.
    for src, dest in (("group_id", "group_perceptual"),
                      ("group_id_rot", "group_rotation_invariant")):
        if src in df.columns:
            cols[dest] = df[src]
    rel = pd.DataFrame(cols)
    rel.to_csv(out / "splits" / "splits.csv", index=False)

    counts = rel.pivot_table(index="class", columns="split", values="filename",
                             aggfunc="count", fill_value=0)
    n_src = rel["source_photograph"].nunique()
    (out / "splits" / "README.md").write_text(
        "# Split indices\n\n"
        "One row per image. No image data is included.\n\n"
        f"**{len(rel)} images derive from {n_src} source photographs.** Partitions "
        "are constructed over `source_photograph`, never over individual images.\n\n"
        "| column | meaning |\n|---|---|\n"
        "| `filename` | basename in the source dataset |\n"
        "| `source_relpath` | path relative to the dataset root |\n"
        "| `class` | one of six nail conditions |\n"
        "| `source_photograph` | **the splitting unit**, parsed from the filename |\n"
        "| `split` | train / val / test, assigned per source photograph |\n"
        "| `eval_representative` | true for the one copy scored in val/test |\n"
        "| `group_perceptual` | superseded: hash + embedding + dihedral grouping |\n"
        "| `group_rotation_invariant` | superseded: 24-rotation grouping |\n"
        "| `sha256` | checksum, to verify you have the same file |\n\n"
        "The two `group_*` columns are retained so that the failure reported in the "
        "paper is reproducible: both report roughly 1.2% redundancy where the true "
        "figure is 65%.\n\n"
        "Training uses every augmented copy of a training photograph. Validation and "
        "test score only the rows with `eval_representative = True`, one per "
        "photograph, so the reported sample size reflects independent photographs "
        "rather than augmented copies.\n\n"
        "## Counts (images)\n\n" + counts.to_string() + "\n",
        encoding="utf-8")

    # ---- code
    shutil.copytree(ROOT / "src" / "nail", out / "src" / "nail",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for p in sorted((ROOT / "scripts").glob("*.py")):
        shutil.copy2(p, out / "scripts" / p.name)

    # ---- reports and figures
    n_rep = 0
    for p in sorted((ROOT / "results").glob("*.md")):
        if p.name.startswith("_"):
            continue
        shutil.copy2(p, out / "reports" / p.name)
        n_rep += 1
    n_fig = 0
    for p in sorted((ROOT / "results" / "figures").glob("*")):
        if p.suffix.lower() in {".png", ".pdf", ".txt"}:
            shutil.copy2(p, out / "figures" / p.name)
            n_fig += 1

    # ---- metadata
    (out / "README.md").write_text(README, encoding="utf-8")
    (out / "LICENSE-CODE").write_text(CODE_LICENSE.format(year=date.today().year),
                                      encoding="utf-8")
    (out / "LICENSE-DATA.md").write_text(DATA_NOTICE, encoding="utf-8")
    (out / "CITATION.cff").write_text(CITATION.format(today=date.today().isoformat()),
                                      encoding="utf-8")
    shutil.copy2(ROOT / "requirements.txt", out / "requirements.txt")
    (out / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n.venv/\ndata/\nresults/runs*/\n*.pt\n*.npz\n",
        encoding="utf-8")

    n_img = len(rel)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Release assembled at {out}")
    print(f"  splits.csv      {n_img} rows, no images")
    print(f"  scripts         {len(list((out / 'scripts').glob('*.py')))} files")
    print(f"  reports         {n_rep} markdown files")
    print(f"  figures         {n_fig} files")
    print(f"  total size      {size:.1f} MB")
    print("\nStill to do by hand:")
    print("  1. Replace REPLACE-ME in CITATION.cff with the GitHub path")
    print("  2. git init, commit, push to a public GitHub repository")
    print("  3. Link the repository to Zenodo and cut a release to mint a DOI")
    print("  4. Put that DOI in the manuscript's data availability statement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
