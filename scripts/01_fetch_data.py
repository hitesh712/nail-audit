"""Fetch the two public source datasets.

Requires Kaggle credentials. Either:
  1. Put kaggle.json at %USERPROFILE%\\.kaggle\\kaggle.json (Kaggle > Settings >
     API > Create New Token), or
  2. Set KAGGLE_USERNAME and KAGGLE_KEY in the environment, or
  3. Download both zips by hand and extract them under data/raw/<source_id>/.

Whichever route you take, record the download date and dataset version in
results/data_provenance.md. The paper must state exactly where these images came
from - the current draft's claim of clinic collection with IRB clearance is not
compatible with sourcing them from Kaggle.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "gurav": "nikhilgurav21/nail-disease-detection-dataset",
    "rasanjana": "josephrasanjana/nail-disease-image-classification-dataset",
}


def have_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not have_credentials():
        print("No Kaggle credentials found.\n")
        print("Set them up with ONE of:")
        print(r"  - place kaggle.json at %USERPROFILE%\.kaggle\kaggle.json")
        print("  - set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        print("\nOr download manually and extract to:")
        for sid, ref in SOURCES.items():
            print(f"  {args.out / sid}   <- https://www.kaggle.com/datasets/{ref}")
        return 1

    import kagglehub

    for sid, ref in SOURCES.items():
        dest = args.out / sid
        if dest.exists() and any(dest.rglob("*")):
            print(f"[skip] {sid} already present at {dest}")
            continue
        print(f"[get ] {sid} <- {ref}")
        cached = Path(kagglehub.dataset_download(ref))
        dest.mkdir(parents=True, exist_ok=True)
        for item in cached.iterdir():
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        print(f"[ok  ] {sid} -> {dest}")

    print("\nNext: python scripts/02_build_manifest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
