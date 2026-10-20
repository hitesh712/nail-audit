"""Produce the manifest the training pipeline consumes for source-aware runs.

Training keeps every augmented copy of each training photograph. Validation and
test keep only the representative copy per source, so early stopping and the
final metric are both computed over independent photographs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_source.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "manifest_source_eval.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    keep = (df["split_source"] == "train") | (df["eval_representative"])
    out = df[keep].copy()
    out["split"] = out["split_source"]
    # The splitting unit, carried through so leakage checks still work downstream.
    out["group_id"] = out["source_stem"].astype("category").cat.codes
    out.to_csv(args.out, index=False)

    counts = out.pivot_table(index="label", columns="split", values="path",
                             aggfunc="count", fill_value=0)
    print(counts.to_string())
    print(f"\ntotal {len(out)} rows -> {args.out}")
    for s in ("train", "val", "test"):
        sub = out[out["split"] == s]
        print(f"  {s:5s} {len(sub):5d} images, {sub['source_stem'].nunique():5d} sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
