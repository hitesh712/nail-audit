"""Rebuild partitions over source photographs, not over images.

The collection contains 3,835 images derived from 1,337 source photographs by
Roboflow augmentation. Partitioning over images - even with perceptual,
embedding and rotation-invariant deduplication - puts augmented copies of the
same photograph on both sides of the boundary. Measured on our previous
partition, 95.8% of test images had a training-set sibling.

Here the unit is `source_stem`. Two further decisions matter:

* Training uses every copy of each training photograph. Augmented copies are
  legitimate training augmentation; nothing is lost.
* Evaluation uses ONE copy per source photograph, chosen as the least augmented.
  Scoring all copies would count the same photograph up to three times and
  report a sample size that overstates the independent evidence.

Both evaluation variants are computed so the difference is visible rather than
assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail import PAPER_CLASSES  # noqa: E402
from nail.utils import df_to_markdown  # noqa: E402


def black_fraction(path: str) -> float:
    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        return float((a.mean(axis=2) < 0.06).mean())
    except Exception:  # noqa: BLE001
        return 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_rf.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "manifest_source.csv")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "29_source_split.md")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["label"].notna()].reset_index(drop=True)

    # One label per source photograph (they agree in practice; majority for safety).
    src = (
        df.groupby("source_stem")["label"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
    )
    groups = src["source_stem"].to_numpy()
    y = src["label"].to_numpy()
    print(f"{len(df)} images from {len(src)} source photographs")

    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=args.seed)
    _, test_idx = next(sgkf.split(groups.reshape(-1, 1), y, groups=groups))
    test_src = set(groups[test_idx])

    rest = np.array([g not in test_src for g in groups])
    sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=args.seed)
    _, val_idx = next(
        sgkf2.split(groups[rest].reshape(-1, 1), y[rest], groups=groups[rest])
    )
    val_src = set(groups[rest][val_idx])

    def assign(stem: str) -> str:
        if stem in test_src:
            return "test"
        if stem in val_src:
            return "val"
        return "train"

    df["split_source"] = df["source_stem"].map(assign)

    # Least-augmented copy per source, for single-copy evaluation.
    print("scoring augmentation level per image...")
    df["black_frac"] = [black_fraction(p) for p in tqdm(df["path"], unit="img")]
    df["eval_representative"] = False
    rep_idx = (
        df[df["split_source"].isin(["val", "test"])]
        .sort_values(["source_stem", "black_frac", "bytes"],
                     ascending=[True, True, False])
        .groupby("source_stem")
        .head(1)
        .index
    )
    df.loc[rep_idx, "eval_representative"] = True
    df.to_csv(args.out, index=False)

    # ---- verification
    straddle = int((df.groupby("source_stem")["split_source"].nunique() > 1).sum())
    train_src = set(df.loc[df["split_source"] == "train", "source_stem"])
    test_rows = df[df["split_source"] == "test"]
    contam = int(test_rows["source_stem"].isin(train_src).sum())

    img_counts = df.pivot_table(index="label", columns="split_source",
                                values="path", aggfunc="count", fill_value=0)
    src_counts = (
        df.drop_duplicates("source_stem")
        .pivot_table(index="label", columns="split_source", values="source_stem",
                     aggfunc="count", fill_value=0)
    )
    n_eval = int(df["eval_representative"].sum())

    lines = [
        "# Source-aware partition",
        "",
        f"- Images: **{len(df)}** from **{df['source_stem'].nunique()}** source "
        "photographs",
        f"- Source photographs straddling a partition boundary: **{straddle}**",
        f"- Test images whose source appears in training: **{contam}** "
        f"({100*contam/max(len(test_rows),1):.1f}%)",
        "",
        "## Source photographs per partition",
        "",
        df_to_markdown(src_counts, index_name="class"),
        "",
        "## Images per partition (training uses every copy)",
        "",
        df_to_markdown(img_counts, index_name="class"),
        "",
        "## Evaluation protocol",
        "",
        f"- Validation and test scored on **one representative copy per source "
        f"photograph** ({n_eval} images total), selected as the copy with the least "
        "rotation fill.",
        "- Scoring every copy would count the same photograph up to three times and "
        "report an independent sample size that does not exist.",
        "",
        "## Comparison with the previous partition",
        "",
        "| | previous (image-level) | this (source-level) |",
        "| --- | --- | --- |",
        f"| Test images sharing a source with training | 95.8% | "
        f"{100*contam/max(len(test_rows),1):.1f}% |",
        f"| Independent test photographs | ~23 | "
        f"{int(test_rows['source_stem'].nunique())} |",
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
