"""Can the classes be predicted without looking at the images?

The two sources have starkly different technical signatures: gurav ships 3,835
images that are all exactly 416x416 JPEG at a median 13.5 KB, while rasanjana
ships 735 distinct resolutions at a median 46.7 KB. Because most classes are
drawn from exactly one source, a model can potentially separate them by
compression artefacts and resolution rather than by pathology.

This script trains on file metadata only - width, height, aspect ratio, byte
size, file extension. No pixels. Whatever accuracy it reaches is accuracy
available without examining a single nail, and it is a lower bound on how much
of the reported performance is attributable to dataset provenance rather than
disease features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail.utils import df_to_markdown  # noqa: E402


def featurise(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["width"] = df["width"]
    x["height"] = df["height"]
    x["aspect"] = df["width"] / df["height"].clip(lower=1)
    x["bytes"] = df["bytes"]
    x["pixels"] = df["width"] * df["height"]
    x["bytes_per_pixel"] = df["bytes"] / x["pixels"].clip(lower=1)
    ext = df["path"].str.lower().str.extract(r"(\.[a-z]+)$")[0].fillna("na")
    for e in [".jpg", ".jpeg", ".png", ".webp"]:
        x[f"ext{e}"] = (ext == e).astype(int)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest", type=Path, default=ROOT / "data" / "manifest_paper_split.csv"
    )
    ap.add_argument("--split-col", default="split_shipped")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "06_metadata.md")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["label"].notna()]

    tr = df[df[args.split_col] == "train"]
    va = df[df[args.split_col] == "val"]
    print(f"train {len(tr)}  val {len(va)}")

    xtr, xva = featurise(tr), featurise(va)
    ytr, yva = tr["label"].to_numpy(), va["label"].to_numpy()

    clf = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, random_state=args.seed, n_jobs=-1
    )
    clf.fit(xtr, ytr)
    pred = clf.predict(xva)

    dummy = DummyClassifier(strategy="most_frequent").fit(xtr, ytr)
    dpred = dummy.predict(xva)

    acc = accuracy_score(yva, pred)
    bacc = balanced_accuracy_score(yva, pred)
    mf1 = f1_score(yva, pred, average="macro", zero_division=0)
    dacc = accuracy_score(yva, dpred)

    imp = (
        pd.DataFrame({"feature": xtr.columns, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .set_index("feature")
        .round(4)
    )

    per_class = (
        pd.DataFrame({"true": yva, "pred": pred})
        .assign(correct=lambda d: d["true"] == d["pred"])
        .groupby("true")
        .agg(n=("correct", "size"), correct=("correct", "sum"))
        .assign(recall_pct=lambda d: (100 * d["correct"] / d["n"]).round(1))
    )

    lines = [
        "# Metadata-only shortcut test",
        "",
        "Features: width, height, aspect ratio, byte size, pixel count, bytes per",
        "pixel, file extension. **No image content whatsoever.**",
        "",
        "| model | accuracy | balanced accuracy | macro-F1 |",
        "| --- | --- | --- | --- |",
        f"| Majority-class baseline | {dacc*100:.2f}% | - | - |",
        f"| **Random forest on metadata only** | **{acc*100:.2f}%** | "
        f"{bacc*100:.2f}% | {mf1:.3f} |",
        "",
        "## Per class",
        "",
        df_to_markdown(per_class, index_name="class"),
        "",
        "## Feature importance",
        "",
        df_to_markdown(imp, index_name="feature"),
        "",
        "## Why this matters",
        "",
        "Every image in the gurav source is exactly 416x416 JPEG at a median 13.5 KB.",
        "The rasanjana source has 735 distinct resolutions at a median 46.7 KB. Most",
        "classes are drawn from exactly one of the two sources, so resolution and",
        "compression alone carry class information that has nothing to do with nail",
        "pathology.",
        "",
        "Accuracy reached here is a floor on the shortcut available to a CNN, which",
        "also sees JPEG artefacts, sharpness and colour statistics that this test",
        "cannot use. It does not prove the CNN takes the shortcut - but it does show",
        "the shortcut exists, and the draft's design cannot rule it out.",
        "",
        "The fix is a source-disjoint or source-balanced evaluation: a held-out set",
        "in which class cannot be inferred from provenance.",
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
