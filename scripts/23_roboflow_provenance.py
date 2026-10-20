"""Group images by their true source photograph, recovered from the filename.

The collection was exported from Roboflow, whose augmentation pipeline names
derived files `<original stem>_jpg.rf.<32 hex>.jpg`. The stem before `.rf.` is
therefore the identity of the *source photograph*, and every file sharing a stem
is an augmented copy of one original.

This is ground truth for duplication, and it is stronger than any perceptual
signal: two augmentations of one photograph can differ by rotation, crop,
brightness, noise and composited patches simultaneously, which defeats pHash and
embedding similarity alike. Our audits caught 1.2% redundancy; if the stems say
otherwise, every partition built on those audits is contaminated and every
accuracy measured on them is inflated.

Run this before trusting any result on a Roboflow-exported dataset.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail.utils import df_to_markdown  # noqa: E402

RF = re.compile(r"^(?P<stem>.+?)\.rf\.[0-9a-f]{8,}\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def source_stem(filename: str) -> str:
    m = RF.match(filename)
    if m:
        stem = m.group("stem")
        # Roboflow appends the original extension before .rf.
        return re.sub(r"_(jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)
    return Path(filename).stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_rotdedup.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "manifest_rf.csv")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "28_roboflow.md")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df["filename"] = df["path"].map(lambda p: Path(p).name)
    df["source_stem"] = df["filename"].map(source_stem)
    n = len(df)
    n_src = df["source_stem"].nunique()
    matched = int(df["filename"].map(lambda f: bool(RF.match(f))).sum())

    sizes = df["source_stem"].value_counts()
    hist = sizes.value_counts().sort_index()

    # Does the published split respect source identity?
    straddle = df.groupby("source_stem")["split"].nunique()
    n_straddle = int((straddle > 1).sum())
    train_src = set(df.loc[df["split"] == "train", "source_stem"])
    test_rows = df[df["split"] == "test"]
    contaminated = int(test_rows["source_stem"].isin(train_src).sum())
    val_rows = df[df["split"] == "val"]
    val_contam = int(val_rows["source_stem"].isin(train_src).sum())

    # How much did the perceptual audits recover?
    grp_now = int(df["group_id_rot"].nunique())

    df.to_csv(args.out, index=False)

    lines = [
        "# Source-photograph provenance from filenames",
        "",
        f"- Images: **{n}**",
        f"- Filenames matching the Roboflow augmentation pattern: **{matched}** "
        f"({100*matched/n:.1f}%)",
        f"- Distinct source photographs: **{n_src}**",
        f"- Mean augmented copies per source: **{n/n_src:.2f}**",
        "",
        "| grouping | groups | implied redundancy |",
        "| --- | --- | --- |",
        f"| perceptual + embedding + dihedral | {int(df['group_id'].nunique())} | "
        f"{100*(n-int(df['group_id'].nunique()))/n:.1f}% |",
        f"| + rotation-invariant | {grp_now} | {100*(n-grp_now)/n:.1f}% |",
        f"| **source photograph (ground truth)** | **{n_src}** | "
        f"**{100*(n-n_src)/n:.1f}%** |",
        "",
        "## Does the published partition respect source identity?",
        "",
        f"- Source photographs split across partitions: **{n_straddle}**",
        f"- Test images whose source also appears in training: "
        f"**{contaminated} / {len(test_rows)} "
        f"({100*contaminated/max(len(test_rows),1):.1f}%)**",
        f"- Validation images whose source also appears in training: "
        f"**{val_contam} / {len(val_rows)} "
        f"({100*val_contam/max(len(val_rows),1):.1f}%)**",
        "",
        "## Copies per source photograph",
        "",
        df_to_markdown(pd.DataFrame({"sources": hist}).rename_axis("copies"),
                       index_name="copies"),
    ]

    rate = contaminated / max(len(test_rows), 1)
    if rate > 0.10:
        lines += [
            "",
            "## Verdict",
            "",
            f"**The partition is contaminated.** {100*rate:.1f}% of the test set "
            "consists of augmented copies of training photographs. Perceptual and "
            "embedding similarity did not detect them because Roboflow applies "
            "rotation, cropping, brightness, noise and patch compositing "
            "simultaneously, which moves an image far enough in both pixel and "
            "feature space to defeat those signals while preserving class evidence.",
            "",
            "Every accuracy reported on this partition is inflated by an unknown "
            "amount. Splits must be rebuilt over `source_stem` and all experiments "
            "rerun. The effective sample size is "
            f"**{n_src} photographs, not {n} images**.",
        ]
    else:
        lines += [
            "",
            "## Verdict",
            "",
            f"Only {100*rate:.1f}% of test images share a source photograph with "
            "training data, so the partition substantially respects source identity. "
            f"The effective sample size is nonetheless {n_src} photographs rather "
            f"than {n} images, and that is what should be reported.",
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
