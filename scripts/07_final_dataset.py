"""Build the final dataset: gurav only, six classes, orientation-aware dedup.

rasanjana is dropped. That single decision removes, at once:

  * the onychomycosis/onychogryphosis conflation (722 mislabelled images),
  * the source confound (65% reachable from file metadata alone),
  * an "Unknown" licence granting no redistribution rights,
  * 24.2% internal duplication.

and measured accuracy goes *up* rather than down. What remains is a single-source,
single-licence, 0.3%-redundant dataset of 3,835 images.

Duplicate groups here come from the orientation-invariant audit
(`group_id_dihedral`), which is stricter than plain pHash, so splits respect
flipped and rotated copies too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail import PAPER_CLASSES  # noqa: E402
from nail.splits import contamination_rate, make_splits, summarise_splits  # noqa: E402
from nail.utils import df_to_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest", type=Path, default=ROOT / "data" / "manifest_dihedral.csv"
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "manifest_final.csv")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "08_final_dataset.md")
    ap.add_argument("--source", default="gurav")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if "group_id_dihedral" not in df.columns:
        print("manifest lacks group_id_dihedral; run 03b_dihedral_audit.py first")
        return 1

    before = len(df)
    df = df[(df["source"] == args.source) & df["label"].notna()].copy()
    # Use the stricter, orientation-aware grouping as the split unit.
    df["group_id_plain"] = df["group_id"]
    df["group_id"] = df["group_id_dihedral"]

    missing = sorted(set(df["label"]) - set(PAPER_CLASSES))
    if missing:
        print(f"unexpected classes in {args.source}: {missing}")
        return 1

    split_df, report = make_splits(df, test_frac=0.15, val_frac=0.15, seed=args.seed)
    contam = contamination_rate(split_df, "split", held_out="test")
    split_df.to_csv(args.out, index=False)

    n_groups = split_df["group_id"].nunique()
    counts = (
        split_df.groupby("label")
        .agg(images=("path", "size"), groups=("group_id", "nunique"))
        .reindex(PAPER_CLASSES)
        .fillna(0)
        .astype(int)
    )

    lines = [
        "# Final dataset",
        "",
        f"Source: **{args.source}** only. Dropped {before - len(df)} images from other",
        "sources and unmapped folders.",
        "",
        f"- Images: **{len(df)}**",
        f"- Duplicate groups (orientation-aware): **{n_groups}** "
        f"({len(df) - n_groups} redundant, {100*(len(df)-n_groups)/len(df):.1f}%)",
        f"- Classes: **{len(counts)}**",
        "- Licence: CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike)",
        "",
        "## Class distribution",
        "",
        df_to_markdown(counts, index_name="class"),
        "",
        report.to_markdown(),
        "",
        "## Verification",
        "",
        f"- Groups straddling a split boundary: **0** (asserted by `make_splits`)",
        f"- Test images sharing a duplicate group with training: "
        f"**{contam['n_contaminated']} / {contam['n_test']} "
        f"({contam['rate']*100:.1f}%)**",
        "",
        "## What dropping rasanjana removed",
        "",
        "| problem | before | after |",
        "| --- | --- | --- |",
        "| Onychomycosis filed as Onychogryphosis | 722 images | none |",
        "| Psoriasis silently dropped | 433 images | not applicable |",
        "| Source confound (metadata-only accuracy) | 65% vs 53% baseline | none |",
        "| Licence | one CC BY-NC-SA, one Unknown | single CC BY-NC-SA |",
        "| Duplication | 6.9% pooled, 24.2% in rasanjana | 1.2% |",
        "",
        "The cost is 1,032 images and a smaller Healthy_Nail class (343 rather than",
        "653). Measured accuracy rose rather than fell.",
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nManifest -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
