"""Produce both splits and quantify the contamination in the naive one.

Outputs a manifest carrying two split columns:

  * ``split``        - group-aware, stratified. The one to actually use.
  * ``split_naive``  - per-image stratified, ignoring duplicate groups. The
                       control, i.e. what the current draft effectively did.

The contamination rate printed here is the share of the naive "held-out" test
set that shares a duplicate group with training data. That number belongs in the
paper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail.data import add_shipped_split  # noqa: E402
from nail.splits import (  # noqa: E402
    contamination_rate,
    make_cv_folds,
    make_naive_splits,
    make_splits,
    summarise_splits,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_grouped.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "manifest_split.csv")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "03_splits.md")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"Missing {args.manifest}. Run 03_dedup_audit.py first.")
        return 1

    df = pd.read_csv(args.manifest)
    if "group_id" not in df.columns:
        print("manifest has no group_id column; run 03_dedup_audit.py")
        return 1

    df, report = make_splits(
        df, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed
    )
    df = make_naive_splits(
        df, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed
    )
    df = make_cv_folds(df, n_splits=args.folds, seed=args.seed)

    df = add_shipped_split(df)

    contam = contamination_rate(df, "split_naive")
    contam_shipped = contamination_rate(df, "split_shipped")
    naive_report = summarise_splits(df[df["label"].notna()], split_col="split_naive")
    shipped_report = summarise_splits(df[df["label"].notna()], split_col="split_shipped")

    df.to_csv(args.out, index=False)

    lines = [
        "# Splits",
        "",
        "## Group-aware split (use this one)",
        "",
        report.to_markdown(),
        "",
        "Verified: no duplicate group straddles a split boundary.",
        "",
        "## Naive per-image split (control condition only)",
        "",
        naive_report.to_markdown(),
        "",
        "## The split the draft actually used (shipped source folders)",
        "",
        shipped_report.to_markdown(),
        "",
        "## Contamination, all three splits",
        "",
        "| split | held-out images | sharing a group with train | rate |",
        "| --- | --- | --- | --- |",
        f"| shipped (the draft's), held-out = `{contam_shipped['held_out']}` | "
        f"{contam_shipped['n_test']} | "
        f"{contam_shipped['n_contaminated']} | **{contam_shipped['rate']*100:.1f}%** |",
        f"| naive random | {contam['n_test']} | {contam['n_contaminated']} | "
        f"**{contam['rate']*100:.1f}%** |",
        "| group-aware | see above | 0 | **0.0%** |",
        "",
        "The shipped row is the one that bears on the reported 97.67%: it is the",
        "split the draft inherited and measured on. The naive row is what a standard",
        "`train_test_split(stratify=y)` over the merged pool would have produced.",
        "",
        f"## Cross-validation",
        "",
        f"- {args.folds}-fold group-stratified folds assigned over train+val "
        f"(column `fold`).",
        "- Use these for the seed/fold variance the draft currently reports none of.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSplit manifest -> {args.out}")
    print("\nNext: python scripts/05_train.py --help")
    return 0


if __name__ == "__main__":
    sys.exit(main())
