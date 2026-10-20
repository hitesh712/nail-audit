"""Duplicate / near-duplicate audit over the merged pool.

This is the experiment that decides whether the headline number in the current
draft survives. It produces two things the paper needs:

  * ``group_id`` per image, the unit that splitting must respect;
  * a contamination report - how many images are redundant, how many groups span
    both source datasets, and how many span *two different class labels* (which
    is outright label noise).

If the redundancy rate is material, "97.7% on a random split of the merged pool"
is not a measurement of generalisation, and the honest version of this paper is
the one that says so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail.dedup import (  # noqa: E402
    assign_groups,
    collapse_to_representatives,
    threshold_sweep,
)
from nail.utils import df_to_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "manifest_grouped.csv")
    ap.add_argument(
        "--dedup-out", type=Path, default=ROOT / "data" / "manifest_dedup.csv"
    )
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "02_dedup.md")
    ap.add_argument("--max-hamming", type=int, default=6)
    ap.add_argument("--cosine", type=float, default=0.98)
    ap.add_argument("--no-phash", action="store_true")
    ap.add_argument("--no-embeddings", action="store_true")
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="also sweep both thresholds and write the sensitivity table",
    )
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"Missing {args.manifest}. Run 02_build_manifest.py first.")
        return 1

    df = pd.read_csv(args.manifest)
    print(f"Loaded {len(df)} rows")

    grouped, report = assign_groups(
        df,
        use_phash=not args.no_phash,
        use_embeddings=not args.no_embeddings,
        max_hamming=args.max_hamming,
        cosine_threshold=args.cosine,
    )
    grouped.to_csv(args.out, index=False)

    reps = collapse_to_representatives(grouped)
    reps.to_csv(args.dedup_out, index=False)

    sweep_md = ""
    if args.sweep:
        print("\nSweeping thresholds...")
        sweep = threshold_sweep(df)
        sweep.to_csv(args.report.parent / "02_dedup_sweep.csv", index=False)
        sweep_md = "\n".join(
            [
                "## Threshold sensitivity",
                "",
                "Argue the operating point from this table rather than asserting it.",
                "Look for the elbow: below it, unrelated images collapse together and the",
                "group count falls off a cliff; above it, genuine re-encodes stop being",
                "caught. Over-merging is the safer error - it only makes the split more",
                "conservative - but gross over-merging destroys the dataset.",
                "",
                df_to_markdown(sweep.set_index("signal"), index_name="signal"),
                "",
            ]
        )

    body = [
        "# Duplicate and near-duplicate audit",
        "",
        f"Settings: pHash Hamming <= {args.max_hamming}, "
        f"embedding cosine >= {args.cosine}",
        "",
        report.to_markdown(),
        "",
        sweep_md,
        "## What this means for the split",
        "",
        "`group_id` is the indivisible unit for train/val/test assignment. A random",
        "per-image split places near-duplicates of the same photograph on both sides",
        "of the boundary, so the test set is partially memorised rather than held out.",
        "This is the same failure mode as splitting mammograms without respecting",
        "patient identity.",
        "",
        f"- Per-image pool: **{report.n_images}** images",
        f"- Group-collapsed pool: **{report.n_groups}** images "
        f"(one representative per group)",
        "",
        "Report both numbers in the paper's dataset section.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(body), encoding="utf-8")

    print()
    print(report.to_markdown())
    print(f"\nGrouped manifest    -> {args.out}")
    print(f"Deduplicated pool   -> {args.dedup_out}")
    print(f"Report              -> {args.report}")
    print("\nNext: python scripts/04_make_splits.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
