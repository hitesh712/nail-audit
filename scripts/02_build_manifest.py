"""Build the image manifest and report the real class taxonomy on disk.

Prints every folder that did not map to one of the six canonical classes. Those
are decisions you have to make explicitly - extend ALIASES in src/nail/data.py,
or accept that the images are out of scope. Nothing is silently dropped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail import TAXONOMIES  # noqa: E402
from nail.data import (  # noqa: E402
    apply_taxonomy,
    build_manifest,
    source_split_table,
    summarise,
)
from nail.utils import df_to_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "manifest.csv")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "01_manifest.md")
    ap.add_argument("--no-sha", action="store_true", help="skip sha256 (faster, weaker dedup)")
    ap.add_argument(
        "--taxonomy",
        choices=list(TAXONOMIES),
        default="corrected",
        help="'corrected' keeps the 8 classes on disk; 'paper' reproduces the "
             "draft's 6 by folding onychomycosis into Onychogryphosis and "
             "dropping psoriasis",
    )
    args = ap.parse_args()

    sources = {p.name: p for p in sorted(args.raw.iterdir()) if p.is_dir()}
    if not sources:
        print(f"No source folders under {args.raw}. Run 01_fetch_data.py first.")
        return 1

    print(f"Sources: {', '.join(sources)}")
    df, stats = build_manifest(sources, compute_sha=not args.no_sha)

    shipped = source_split_table(df)
    df = apply_taxonomy(df, args.taxonomy)
    df = df[df["label"].notna() | df["label_true"].isna()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    table = summarise(df, TAXONOMIES[args.taxonomy])

    lines = [
        "# Manifest",
        "",
        f"- Taxonomy: **{args.taxonomy}** "
        f"({len(TAXONOMIES[args.taxonomy])} classes)",
        f"- Files found: **{stats.n_files}**",
        f"- Mapped to a canonical class: **{stats.n_mapped}**",
        f"- Unmapped: **{stats.n_unmapped}**",
        f"- Unreadable/corrupt: **{len(stats.unreadable)}**",
        "",
    ]
    if len(shipped):
        lines += [
            "## Splits the sources shipped with",
            "",
            "The draft did not construct a split. These are the folders the two",
            "Kaggle datasets arrived in, and the draft's Table II reproduces them",
            "exactly - which is why its per-class validation fractions range from",
            "1.5% to 12.6%.",
            "",
            df_to_markdown(shipped, index_name="class"),
            "",
        ]
    if len(table):
        lines += [
            "## Mapped images per class per source",
            "",
            df_to_markdown(table, index_name="class"),
            "",
        ]
    if stats.unmapped_folders:
        lines += [
            "## Unmapped folders (decide explicitly)",
            "",
            "These folder names are present on disk but are not one of the six target",
            "classes. Either extend `ALIASES` in `src/nail/data.py` or state in the paper",
            "that they were excluded and why.",
            "",
            "| folder | images |",
            "| --- | --- |",
        ]
        for name, count in stats.unmapped_folders.items():
            lines.append(f"| `{name}` | {count} |")
        lines.append("")
    if stats.unreadable:
        lines += ["## Unreadable files", ""]
        lines += [f"- `{u}`" for u in stats.unreadable[:50]]

    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\nManifest -> {args.out}")
    print(f"Report   -> {args.report}")
    print("\nNext: python scripts/03_dedup_audit.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
