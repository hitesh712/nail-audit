"""Was either source distributed pre-augmented?

Plain pHash is blind to flips and 90-degree rotations. If a dataset was
augmented before publication, a standard dedup reports it as clean while
flipped copies of the same photograph sit on both sides of any split.

gurav is anomalously easy - 96.7% on 6 classes, against 85.3% on rasanjana's 3 -
and every one of its 3,835 images is exactly 416x416, which is what a
preprocessing pipeline produces. This checks whether that easiness is explained
by orientation-augmented duplicates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail.dedup import (  # noqa: E402
    UnionFind,
    compute_dihedral_phashes,
    compute_phashes,
    dihedral_phash_pairs,
    phash_pairs,
)
from nail.utils import df_to_markdown  # noqa: E402


def group_stats(n: int, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, int]:
    uf = UnionFind(n)
    for i, j in pairs:
        uf.union(i, j)
    g = uf.groups()
    return g, int(len(np.unique(g)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_grouped.csv")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "02b_dihedral.md")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "manifest_dihedral.csv")
    ap.add_argument("--max-hamming", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest).reset_index(drop=True)
    rows = []
    all_groups = np.full(len(df), -1, dtype=int)
    offset = 0

    for source, sub in df.groupby("source"):
        sub = sub.reset_index()
        paths = sub["path"].tolist()
        n = len(paths)
        print(f"\n=== {source}: {n} images ===")

        plain = compute_phashes(paths)
        p_pairs = phash_pairs(plain, max_hamming=args.max_hamming)
        _, n_plain = group_stats(n, p_pairs)

        dih = compute_dihedral_phashes(paths)
        d_pairs = dihedral_phash_pairs(dih, max_hamming=args.max_hamming)
        g_dih, n_dih = group_stats(n, d_pairs)

        all_groups[sub["index"].to_numpy()] = g_dih + offset
        offset += n_dih

        extra = len(set(d_pairs) - set(p_pairs))
        rows.append(
            {
                "source": source,
                "images": n,
                "groups_plain": n_plain,
                "groups_dihedral": n_dih,
                "redundant_plain": n - n_plain,
                "redundant_dihedral": n - n_dih,
                "pct_redundant_plain": round(100.0 * (n - n_plain) / n, 1),
                "pct_redundant_dihedral": round(100.0 * (n - n_dih) / n, 1),
                "extra_pairs_from_orientation": extra,
            }
        )
        print(f"  plain pHash:    {n_plain} groups ({n - n_plain} redundant)")
        print(f"  dihedral pHash: {n_dih} groups ({n - n_dih} redundant)")
        print(f"  extra pairs found by orientation matching: {extra}")

    out = pd.DataFrame(rows).set_index("source")
    df["group_id_dihedral"] = all_groups
    df.to_csv(args.out, index=False)

    lines = [
        "# Orientation-invariant duplicate audit",
        "",
        f"pHash Hamming <= {args.max_hamming}, matched under all 8 dihedral",
        "transforms (4 rotations x 2 mirrorings), computed per source.",
        "",
        df_to_markdown(out, index_name="source"),
        "",
        "## Reading this",
        "",
        "If `redundant_dihedral` greatly exceeds `redundant_plain` for a source,",
        "that source was distributed with orientation-augmented copies. Those",
        "copies are invisible to a standard deduplication and would sit on both",
        "sides of any split, including the group-aware one used elsewhere in this",
        "pipeline.",
        "",
        "Where that holds, every accuracy reported on that source - including this",
        "pipeline's own group-aware numbers - needs recomputing with",
        "`group_id_dihedral` as the split unit.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nReport -> {args.report}")
    print(f"Manifest -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
