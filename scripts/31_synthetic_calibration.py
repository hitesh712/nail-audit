"""Threshold calibration on a synthetic pool with known duplicate structure.

The real collection cannot calibrate a similarity threshold, because there the
duplicate structure is what we are trying to recover. So the operating points are
argued from a pool whose ground truth is known by construction: distinct originals
generated independently, each with near-duplicate variants derived from it.

Two errors matter and they are not symmetric. Over-merging (a predicted group that
spans more than one original) only makes a partition more conservative. Under-merging
is the contamination an audit exists to prevent. This sweep measures both so the
manuscript can state where each signal stops being trustworthy.

    python scripts/31_synthetic_calibration.py

Writes results/44_synthetic_calibration.md and results/44_synthetic_calibration.csv.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nail.data import build_manifest  # noqa: E402
from nail.dedup import (  # noqa: E402
    UnionFind,
    compute_embeddings,
    compute_phashes,
    embedding_pairs,
    phash_pairs,
)

COSINES = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]
HAMMINGS = [2, 4, 5, 6, 8, 10, 12]


def score(truth: np.ndarray, group_id: np.ndarray) -> dict[str, float]:
    """Purity and recall of a predicted grouping against known originals.

    ``impure`` counts predicted groups spanning more than one true original, which is
    over-merging. ``recall`` is the share of true families that landed entirely in a
    single predicted group, which is what prevents leakage.
    """
    df = pd.DataFrame({"truth": truth, "pred": group_id})
    per_pred = df.groupby("pred")["truth"].nunique()
    per_truth = df.groupby("truth")["pred"].nunique()
    return {
        "n_groups": int(per_pred.size),
        "n_impure": int((per_pred > 1).sum()),
        "pct_impure": float((per_pred > 1).mean() * 100),
        "family_recall_pct": float((per_truth == 1).mean() * 100),
    }


def group_from_pairs(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    uf = UnionFind(n)
    for i, j in pairs:
        uf.union(i, j)
    return uf.groups()


def main() -> int:
    ap = argparse.ArgumentParser()
    # 60 is the pool the self-test builds and the figure the manuscript quotes.
    # Larger pools are a robustness check, written to their own file.
    ap.add_argument("--n-base", type=int, default=60,
                    help="distinct originals in the synthetic pool")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    # Imported late: 00_smoke_test is the authority on how the pool is built, and
    # reusing it keeps the calibration pool identical to the one the self-test uses.
    smoke = __import__("00_smoke_test")

    tmp = Path(tempfile.mkdtemp(prefix="nail_calib_"))
    try:
        print(f"building synthetic pool (n_base={args.n_base}) ...")
        truth_map = smoke.build_fake_pool(tmp, n_base=args.n_base)
        sources = {p.name: p for p in sorted(tmp.iterdir()) if p.is_dir()}
        df, stats = build_manifest(sources)
        print(f"{stats.n_files} images from {args.n_base} originals")

        truth = df["path"].map({k: v for k, v in truth_map.items()}).to_numpy()
        if pd.isna(truth).any():
            print("could not map every image to a known original")
            return 1

        paths = df["path"].tolist()
        n = len(paths)

        print("computing perceptual hashes ...")
        hashes = compute_phashes(paths)
        print("computing embeddings ...")
        feats = compute_embeddings(paths)

        rows = []
        for h in HAMMINGS:
            pairs = phash_pairs(hashes, max_hamming=h)
            rows.append({"signal": "phash", "threshold": float(h),
                         "n_pairs": len(pairs),
                         **score(truth, group_from_pairs(n, pairs))})
        for c in COSINES:
            pairs = embedding_pairs(feats, threshold=c)
            rows.append({"signal": "embedding", "threshold": c,
                         "n_pairs": len(pairs),
                         **score(truth, group_from_pairs(n, pairs))})

        out = pd.DataFrame(rows)
        stem = "44_synthetic_calibration" if args.n_base == 60 \
            else f"44_synthetic_calibration_n{args.n_base}"
        csv = ROOT / "results" / f"{stem}.csv"
        out.to_csv(csv, index=False)

        lines = [
            "# Threshold calibration on a synthetic pool",
            "",
            f"Pool: **{n} images** generated from **{args.n_base} distinct originals**. "
            "Each original contributes a downscale-and-restore copy, a brightness-"
            "perturbed copy, and for every fifth original a byte-identical copy, placed "
            "across two nominal sources. The duplicate structure is therefore known "
            "rather than inferred.",
            "",
            "`impure` counts predicted groups that span more than one original, which is "
            "over-merging. `family recall` is the share of originals whose copies all "
            "landed in one predicted group, which is what prevents leakage. Each signal "
            "is evaluated alone, not unioned.",
            "",
            "| signal | threshold | pairs | groups | impure groups | % impure | family recall % |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            thr = f"{r['threshold']:.3f}" if r["signal"] == "embedding" else f"{int(r['threshold'])}"
            lines.append(
                f"| {r['signal']} | {thr} | {r['n_pairs']} | {r['n_groups']} | "
                f"{r['n_impure']} | {r['pct_impure']:.1f} | {r['family_recall_pct']:.1f} |"
            )

        emb = out[out.signal == "embedding"]
        clean = emb[emb.pct_impure == 0]["threshold"]
        lines += [
            "",
            "## Reading this",
            "",
            f"The embedding signal merges no distinct originals at cosine "
            f"{clean.min():.2f} and above."
            if len(clean) else
            "The embedding signal merged distinct originals at every cosine tested.",
            "",
            "The operating point used on the real collection (perceptual Hamming <= 6, "
            "cosine >= 0.98) is chosen from this table, favouring recall over precision.",
        ]
        md = ROOT / "results" / f"{stem}.md"
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")

        print()
        print(out.to_string(index=False))
        print(f"\nwrote {md}")
        print(f"wrote {csv}")
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"kept pool at {tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
