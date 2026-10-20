"""Self-test on synthetic data with known duplicates.

Builds a fake pool where the ground-truth duplicate structure is known, then
checks that the dedup finds it, that the group-aware split never straddles a
group, and that the naive split demonstrably leaks. Run this before touching
real data - if it fails, the audit on real data cannot be trusted.

    python scripts/00_smoke_test.py
    python scripts/00_smoke_test.py --with-embeddings   # also exercises the GPU path
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail import CLASSES  # noqa: E402
from nail.data import build_manifest  # noqa: E402
from nail.dedup import UnionFind, assign_groups  # noqa: E402
from nail.metrics import (  # noqa: E402
    bootstrap_metrics,
    expected_calibration_error,
    risk_coverage_curve,
)
from nail.splits import (  # noqa: E402
    LeakageError,
    contamination_rate,
    make_naive_splits,
    make_splits,
)

PASS, FAIL = "[ ok ]", "[FAIL]"
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{PASS if condition else FAIL} {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        _failures.append(name)


def make_base_image(seed: int, size: int = 128) -> Image.Image:
    """A deterministic image with real structural variety.

    Distinct originals must be distinct in *feature* space, not just pixel space,
    or the embedding signal has nothing to separate. A pool of near-identical
    coloured blobs would make this test measure the generator, not the audit.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    nx, ny = xx / size, yy / size
    motif = seed % 6

    if motif == 0:  # off-centre blob
        cx, cy = rng.uniform(0.2, 0.8, 2)
        base = np.exp(-(((nx - cx) ** 2 + (ny - cy) ** 2) / rng.uniform(0.01, 0.06)))
    elif motif == 1:  # diagonal stripes
        freq, ang = rng.uniform(6, 18), rng.uniform(0, np.pi)
        base = 0.5 + 0.5 * np.sin(freq * (nx * np.cos(ang) + ny * np.sin(ang)) * np.pi)
    elif motif == 2:  # checkerboard
        n = int(rng.integers(4, 12))
        base = ((np.floor(nx * n) + np.floor(ny * n)) % 2).astype(np.float32)
    elif motif == 3:  # concentric rings
        cx, cy = rng.uniform(0.3, 0.7, 2)
        r = np.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
        base = 0.5 + 0.5 * np.sin(r * rng.uniform(20, 45))
    elif motif == 4:  # linear gradient with a bar
        base = nx * rng.uniform(0.5, 1.0) + ny * rng.uniform(0.0, 0.5)
        y0 = rng.uniform(0.2, 0.7)
        base[(ny > y0) & (ny < y0 + 0.15)] = 1.0
    else:  # radial wedges
        cx, cy = rng.uniform(0.35, 0.65, 2)
        theta = np.arctan2(ny - cy, nx - cx)
        base = 0.5 + 0.5 * np.sin(theta * int(rng.integers(3, 9)))

    base = (base - base.min()) / max(float(np.ptp(base)), 1e-6)

    # pHash is a low-frequency DCT signature, so periodic motifs (checkerboards,
    # stripes) collide across distinct images by construction. Real photographs
    # carry a unique low-frequency envelope; give each base image one too, or the
    # test measures pHash's known behaviour on textures rather than the audit.
    a, b, c = rng.uniform(-1.0, 1.0, 3)
    envelope = a * nx + b * ny + c * nx * ny
    envelope = (envelope - envelope.min()) / max(float(np.ptp(envelope)), 1e-6)
    base = 0.55 * base + 0.45 * envelope

    palette = rng.uniform(40, 255, 3)
    tint = rng.uniform(0, 70, 3)
    arr = np.stack([base * palette[c] + tint[c] for c in range(3)], axis=-1)
    arr += rng.normal(0, 3, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def build_fake_pool(root: Path, n_base: int = 60, n_dupes: int = 2) -> dict[str, int]:
    """Two 'sources' sharing images, plus re-encoded/rescaled near-duplicates."""
    truth: dict[str, int] = {}
    rng = np.random.default_rng(0)

    for i in range(n_base):
        cls = CLASSES[i % len(CLASSES)]
        img = make_base_image(i)

        # Original goes to source A.
        d = root / "srcA" / cls
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"img{i:03d}.png"
        img.save(p)
        truth[str(p.resolve())] = i

        # Near-duplicates: some in source A, some in source B (cross-source overlap).
        for k in range(n_dupes):
            variant = img.copy()
            if k == 0:
                variant = variant.resize((80, 80)).resize((96, 96))
            else:
                variant = ImageEnhance.Brightness(variant).enhance(1.04)
            src = "srcA" if k == 0 else "srcB"
            d2 = root / src / cls
            d2.mkdir(parents=True, exist_ok=True)
            p2 = d2 / f"img{i:03d}_v{k}.jpg"
            variant.save(p2, quality=92)
            truth[str(p2.resolve())] = i

        # A byte-identical copy in source B for a subset.
        if i % 5 == 0:
            d3 = root / "srcB" / cls
            d3.mkdir(parents=True, exist_ok=True)
            p3 = d3 / f"img{i:03d}_exact.png"
            shutil.copy2(p, p3)
            truth[str(p3.resolve())] = i

    return truth


def test_unionfind() -> None:
    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(4, 5)
    g = uf.groups()
    check("union-find merges transitively", g[0] == g[1] == g[2])
    check("union-find keeps singletons apart", g[3] != g[0] and g[4] == g[5])
    check("union-find group count", len(set(g.tolist())) == 3, f"got {len(set(g.tolist()))}")


def test_metrics() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    probs = rng.dirichlet(np.ones(3), 200)
    probs[np.arange(200), y] += 1.5
    probs /= probs.sum(1, keepdims=True)
    pred = probs.argmax(1)

    ci = bootstrap_metrics(y, pred, probs, n_boot=200)
    ok = all(ci[k].lo <= ci[k].point <= ci[k].hi for k in ["accuracy", "macro_f1"])
    check("bootstrap CI brackets the point estimate", ok)

    ece = expected_calibration_error(y, probs)
    check("ECE in [0,1]", 0.0 <= ece <= 1.0, f"{ece:.4f}")

    rc = risk_coverage_curve(y, probs)
    check("coverage reaches 1.0", np.isclose(rc.coverage[-1], 1.0))
    check(
        "selective accuracy at high confidence >= overall",
        rc.at_threshold[0.9]["selective_accuracy"] >= (pred == y).mean() - 1e-9
        or np.isnan(rc.at_threshold[0.9]["selective_accuracy"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-embeddings", action="store_true")
    ap.add_argument("--no-phash", action="store_true",
                    help="sha256 only; lets the split/metric logic be checked "
                         "without imagehash installed")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print("--- unit checks ---")
    test_unionfind()
    test_metrics()

    tmp = Path(tempfile.mkdtemp(prefix="nail_smoke_"))
    try:
        print("\n--- building synthetic pool ---")
        truth = build_fake_pool(tmp)
        print(f"wrote {len(truth)} images to {tmp}")

        sources = {p.name: p for p in sorted(tmp.iterdir()) if p.is_dir()}
        df, stats = build_manifest(sources)
        check("manifest picked up every image", stats.n_files == len(truth),
              f"{stats.n_files} vs {len(truth)}")
        check("all folders mapped to canonical classes", stats.n_unmapped == 0)
        check("no unreadable files", len(stats.unreadable) == 0)

        print("\n--- dedup ---")
        use_phash = not args.no_phash
        grouped, report = assign_groups(
            df, use_phash=use_phash, use_embeddings=args.with_embeddings
        )
        print(report.to_markdown())

        grouped["true_group"] = grouped["path"].map(truth)
        purity = grouped.groupby("group_id")["true_group"].nunique()
        pct_impure = float((purity > 1).mean())
        sizes = grouped.groupby("group_id").size()

        recall = grouped.groupby("true_group")["group_id"].nunique()
        found = float((recall == 1).mean())
        if use_phash:
            # Over-merging is the safe error: it only makes the split more
            # conservative. Under-merging is what leaks. So we require high recall
            # and merely *bounded* over-merging, rather than perfect purity.
            check("over-merging stays bounded (<=5% of groups)", pct_impure <= 0.05,
                  f"{pct_impure*100:.1f}% impure")
            check("no runaway groups (max size <= 8)", int(sizes.max()) <= 8,
                  f"max size {int(sizes.max())}")
            check("duplicate families recovered (>=85%)", found >= 0.85, f"{found*100:.0f}%")
            check("cross-source overlap detected", report.cross_source_groups > 0,
                  f"{report.cross_source_groups} groups")
        else:
            check("no predicted group merges distinct originals", (purity == 1).all(),
                  f"{int((purity > 1).sum())} impure groups")
            # sha256 alone can only catch the byte-identical copies (every 5th image).
            check("exact duplicates grouped", report.n_exact_pairs > 0,
                  f"{report.n_exact_pairs} pairs")
            check("sha-only leaves near-duplicates ungrouped (expected)", found < 0.8,
                  f"{found*100:.0f}% - phash is what closes this gap")

        print("\n--- splits ---")
        split_df, split_report = make_splits(grouped, seed=0)
        print(split_report.to_markdown())

        per_group = split_df[split_df["label"].notna()].groupby("group_id")["split"].nunique()
        check("group-aware split: no group straddles a boundary", (per_group == 1).all())
        check("all three partitions non-empty",
              set(split_df.loc[split_df["label"].notna(), "split"]) == {"train", "val", "test"})

        split_df = make_naive_splits(split_df, seed=0)
        contam = contamination_rate(split_df, "split_naive")
        print(f"\nnaive-split contamination: {contam['n_contaminated']}/{contam['n_test']} "
              f"= {contam['rate']*100:.1f}%")
        min_leak = 0.2 if use_phash else 0.02
        check("naive split demonstrably leaks", contam["rate"] > min_leak,
              f"{contam['rate']*100:.1f}%")

        grouped_contam = contamination_rate(split_df, "split")
        check("group-aware split has zero contamination",
              grouped_contam["n_contaminated"] == 0,
              f"{grouped_contam['n_contaminated']} contaminated")

        print("\n--- leakage guard ---")
        broken = split_df.copy()
        multi = broken.groupby("group_id").filter(lambda g: len(g) > 1)
        if len(multi):
            gid = multi.iloc[0]["group_id"]
            rows = broken.index[broken["group_id"] == gid].tolist()
            broken.loc[rows[0], "split"] = "train"
            broken.loc[rows[1], "split"] = "test"
            try:
                from nail.splits import _assert_no_leakage
                _assert_no_leakage(broken[broken["label"].notna()])
                check("leakage guard raises on a straddling group", False)
            except LeakageError:
                check("leakage guard raises on a straddling group", True)
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"\nkept synthetic pool at {tmp}")

    print("\n" + "=" * 55)
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {_failures}")
        return 1
    print("All checks passed. The audit logic is sound; run it on real data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
