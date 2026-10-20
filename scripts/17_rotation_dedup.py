"""Rotation-invariant duplicate audit.

The distributed images are augmented derivatives: 18.9% of pixels per image are
near-black on average, the fill left by rotating without cropping. Our earlier
audit matched under the eight dihedral transforms, which covers 90-degree steps
only. An image rotated by 23 degrees is invisible to it, and to embedding
similarity, and would sit on both sides of a split undetected.

This matches every image against every other under rotations in 15-degree steps,
hashing a centre crop so that the black fill introduced by rotation does not itself
dominate the hash.

If this finds substantially more redundancy than the dihedral audit, the effective
sample size is smaller than the image count and every accuracy reported on this
data - ours included - needs restating over the corrected groups.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail.dedup import UnionFind, _hamming_matrix  # noqa: E402
from nail.utils import df_to_markdown  # noqa: E402


def rotation_hashes(paths: list[str], n_rot: int = 24, crop: float = 0.6,
                    hash_size: int = 8) -> np.ndarray:
    """pHash of a centre crop at `n_rot` evenly spaced rotations."""
    import imagehash

    nbytes = (hash_size * hash_size) // 8
    out = np.zeros((len(paths), n_rot, nbytes), dtype=np.uint8)
    angles = [i * 360.0 / n_rot for i in range(n_rot)]
    for i, p in enumerate(tqdm(paths, desc="rot-hash", unit="img")):
        try:
            with Image.open(p) as im:
                rgb = im.convert("RGB")
            w, h = rgb.size
            s = int(min(w, h) * crop)
            box = ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
            for k, a in enumerate(angles):
                v = rgb if a == 0 else rgb.rotate(a, resample=Image.BILINEAR)
                out[i, k] = np.packbits(
                    imagehash.phash(v.crop(box), hash_size=hash_size).hash.flatten()
                )
        except Exception:  # noqa: BLE001
            continue
    return out


def rotation_pairs(hashes: np.ndarray, max_hamming: int = 6,
                   chunk: int = 128) -> list[tuple[int, int]]:
    n = len(hashes)
    identity = hashes[:, 0, :]
    n_rot = hashes.shape[1]
    pairs: list[tuple[int, int]] = []
    for start in tqdm(range(0, n, chunk), desc="rot-pairs", unit="chunk"):
        stop = min(start + chunk, n)
        best = None
        for k in range(n_rot):
            d = _hamming_matrix(identity[start:stop], hashes[:, k, :])
            best = d if best is None else np.minimum(best, d)
        rows, cols = np.where(best <= max_hamming)
        for r, c in zip(rows, cols):
            i, j = start + int(r), int(c)
            if i < j:
                pairs.append((i, j))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_final.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "manifest_rotdedup.csv")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "24_rotation_dedup.md")
    ap.add_argument("--max-hamming", type=int, default=6)
    ap.add_argument("--n-rot", type=int, default=24)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest).reset_index(drop=True)
    paths = df["path"].tolist()
    n = len(paths)
    print(f"{n} images, matching under {args.n_rot} rotations")

    h = rotation_hashes(paths, n_rot=args.n_rot)
    pairs = rotation_pairs(h, max_hamming=args.max_hamming)

    uf = UnionFind(n)
    for i, j in pairs:
        uf.union(i, j)
    g = uf.groups()
    df["group_id_rot"] = g
    n_groups = int(len(np.unique(g)))

    prev_groups = int(df["group_id"].nunique())
    sizes = pd.Series(g).value_counts()
    hist = sizes.value_counts().sort_index()

    # Does the existing split survive the stricter grouping?
    straddle = df.groupby("group_id_rot")["split"].nunique()
    n_straddle = int((straddle > 1).sum())
    train_groups = set(df.loc[df["split"] == "train", "group_id_rot"])
    test_rows = df[df["split"] == "test"]
    contaminated = int(test_rows["group_id_rot"].isin(train_groups).sum())

    df.to_csv(args.out, index=False)

    lines = [
        "# Rotation-invariant duplicate audit",
        "",
        f"{n} images matched under {args.n_rot} rotations of a centre crop, "
        f"pHash Hamming <= {args.max_hamming}.",
        "",
        "| grouping | groups | redundant | % |",
        "| --- | --- | --- | --- |",
        f"| dihedral (previous) | {prev_groups} | {n - prev_groups} | "
        f"{100*(n-prev_groups)/n:.1f}% |",
        f"| **rotation-invariant** | **{n_groups}** | **{n - n_groups}** | "
        f"**{100*(n-n_groups)/n:.1f}%** |",
        "",
        "## Effect on the existing split",
        "",
        f"- Rotation groups straddling a split boundary: **{n_straddle}**",
        f"- Test images sharing a rotation group with training data: "
        f"**{contaminated} / {len(test_rows)} "
        f"({100*contaminated/max(len(test_rows),1):.1f}%)**",
        "",
        "## Group size distribution",
        "",
        df_to_markdown(
            pd.DataFrame({"groups": hist}).rename_axis("group size"),
            index_name="group size",
        ),
    ]

    extra = (n - n_groups) - (n - prev_groups)
    if contaminated > 0:
        lines += [
            "",
            "## Verdict",
            "",
            f"**The split is contaminated under the stricter grouping.** "
            f"{contaminated} test images are rotated copies of training images that "
            f"the dihedral audit could not see. Splits must be rebuilt over "
            f"`group_id_rot` and every accuracy restated.",
        ]
    elif extra > 0:
        lines += [
            "",
            "## Verdict",
            "",
            f"Rotation-invariant matching finds **{extra} additional redundant "
            f"images** the dihedral audit missed, but the existing split already "
            f"separates them: **zero** test images share a rotation group with "
            f"training data. The reported results stand. The higher redundancy "
            f"should still be disclosed, since it means the effective sample size "
            f"is below the image count.",
        ]
    else:
        lines += [
            "",
            "## Verdict",
            "",
            "Rotation-invariant matching finds no redundancy beyond the dihedral "
            "audit. The augmentation visible in the images does not produce "
            "detectable near-duplicate pairs within this collection.",
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
