"""Duplicate and near-duplicate detection across the merged image pool.

Three independent signals, unioned into connected components:

1. ``sha256``    - byte-identical files (the same image shipped twice).
2. perceptual    - pHash within a small Hamming radius, which catches re-encodes,
                   rescales, minor crops and the pre-augmented copies that public
                   nail sets are full of.
3. embedding     - ImageNet ResNet-50 penultimate features above a cosine
                   threshold, which catches the same lesion photographed or
                   cropped differently.

The resulting ``group_id`` is the unit that must not straddle a train/test
boundary. It plays exactly the role a patient id plays in mammography: split on
the group, not on the image, or the test set is contaminated by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- union-find


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> np.ndarray:
        roots = np.array([self.find(i) for i in range(len(self.parent))])
        _, group_id = np.unique(roots, return_inverse=True)
        return group_id


# ---------------------------------------------------------------- popcount

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between two arrays of packed uint8 hash bytes.

    ``a`` is (n, nbytes), ``b`` is (m, nbytes); returns (n, m) uint16.
    """
    xor = np.bitwise_xor(a[:, None, :], b[None, :, :])
    return _POPCOUNT8[xor].sum(axis=2).astype(np.uint16)


# ---------------------------------------------------------------- signals


def exact_pairs(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Pairs of row indices sharing a sha256."""
    pairs: list[tuple[int, int]] = []
    if "sha256" not in df.columns or df["sha256"].isna().all():
        return pairs
    for _, idx in df.groupby("sha256").indices.items():
        idx = np.asarray(idx)
        for j in range(1, len(idx)):
            pairs.append((int(idx[0]), int(idx[j])))
    return pairs


def compute_phashes(paths: list[str], hash_size: int = 8) -> np.ndarray:
    """Packed pHash bytes, shape (n, hash_size**2 // 8). NaN rows become all-zero."""
    import imagehash
    from PIL import Image
    from tqdm import tqdm

    nbytes = (hash_size * hash_size) // 8
    out = np.zeros((len(paths), nbytes), dtype=np.uint8)
    for i, p in enumerate(tqdm(paths, desc="phash", unit="img")):
        try:
            with Image.open(p) as im:
                h = imagehash.phash(im.convert("RGB"), hash_size=hash_size)
            out[i] = np.packbits(h.hash.flatten())
        except Exception:  # noqa: BLE001 - unreadable images stay all-zero
            continue
    return out


def phash_pairs(
    hashes: np.ndarray, max_hamming: int = 5, chunk: int = 512
) -> list[tuple[int, int]]:
    """Index pairs whose pHash Hamming distance is <= ``max_hamming``."""
    from tqdm import tqdm

    n = len(hashes)
    pairs: list[tuple[int, int]] = []
    for start in tqdm(range(0, n, chunk), desc="phash-pairs", unit="chunk"):
        stop = min(start + chunk, n)
        dist = _hamming_matrix(hashes[start:stop], hashes)
        rows, cols = np.where(dist <= max_hamming)
        for r, c in zip(rows, cols):
            i = start + int(r)
            j = int(c)
            if i < j:
                pairs.append((i, j))
    return pairs


def compute_dihedral_phashes(
    paths: list[str], hash_size: int = 8
) -> np.ndarray:
    """pHash under all 8 dihedral transforms, shape (n, 8, nbytes).

    Plain pHash is not invariant to flips or 90-degree rotations, so a dataset
    that was augmented with flips before publication looks entirely
    duplicate-free to it. Public medical image sets are frequently distributed
    pre-augmented, which makes this a real blind spot rather than a theoretical
    one: those copies would sit on both sides of a split undetected.
    """
    import imagehash
    from PIL import Image
    from tqdm import tqdm

    nbytes = (hash_size * hash_size) // 8
    out = np.zeros((len(paths), 8, nbytes), dtype=np.uint8)
    for i, p in enumerate(tqdm(paths, desc="dihedral-phash", unit="img")):
        try:
            with Image.open(p) as im:
                rgb = im.convert("RGB")
            variants = []
            for base in (rgb, rgb.transpose(Image.FLIP_LEFT_RIGHT)):
                for rot in (0, 90, 180, 270):
                    variants.append(base if rot == 0 else base.rotate(rot, expand=True))
            for k, v in enumerate(variants):
                h = imagehash.phash(v, hash_size=hash_size)
                out[i, k] = np.packbits(h.hash.flatten())
        except Exception:  # noqa: BLE001
            continue
    return out


def dihedral_phash_pairs(
    hashes: np.ndarray, max_hamming: int = 6, chunk: int = 256
) -> list[tuple[int, int]]:
    """Pairs matching under ANY of the 8 orientations.

    ``hashes`` is (n, 8, nbytes) from :func:`compute_dihedral_phashes`. Only one
    side needs to vary orientation, since the dihedral group is closed under
    composition.
    """
    from tqdm import tqdm

    n = len(hashes)
    identity = hashes[:, 0, :]
    pairs: list[tuple[int, int]] = []
    for start in tqdm(range(0, n, chunk), desc="dihedral-pairs", unit="chunk"):
        stop = min(start + chunk, n)
        best = None
        for k in range(8):
            d = _hamming_matrix(identity[start:stop], hashes[:, k, :])
            best = d if best is None else np.minimum(best, d)
        rows, cols = np.where(best <= max_hamming)
        for r, c in zip(rows, cols):
            i, j = start + int(r), int(c)
            if i < j:
                pairs.append((i, j))
    return pairs


def compute_embeddings(
    paths: list[str], batch_size: int = 64, device: str | None = None
) -> np.ndarray:
    """L2-normalised ImageNet ResNet-50 pool5 features, shape (n, 2048)."""
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.models import ResNet50_Weights, resnet50
    from tqdm import tqdm

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class _DS(Dataset):
        def __len__(self) -> int:
            return len(paths)

        def __getitem__(self, i: int):
            try:
                with Image.open(paths[i]) as im:
                    return tf(im.convert("RGB")), i
            except Exception:  # noqa: BLE001
                return torch.zeros(3, 224, 224), i

    loader = DataLoader(_DS(), batch_size=batch_size, num_workers=0, shuffle=False)
    feats = np.zeros((len(paths), 2048), dtype=np.float32)
    with torch.no_grad():
        for xb, idx in tqdm(loader, desc="embed", unit="batch"):
            out = model(xb.to(device)).cpu().numpy()
            feats[idx.numpy()] = out

    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return feats / norms


def embedding_pairs(
    feats: np.ndarray, threshold: float = 0.98, chunk: int = 512
) -> list[tuple[int, int]]:
    """Index pairs with cosine similarity >= ``threshold`` (features pre-normalised)."""
    from tqdm import tqdm

    n = len(feats)
    pairs: list[tuple[int, int]] = []
    for start in tqdm(range(0, n, chunk), desc="embed-pairs", unit="chunk"):
        stop = min(start + chunk, n)
        sims = feats[start:stop] @ feats.T
        rows, cols = np.where(sims >= threshold)
        for r, c in zip(rows, cols):
            i = start + int(r)
            j = int(c)
            if i < j:
                pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------- orchestration


@dataclass
class DedupReport:
    n_images: int
    n_groups: int
    n_exact_pairs: int
    n_phash_pairs: int
    n_embed_pairs: int
    n_images_in_multi_groups: int
    n_redundant: int
    cross_source_groups: int
    cross_label_groups: int
    cross_label_examples: list[dict] = field(default_factory=list)
    group_size_hist: dict[int, int] = field(default_factory=dict)

    def to_markdown(self) -> str:
        pct = 100.0 * self.n_redundant / max(self.n_images, 1)
        lines = [
            "## Deduplication report",
            "",
            f"- Images scanned: **{self.n_images}**",
            f"- Distinct groups after union: **{self.n_groups}**",
            f"- Redundant images (image count - group count): **{self.n_redundant}** "
            f"({pct:.1f}% of the pool)",
            f"- Images sitting in a group of size > 1: **{self.n_images_in_multi_groups}**",
            "",
            "### Pairs contributed by each signal (pre-union)",
            f"- sha256 exact: {self.n_exact_pairs}",
            f"- pHash near-duplicate: {self.n_phash_pairs}",
            f"- embedding near-duplicate: {self.n_embed_pairs}",
            "",
            "### Contamination indicators",
            f"- Groups spanning **more than one source dataset**: "
            f"**{self.cross_source_groups}**",
            f"- Groups spanning **more than one class label**: "
            f"**{self.cross_label_groups}**",
        ]
        if self.cross_label_examples:
            lines += [
                "",
                "Cross-label groups are label noise: the same image filed under two "
                "different diagnoses. Examples:",
                "",
            ]
            for ex in self.cross_label_examples[:10]:
                lines.append(f"- group {ex['group_id']}: {ex['labels']} ({ex['n']} images)")
        lines += ["", "### Group size distribution", ""]
        for size, count in sorted(self.group_size_hist.items()):
            lines.append(f"- size {size}: {count} groups")
        return "\n".join(lines)


def assign_groups(
    df: pd.DataFrame,
    use_phash: bool = True,
    use_embeddings: bool = True,
    max_hamming: int = 6,
    cosine_threshold: float = 0.98,
) -> tuple[pd.DataFrame, DedupReport]:
    """Attach a ``group_id`` to every row and report what the signals found.

    Defaults come from the sweep in ``scripts/00_smoke_test.py`` against a pool
    with known duplicate structure, not from guesswork:

    ==========================  =========  ===========  ==============
    setting                     impure     recovered    max group size
    ==========================  =========  ===========  ==============
    pHash <=5, cosine >=0.98    0.0%       76.7%        4
    pHash <=6, cosine >=0.98    1.4%       86.7%        6
    pHash <=8, cosine >=0.98    5.2%       96.7%        10
    ==========================  =========  ===========  ==============

    We take pHash <=6. The two errors are not symmetric: over-merging only makes
    the split more conservative (two distinct images are forced into the same
    partition), while under-merging puts a near-duplicate pair across the
    train/test boundary, which is the leakage this whole module exists to
    prevent. Recall is therefore worth more than purity here, up to the point
    where group sizes start to blow up.

    Re-run the sweep on the real pool and quote the chosen operating point in the
    paper - the elbow will not sit in exactly the same place on real photographs.
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)
    uf = UnionFind(n)

    ex = exact_pairs(df)
    for i, j in ex:
        uf.union(i, j)

    ph: list[tuple[int, int]] = []
    if use_phash:
        hashes = compute_phashes(df["path"].tolist())
        ph = phash_pairs(hashes, max_hamming=max_hamming)
        for i, j in ph:
            uf.union(i, j)

    em: list[tuple[int, int]] = []
    if use_embeddings:
        feats = compute_embeddings(df["path"].tolist())
        em = embedding_pairs(feats, threshold=cosine_threshold)
        for i, j in em:
            uf.union(i, j)

    df["group_id"] = uf.groups()

    sizes = df.groupby("group_id").size()
    hist: dict[int, int] = sizes.value_counts().sort_index().to_dict()
    hist = {int(k): int(v) for k, v in hist.items()}

    per_group = df.groupby("group_id").agg(
        n=("path", "size"),
        n_sources=("source", "nunique"),
        n_labels=("label", "nunique"),
    )
    cross_label = per_group[per_group["n_labels"] > 1]

    examples = []
    for gid in cross_label.index[:25]:
        labels = sorted(df.loc[df["group_id"] == gid, "label"].dropna().unique().tolist())
        examples.append(
            {"group_id": int(gid), "labels": labels, "n": int(cross_label.loc[gid, "n"])}
        )

    report = DedupReport(
        n_images=n,
        n_groups=int(df["group_id"].nunique()),
        n_exact_pairs=len(ex),
        n_phash_pairs=len(ph),
        n_embed_pairs=len(em),
        n_images_in_multi_groups=int(sizes[sizes > 1].sum()),
        n_redundant=n - int(df["group_id"].nunique()),
        cross_source_groups=int((per_group["n_sources"] > 1).sum()),
        cross_label_groups=int(len(cross_label)),
        cross_label_examples=examples,
        group_size_hist=hist,
    )
    return df, report


def threshold_sweep(
    df: pd.DataFrame,
    cosine_values: tuple[float, ...] = (0.93, 0.95, 0.97, 0.98, 0.99, 0.995),
    hamming_values: tuple[int, ...] = (2, 4, 5, 6, 8),
) -> pd.DataFrame:
    """How many groups survive at each threshold, for each signal independently.

    The operating point should be argued from this table, not guessed. Read it for
    the elbow: thresholds below it collapse unrelated images into one group (group
    count falls off a cliff), thresholds above it stop catching real re-encodes.
    Over-merging is the safer error - it only makes the split more conservative -
    but gross over-merging destroys the dataset, so the elbow matters.
    """
    n = len(df)
    rows: list[dict] = []

    hashes = compute_phashes(df["path"].tolist())
    for h in hamming_values:
        uf = UnionFind(n)
        for i, j in exact_pairs(df):
            uf.union(i, j)
        pairs = phash_pairs(hashes, max_hamming=h)
        for i, j in pairs:
            uf.union(i, j)
        g = uf.groups()
        rows.append(
            {
                "signal": "phash",
                "threshold": h,
                "n_pairs": len(pairs),
                "n_groups": int(len(np.unique(g))),
                "pct_redundant": 100.0 * (n - len(np.unique(g))) / max(n, 1),
            }
        )

    feats = compute_embeddings(df["path"].tolist())
    for c in cosine_values:
        uf = UnionFind(n)
        for i, j in exact_pairs(df):
            uf.union(i, j)
        pairs = embedding_pairs(feats, threshold=c)
        for i, j in pairs:
            uf.union(i, j)
        g = uf.groups()
        rows.append(
            {
                "signal": "embedding",
                "threshold": c,
                "n_pairs": len(pairs),
                "n_groups": int(len(np.unique(g))),
                "pct_redundant": 100.0 * (n - len(np.unique(g))) / max(n, 1),
            }
        )

    return pd.DataFrame(rows)


def collapse_to_representatives(df: pd.DataFrame) -> pd.DataFrame:
    """One row per group: the largest image, which is usually the least re-encoded."""
    order = df.sort_values(["group_id", "bytes"], ascending=[True, False])
    return order.groupby("group_id", as_index=False).first()


def audit_split_leakage(df: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
    """Groups that straddle a split boundary. Must be empty for the split to be valid."""
    per_group = df.groupby("group_id")[split_col].nunique()
    bad = per_group[per_group > 1].index.tolist()
    return df[df["group_id"].isin(bad)].sort_values(["group_id", split_col])
