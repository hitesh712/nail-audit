"""Manifest construction over the raw image pool.

The two public sources ship different folder taxonomies, so nothing here assumes
the paper's six class names exist on disk. We discover the folders that are
actually present, map them through an explicit alias table, and loudly report
whatever failed to map instead of silently dropping it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from . import CLASSES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Folder name (normalised) -> canonical class. Extend this as new sources land;
# `build_manifest` reports every folder that misses the table.
ALIASES: dict[str, str] = {
    # Acral lentiginous melanoma
    "acral lentiginous melanoma": "Acral_Lentiginous_Melanoma",
    "acral melanoma": "Acral_Lentiginous_Melanoma",
    "melanoma": "Acral_Lentiginous_Melanoma",
    "subungual melanoma": "Acral_Lentiginous_Melanoma",
    "alm": "Acral_Lentiginous_Melanoma",
    # Blue finger
    "blue finger": "Blue_Finger",
    "bluish nail": "Blue_Finger",
    "blue nail": "Blue_Finger",
    "cyanosis": "Blue_Finger",
    # Clubbing
    "clubbing": "Clubbing",
    "nail clubbing": "Clubbing",
    "clubbed nail": "Clubbing",
    # Healthy
    "healthy nail": "Healthy_Nail",
    "healthy": "Healthy_Nail",
    "normal": "Healthy_Nail",
    "normal nail": "Healthy_Nail",
    "nail normal": "Healthy_Nail",
    # Onychogryphosis - a gross thickening deformity ("ram's horn nail")
    "onychogryphosis": "Onychogryphosis",
    "onychogryposis": "Onychogryphosis",
    "ram s horn nail": "Onychogryphosis",
    "rams horn nail": "Onychogryphosis",
    # Onychomycosis - a fungal infection. A DIFFERENT disease, kept separate
    # here. The draft folds this folder into Onychogryphosis; see PAPER_MERGES.
    "onychomycosis": "Onychomycosis",
    "onychomycosis nail": "Onychomycosis",
    "fungal": "Onychomycosis",
    "fungal nail": "Onychomycosis",
    # Psoriasis - present in the rasanjana source, dropped by the draft
    "psoriasis": "Psoriasis",
    "nail psoriasis": "Psoriasis",
    # Pitting
    "pitting": "Pitting",
    "nail pitting": "Pitting",
    "pitted nail": "Pitting",
}


def normalise_folder(name: str) -> str:
    """Lowercase, strip punctuation/underscores, collapse whitespace."""
    s = name.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class ManifestStats:
    n_files: int
    n_mapped: int
    n_unmapped: int
    unmapped_folders: dict[str, int]
    unreadable: list[str]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _class_folder_for(path: Path, root: Path) -> str:
    """Class label folder = the last directory component under the source root.

    Handles both `root/<class>/img.jpg` and `root/train/<class>/img.jpg` layouts,
    since Kaggle sets vary in whether they ship a pre-made split.
    """
    return path.parent.name


def build_manifest(
    sources: dict[str, Path],
    compute_sha: bool = True,
) -> tuple[pd.DataFrame, ManifestStats]:
    """Walk each source root and build one row per image.

    `sources` maps a short source id (e.g. "gurav") to its extracted root dir.
    Rows whose folder does not map to a canonical class keep ``label = None``
    so downstream code can decide whether to drop or re-map them.
    """
    rows: list[dict] = []
    unmapped: dict[str, int] = {}
    unreadable: list[str] = []

    for source_id, root in sources.items():
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"source root does not exist: {root}")

        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            raw_class = _class_folder_for(path, root)
            key = normalise_folder(raw_class)
            label = ALIASES.get(key)
            if label is None:
                unmapped[raw_class] = unmapped.get(raw_class, 0) + 1

            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    width, height = im.size
                    mode = im.mode
            except Exception as exc:  # noqa: BLE001 - we want every failure recorded
                unreadable.append(f"{path}: {exc}")
                continue

            rows.append(
                {
                    "path": str(path.resolve()),
                    "rel_path": str(path.relative_to(root)),
                    "source": source_id,
                    "raw_class": raw_class,
                    "label": label,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path) if compute_sha else None,
                }
            )

    df = pd.DataFrame(rows)
    stats = ManifestStats(
        n_files=len(df),
        n_mapped=int(df["label"].notna().sum()) if len(df) else 0,
        n_unmapped=int(df["label"].isna().sum()) if len(df) else 0,
        unmapped_folders=dict(sorted(unmapped.items(), key=lambda kv: -kv[1])),
        unreadable=unreadable,
    )
    return df, stats


def apply_taxonomy(df: pd.DataFrame, taxonomy: str = "corrected") -> pd.DataFrame:
    """Remap labels to the requested class set.

    ``corrected`` keeps the eight classes actually present on disk.
    ``paper`` reproduces the draft: onychomycosis is folded into
    Onychogryphosis and psoriasis is dropped, which is what makes the reported
    4,867-image dataset reconstruct exactly.
    """
    from . import PAPER_MERGES, TAXONOMIES

    if taxonomy not in TAXONOMIES:
        raise ValueError(f"unknown taxonomy '{taxonomy}'; pick one of {list(TAXONOMIES)}")

    out = df.copy()
    out["label_true"] = out["label"]
    if taxonomy == "paper":
        for src, dest in PAPER_MERGES.items():
            mask = out["label"] == src
            out.loc[mask, "label"] = dest  # None drops the class
    return out


def summarise(df: pd.DataFrame, classes: list[str] | None = None) -> pd.DataFrame:
    """Per-class x per-source counts, for the dataset table in the paper."""
    mapped = df[df["label"].notna()]
    if mapped.empty:
        return pd.DataFrame()
    order = classes if classes is not None else sorted(mapped["label"].unique())
    table = (
        mapped.pivot_table(
            index="label", columns="source", values="path", aggfunc="count", fill_value=0
        )
        .reindex([c for c in order if c in set(mapped["label"])])
        .fillna(0)
        .astype(int)
    )
    table["total"] = table.sum(axis=1)
    return table


def shipped_split_of(rel_path: str) -> str:
    """Which split folder the source dataset shipped this image in."""
    for p in Path(rel_path).parts:
        low = p.lower()
        if low in {"train", "training"}:
            return "train"
        if low in {"val", "valid", "validation"}:
            return "val"
        if low in {"test", "testing"}:
            return "test"
    return "unspecified"


def add_shipped_split(df: pd.DataFrame, col: str = "split_shipped") -> pd.DataFrame:
    """Reproduce the draft's split exactly: the folders the sources arrived in.

    The draft's Table II is these folders verbatim, with rasanjana's ``test``
    folder used as validation. That is the split its 97.67% was measured on.
    """
    out = df.copy()
    mapped = out["rel_path"].map(shipped_split_of)
    out[col] = mapped.replace({"test": "val"})
    return out


def source_split_table(df: pd.DataFrame) -> pd.DataFrame:
    """Counts broken down by source AND the split folder each source shipped.

    The draft did not create a split - it inherited the folders the sources came
    with. This table is what shows that.
    """
    mapped = df[df["label"].notna()].copy()
    if mapped.empty:
        return pd.DataFrame()

    def shipped(rel: str) -> str:
        parts = Path(rel).parts
        for p in parts:
            low = p.lower()
            if low in {"train", "training"}:
                return "train"
            if low in {"val", "valid", "validation"}:
                return "val"
            if low in {"test", "testing"}:
                return "test"
        return "unspecified"

    mapped["shipped_split"] = mapped["rel_path"].map(shipped)
    return (
        mapped.pivot_table(
            index="label",
            columns=["source", "shipped_split"],
            values="path",
            aggfunc="count",
            fill_value=0,
        )
        .astype(int)
    )
