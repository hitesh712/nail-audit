"""Group-aware stratified splitting.

Every split here is made over ``group_id``, never over individual images. A
duplicate group is indivisible: all of its images land in the same partition or
none do. This is the same discipline as patient-wise splitting in mammography,
and it is what the reported split in the current draft is missing.

Class balance is preserved as closely as a group constraint allows, and every
function verifies its own output before returning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import CLASSES


class LeakageError(RuntimeError):
    """Raised when a group straddles a split boundary."""


def group_labels(df: pd.DataFrame) -> pd.Series:
    """One label per group (majority vote), so a group can be stratified on."""
    return (
        df.groupby("group_id")["label"]
        .agg(lambda s: s.value_counts().idxmax() if s.notna().any() else np.nan)
        .rename("group_label")
    )


def _assert_no_leakage(df: pd.DataFrame, split_col: str = "split") -> None:
    per_group = df.groupby("group_id")[split_col].nunique()
    bad = per_group[per_group > 1]
    if len(bad):
        raise LeakageError(
            f"{len(bad)} duplicate group(s) straddle a split boundary; "
            f"first offenders: {bad.index[:5].tolist()}"
        )


@dataclass
class SplitReport:
    counts: pd.DataFrame
    fractions: dict[str, float]
    n_groups: dict[str, int]

    def to_markdown(self) -> str:
        from .utils import df_to_markdown

        lines = ["## Split composition", "", "### Images per class per split", ""]
        lines.append(df_to_markdown(self.counts, index_name="class"))
        lines += ["", "### Split sizes", ""]
        for k in self.fractions:
            lines.append(
                f"- **{k}**: {int(self.counts[k].sum())} images "
                f"({self.fractions[k]*100:.1f}%), {self.n_groups[k]} groups"
            )
        # Flag any split whose class mix is badly skewed.
        lines += ["", "### Per-split class share (%)", ""]
        share = 100.0 * self.counts / self.counts.sum(axis=0)
        lines.append(df_to_markdown(share.round(1), index_name="class"))
        return "\n".join(lines)


def summarise_splits(df: pd.DataFrame, split_col: str = "split") -> SplitReport:
    counts = (
        df.pivot_table(
            index="label", columns=split_col, values="path", aggfunc="count", fill_value=0
        )
        .reindex([c for c in CLASSES if c in set(df["label"].dropna())])
        .fillna(0)
        .astype(int)
    )
    total = counts.values.sum()
    fractions = {c: counts[c].sum() / total for c in counts.columns}
    n_groups = {
        c: int(df.loc[df[split_col] == c, "group_id"].nunique()) for c in counts.columns
    }
    return SplitReport(counts=counts, fractions=fractions, n_groups=n_groups)


def make_splits(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, SplitReport]:
    """Assign train/val/test at the group level, stratified by class.

    Test is carved out first and is intended to be touched exactly once, at the
    very end. Validation is for early stopping and threshold selection only.
    """
    df = df.reset_index(drop=True).copy()
    labelled = df[df["label"].notna()].copy()
    if labelled.empty:
        raise ValueError("no labelled rows; check the alias table in data.py")

    glabels = group_labels(labelled)
    groups = glabels.index.to_numpy()
    y = glabels.to_numpy()

    # Carve out the test groups.
    n_splits_test = max(2, int(round(1.0 / test_frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(groups.reshape(-1, 1), y, groups=groups))
    test_groups = set(groups[test_idx])

    # Carve validation out of what remains.
    rest_mask = np.array([g not in test_groups for g in groups])
    rest_groups, rest_y = groups[rest_mask], y[rest_mask]
    rel_val = val_frac / max(1e-9, 1.0 - test_frac)
    n_splits_val = max(2, int(round(1.0 / rel_val)))
    sgkf2 = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf2.split(rest_groups.reshape(-1, 1), rest_y, groups=rest_groups))
    val_groups = set(rest_groups[val_idx])

    def assign(gid: int) -> str:
        if gid in test_groups:
            return "test"
        if gid in val_groups:
            return "val"
        return "train"

    df["split"] = df["group_id"].map(assign)
    df.loc[df["label"].isna(), "split"] = "unlabelled"

    labelled_out = df[df["label"].notna()]
    _assert_no_leakage(labelled_out)
    return df, summarise_splits(labelled_out)


def make_naive_splits(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    seed: int = 42,
    split_col: str = "split_naive",
) -> pd.DataFrame:
    """Per-image stratified split that ignores duplicate groups.

    This is the control condition, not a recommendation. It reproduces what a
    standard ``train_test_split(stratify=y)`` over the merged pool does, so the
    gap against :func:`make_splits` measures how much of the reported accuracy is
    duplicate leakage rather than generalisation.
    """
    from sklearn.model_selection import train_test_split

    df = df.reset_index(drop=True).copy()
    labelled = df[df["label"].notna()]
    idx = labelled.index.to_numpy()
    y = labelled["label"].to_numpy()

    tr_idx, test_idx = train_test_split(
        idx, test_size=test_frac, stratify=y, random_state=seed
    )
    rel_val = val_frac / max(1e-9, 1.0 - test_frac)
    tr_idx, val_idx = train_test_split(
        tr_idx,
        test_size=rel_val,
        stratify=labelled.loc[tr_idx, "label"].to_numpy(),
        random_state=seed,
    )

    df[split_col] = "unlabelled"
    df.loc[tr_idx, split_col] = "train"
    df.loc[val_idx, split_col] = "val"
    df.loc[test_idx, split_col] = "test"
    return df


def contamination_rate(
    df: pd.DataFrame,
    split_col: str = "split_naive",
    held_out: str | None = None,
    train_name: str = "train",
) -> dict[str, float]:
    """Fraction of held-out images sharing a duplicate group with a training image.

    This is the single number that quantifies the leakage: the share of the
    "held-out" set the model has effectively already seen. When ``held_out`` is
    None it defaults to ``test`` if that split exists, otherwise ``val`` - the
    shipped split from the Kaggle sources has no test partition at all.
    """
    sub = df[df["label"].notna()]
    present = set(sub[split_col].unique())
    if held_out is None:
        held_out = "test" if "test" in present else "val"

    train_groups = set(sub.loc[sub[split_col] == train_name, "group_id"])
    held_rows = sub[sub[split_col] == held_out]
    if held_rows.empty:
        return {
            "held_out": held_out,
            "n_test": 0,
            "n_contaminated": 0,
            "rate": float("nan"),
        }
    contaminated = held_rows["group_id"].isin(train_groups)
    return {
        "held_out": held_out,
        "n_test": int(len(held_rows)),
        "n_contaminated": int(contaminated.sum()),
        "rate": float(contaminated.mean()),
    }


def make_cv_folds(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 42, split_col: str = "fold"
) -> pd.DataFrame:
    """Assign a CV fold id per group over the train+val portion.

    Use this for the seed/fold variance the paper currently has none of. Rows
    outside train/val (i.e. test, unlabelled) get fold ``-1``.
    """
    df = df.reset_index(drop=True).copy()
    pool = df[(df["label"].notna()) & (df.get("split", "train").isin(["train", "val"]))]
    if pool.empty:
        raise ValueError("no train/val rows available for CV folds")

    glabels = group_labels(pool)
    groups = glabels.index.to_numpy()
    y = glabels.to_numpy()

    fold_of: dict[int, int] = {}
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, va) in enumerate(sgkf.split(groups.reshape(-1, 1), y, groups=groups)):
        for g in groups[va]:
            fold_of[int(g)] = fold

    df[split_col] = df["group_id"].map(lambda g: fold_of.get(int(g), -1)).astype(int)

    # A group must live in exactly one fold.
    check = df[df[split_col] >= 0].groupby("group_id")[split_col].nunique()
    if (check > 1).any():
        raise LeakageError("a group was assigned to multiple CV folds")
    return df
