"""Audit the draft's own split, in the draft's own taxonomy.

Reproduces Table II exactly (6 classes, 4567 train / 300 val), verifies the
reproduction against the published numbers, then measures how much of that
300-image validation set is a near-duplicate of something in training.

This is the number that bears directly on the reported 97.67%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail import PAPER_CLASSES  # noqa: E402
from nail.data import add_shipped_split, apply_taxonomy  # noqa: E402
from nail.splits import contamination_rate  # noqa: E402
from nail.utils import df_to_markdown  # noqa: E402

# Table II of the draft, transcribed.
PAPER_TABLE_II = {
    "Acral_Lentiginous_Melanoma": (735, 18),
    "Blue_Finger": (603, 9),
    "Clubbing": (767, 16),
    "Healthy_Nail": (571, 82),
    "Onychogryphosis": (1252, 159),
    "Pitting": (639, 16),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_grouped.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "manifest_paper_split.csv")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "03b_paper_split.md")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if "group_id" not in df.columns:
        print("manifest has no group_id; run 03_dedup_audit.py first")
        return 1

    # Group ids are taxonomy-independent, so the expensive dedup is reused.
    df = apply_taxonomy(df, "paper")
    df = df[df["label"].notna()]
    df = add_shipped_split(df)
    df.to_csv(args.out, index=False)

    counts = (
        df.pivot_table(
            index="label", columns="split_shipped", values="path",
            aggfunc="count", fill_value=0,
        )
        .reindex(PAPER_CLASSES)
        .fillna(0)
        .astype(int)
    )

    lines = ["# The draft's split, audited", "", "## Reproduction of Table II", ""]
    lines += [
        "| class | ours (train/val) | published (train/val) | match |",
        "| --- | --- | --- | --- |",
    ]
    all_match = True
    for cls in PAPER_CLASSES:
        tr = int(counts.loc[cls, "train"]) if "train" in counts.columns else 0
        va = int(counts.loc[cls, "val"]) if "val" in counts.columns else 0
        ptr, pva = PAPER_TABLE_II[cls]
        ok = (tr, va) == (ptr, pva)
        all_match &= ok
        lines.append(
            f"| {cls} | {tr} / {va} | {ptr} / {pva} | {'YES' if ok else 'NO'} |"
        )
    tot_tr = int(counts["train"].sum())
    tot_va = int(counts["val"].sum())
    tot_ok = (tot_tr, tot_va) == (4567, 300)
    all_match &= tot_ok
    lines.append(
        f"| **Total** | **{tot_tr} / {tot_va}** | **4567 / 300** | "
        f"**{'YES' if tot_ok else 'NO'}** |"
    )

    lines += [
        "",
        f"Exact reproduction: **{'YES' if all_match else 'NO'}**.",
        "",
        "The draft did not construct a split. Every number in Table II is the",
        "folder structure the two Kaggle datasets shipped with, concatenated.",
        "That is why the per-class validation share ranges from 1.5% to 12.6%",
        "while the text states twice that stratified sampling was used.",
        "",
    ]

    contam = contamination_rate(df, "split_shipped", held_out="val")
    lines += [
        "## Duplicate contamination of the reported validation set",
        "",
        f"- Validation images: **{contam['n_test']}**",
        f"- Sharing a duplicate group with a training image: "
        f"**{contam['n_contaminated']}**",
        f"- **Contamination rate: {contam['rate']*100:.1f}%**",
        "",
        "Those images are near-duplicates of training data. Performance measured",
        "on them is partly recall of memorised examples, not generalisation.",
        "",
    ]

    per_class = []
    train_groups = set(df.loc[df["split_shipped"] == "train", "group_id"])
    val = df[df["split_shipped"] == "val"]
    for cls in PAPER_CLASSES:
        rows = val[val["label"] == cls]
        if rows.empty:
            continue
        c = rows["group_id"].isin(train_groups)
        per_class.append(
            {
                "class": cls,
                "val_n": len(rows),
                "contaminated": int(c.sum()),
                "rate_pct": round(100.0 * float(c.mean()), 1),
            }
        )
    pc = pd.DataFrame(per_class).set_index("class")
    lines += ["### Per class", "", df_to_markdown(pc, index_name="class"), ""]

    # Label noise inside the paper's own taxonomy.
    cross = (
        df.groupby("group_id")["label"].nunique().pipe(lambda s: s[s > 1]).index.tolist()
    )
    lines += [
        "## Label noise",
        "",
        f"- Duplicate groups spanning more than one class: **{len(cross)}**",
        "",
        "Each is the same image filed under two different diagnoses.",
        "",
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
