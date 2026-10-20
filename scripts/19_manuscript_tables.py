"""Emit the manuscript's headline tables from the clean test subset.

Single source of truth for Tables 2 and 3. Transcribing numbers between reports by
hand is how a ResNet-50 row ends up in an EfficientNet table; this removes the
opportunity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail import PAPER_CLASSES  # noqa: E402
from nail.metrics import core_metrics, per_class_table  # noqa: E402

C2I = {c: i for i, c in enumerate(PAPER_CLASSES)}
PRETTY = {
    "Acral_Lentiginous_Melanoma": "Acral lentiginous melanoma",
    "Blue_Finger": "Blue finger",
    "Clubbing": "Clubbing",
    "Healthy_Nail": "Healthy nail",
    "Onychogryphosis": "Onychogryphosis",
    "Pitting": "Pitting",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_rotdedup.csv")
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "runs_headline")
    ap.add_argument("--pattern", default="single_efficientnet_b0_s*")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "27_manuscript_tables.md")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    test = df[(df["split"] == "test") & df["label"].notna()].reset_index(drop=True)

    # Drop any test item sharing a group with training. On the source-aware
    # manifest this removes nothing, because the partition was built over source
    # photographs and is clean by construction; the check stays so the script
    # cannot silently report a contaminated set if pointed at an older manifest.
    group_col = next(
        (c for c in ("source_stem", "group_id_rot", "group_id") if c in df.columns),
        None,
    )
    if group_col is None:
        keep = np.ones(len(test), dtype=bool)
    else:
        train_groups = set(df.loc[df["split"] == "train", group_col])
        keep = ~test[group_col].isin(train_groups).to_numpy()
    print(f"grouping column: {group_col}; "
          f"{int((~keep).sum())} contaminated test items excluded")

    dirs = sorted(d for d in args.runs.glob(args.pattern)
                  if (d / "test_predictions.npz").exists())
    metrics, per_class = [], []
    for d in dirs:
        z = np.load(d / "test_predictions.npz")
        y, probs = z["y"][keep], z["probs"][keep]
        pred = probs.argmax(1)
        metrics.append(core_metrics(y, pred, probs))
        per_class.append(per_class_table(y, pred, PAPER_CLASSES))

    n = int(keep.sum())
    lines = [
        "# Manuscript tables (clean test subset)",
        "",
        f"EfficientNet-B0, {len(dirs)} seeds, n = {n} "
        f"({int((~keep).sum())} rotation-contaminated images excluded).",
        "",
        "## Table 2 - Held-out test performance",
        "",
        "| Metric | Mean | SD |",
        "|---|---|---|",
    ]
    label = {"balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro-F1",
             "accuracy": "Accuracy", "macro_auc": "Macro AUC"}
    for key in ["balanced_accuracy", "macro_f1", "accuracy", "macro_auc"]:
        v = np.array([m[key] for m in metrics], dtype=float)
        lines.append(f"| {label[key]} | {v.mean():.4f} | {v.std(ddof=1):.4f} |")

    lines += [
        "",
        "## Table 3 - Per-class performance, mean over seeds",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    total = 0
    for cls in PAPER_CLASSES:
        p = np.mean([r[cls]["precision"] for r in per_class])
        rc = np.mean([r[cls]["recall"] for r in per_class])
        f1 = np.mean([r[cls]["f1"] for r in per_class])
        sup = int(per_class[0][cls]["support"])
        total += sup
        lines.append(f"| {PRETTY[cls]} | {p:.3f} | {rc:.3f} | {f1:.3f} | {sup} |")
    lines += [f"| **Total** | | | | **{total}** |", ""]

    ok = total == n
    lines.append(
        f"Support total {total} {'matches' if ok else 'DOES NOT MATCH'} "
        f"the reported n = {n}."
    )

    # Table 5 - selective prediction, averaged over seeds on the clean subset,
    # using the validation-fitted temperature as the deployed model would.
    from nail.metrics import softmax

    lines += [
        "",
        "## Table 5 - Coverage and selective accuracy (mean over seeds)",
        "",
        "| tau | Coverage | Selective accuracy |",
        "|---|---|---|",
    ]
    rows = {t: {"cov": [], "acc": []} for t in (0.5, 0.7, 0.9)}
    for d in dirs:
        z = np.load(d / "test_predictions.npz")
        y = z["y"][keep]
        probs = softmax(z["logits"][keep], float(z["temperature"]))
        conf, pred = probs.max(1), probs.argmax(1)
        correct = (pred == y)
        for t in rows:
            m = conf >= t
            rows[t]["cov"].append(float(m.mean()))
            rows[t]["acc"].append(float(correct[m].mean()) if m.sum() else np.nan)
    for t in (0.5, 0.7, 0.9):
        lines.append(
            f"| {t:.2f} | {np.mean(rows[t]['cov'])*100:.1f}% | "
            f"{np.nanmean(rows[t]['acc'])*100:.2f}% |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
