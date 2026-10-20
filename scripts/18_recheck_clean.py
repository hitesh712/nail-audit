"""Restate the headline metrics with rotation-contaminated test images removed.

The rotation-invariant audit found 7 test images sharing a duplicate group with
training data - pairs the dihedral grouping could not see. This measures the
result both ways so the effect is quantified rather than assumed negligible.

A sensitivity check is not a substitute for rebuilding the split; it establishes
whether rebuilding would change any conclusion.
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
from nail.metrics import bootstrap_metrics, core_metrics  # noqa: E402

C2I = {c: i for i, c in enumerate(PAPER_CLASSES)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_rotdedup.csv")
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "runs_headline")
    ap.add_argument("--pattern", default="single_efficientnet_b0_s*")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "25_clean_recheck.md")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    test = df[(df["split"] == "test") & df["label"].notna()].reset_index(drop=True)
    train_groups = set(df.loc[df["split"] == "train", "group_id_rot"])
    contaminated = test["group_id_rot"].isin(train_groups).to_numpy()
    n_bad = int(contaminated.sum())
    print(f"test n={len(test)}, rotation-contaminated={n_bad}")

    dirs = sorted(d for d in args.runs.glob(args.pattern)
                  if (d / "test_predictions.npz").exists())
    full, clean, dirty_correct = [], [], []
    for d in dirs:
        z = np.load(d / "test_predictions.npz")
        y, probs = z["y"], z["probs"]
        if len(y) != len(test):
            print(f"{d.name}: prediction length {len(y)} != test rows {len(test)}")
            return 1
        pred = probs.argmax(1)
        full.append(core_metrics(y, pred, probs))
        clean.append(core_metrics(y[~contaminated], pred[~contaminated],
                                  probs[~contaminated]))
        if n_bad:
            dirty_correct.append(float((pred[contaminated] == y[contaminated]).mean()))

    def agg(rows, key):
        v = np.array([r[key] for r in rows], dtype=float)
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)

    lines = [
        "# Sensitivity to rotation-group contamination",
        "",
        f"{len(dirs)} seeds. Test partition n={len(test)}, of which **{n_bad}** "
        f"({100*n_bad/len(test):.1f}%) share a rotation-invariant duplicate group "
        f"with training data.",
        "",
        "| metric | full test set | contaminated removed | change |",
        "| --- | --- | --- | --- |",
    ]
    for key in ["balanced_accuracy", "macro_f1", "accuracy"]:
        fm, fs = agg(full, key)
        cm, cs = agg(clean, key)
        lines.append(
            f"| {key} | {fm:.4f} ± {fs:.4f} | {cm:.4f} ± {cs:.4f} | "
            f"**{(cm-fm)*100:+.2f} pp** |"
        )

    if n_bad:
        lines += [
            "",
            f"Accuracy on the {n_bad} contaminated images alone: "
            f"{np.mean(dirty_correct)*100:.1f}%.",
        ]

    fm, _ = agg(full, "accuracy")
    cm, _ = agg(clean, "accuracy")
    delta = abs(cm - fm) * 100
    if delta < 0.3:
        verdict = (
            f"Removing the contaminated images changes accuracy by {delta:.2f} "
            f"percentage points. **No conclusion in this paper depends on them.** "
            f"We report the clean figure and disclose the contamination rate; "
            f"rebuilding the split would not alter any finding."
        )
    else:
        verdict = (
            f"Removing the contaminated images changes accuracy by {delta:.2f} "
            f"percentage points, which is material. **The split must be rebuilt "
            f"over the rotation-invariant grouping and the models retrained.**"
        )
    lines += ["", "## Verdict", "", verdict, "",
              "The reportable number is the one measured with contaminated images",
              "excluded, with the contamination rate stated alongside it."]

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
