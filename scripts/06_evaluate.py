"""Final evaluation on the held-out test split. Run this once, at the end.

Produces everything the results section needs and the draft is missing:
  * balanced accuracy / macro-F1 as the headline, with bootstrap 95% intervals
  * per-class precision/recall/F1 with support
  * calibration: ECE before and after temperature scaling (fitted on val)
  * a risk-coverage table, so the 0.7 confidence threshold is reported with its
    coverage instead of silently discarding hard cases
  * confusion matrix
  * TTA on/off comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail.evaluate import predict_split  # noqa: E402
from nail.metrics import (  # noqa: E402
    bootstrap_metrics,
    confusion,
    expected_calibration_error,
    fit_temperature,
    per_class_table,
    risk_coverage_curve,
    softmax,
)
from nail.train import TrainConfig  # noqa: E402


def cfg_from_run(run_dir: Path) -> tuple[TrainConfig, list[str]]:
    hist = json.loads((run_dir / "history.json").read_text())
    stored = dict(hist["config"])
    classes = stored.pop("classes", None)
    return TrainConfig(**stored), classes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="a results/runs/<tag> dir")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_split.csv")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    if not (args.run / "best.pt").exists():
        print(f"No checkpoint at {args.run / 'best.pt'}")
        return 1

    cfg, classes = cfg_from_run(args.run)
    manifest = pd.read_csv(args.manifest)
    manifest["split"] = manifest[args.split_col]
    if classes is None:
        from nail.train import classes_in

        classes = classes_in(manifest)

    out: list[str] = [f"# Test evaluation - `{args.run.name}`", ""]
    out += [
        f"- Model: `{cfg.model}`",
        f"- Split column: `{args.split_col}`, split: `{args.split}`",
        f"- Augmentation: {cfg.augment}, balancing: `{cfg.balancing}`, "
        f"layer4 unfrozen: {cfg.unfreeze_layer4}",
        "",
    ]

    results = {}
    for use_tta in (False, True):
        label = "with TTA" if use_tta else "no TTA"
        logits, y, rows = predict_split(
            manifest, args.run / "best.pt", cfg, split=args.split, use_tta=use_tta, classes=classes
        )
        probs = softmax(logits)
        pred = probs.argmax(1)

        ci = bootstrap_metrics(y, pred, probs, n_boot=args.n_boot)
        results[label] = {"logits": logits, "y": y, "probs": probs, "ci": ci}

        out += [f"## Headline metrics ({label})", ""]
        out += ["| metric | value [95% CI] |", "| --- | --- |"]
        for name in ["balanced_accuracy", "macro_f1", "accuracy", "weighted_f1", "macro_auc"]:
            if name in ci:
                out.append(f"| {name} | {ci[name]} |")
        out.append("")

    # Per-class detail on the no-TTA model.
    y = results["no TTA"]["y"]
    probs = results["no TTA"]["probs"]
    pred = probs.argmax(1)

    out += ["## Per-class performance (no TTA)", "",
            "| class | precision | recall | F1 | support |", "| --- | --- | --- | --- | --- |"]
    for cls, m in per_class_table(y, pred, classes).items():
        out.append(
            f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {m['support']} |"
        )
    out += ["", "Any class with support in single digits cannot carry a claim. "
                "State the support next to every per-class number in the paper.", ""]

    # Calibration, temperature fitted on validation.
    val_logits_path = args.run / "val_logits.npz"
    temp = 1.0
    if val_logits_path.exists():
        d = np.load(val_logits_path)
        temp = fit_temperature(d["logits"], d["y"])
    ece_raw = expected_calibration_error(y, probs)
    ece_cal = expected_calibration_error(y, softmax(results["no TTA"]["logits"], temp))

    out += [
        "## Calibration",
        "",
        f"- Temperature fitted on validation: **T = {temp:.3f}**",
        f"- ECE before scaling: **{ece_raw:.4f}**",
        f"- ECE after scaling: **{ece_cal:.4f}**",
        "",
        "The draft treats softmax outputs as confidence without ever checking "
        "whether they are calibrated. If T departs materially from 1.0, the raw "
        "confidences were mis-scaled and the 0.7 threshold was arbitrary.",
        "",
    ]

    # Selective prediction.
    rc = risk_coverage_curve(y, softmax(results["no TTA"]["logits"], temp))
    out += [rc.to_markdown(), ""]

    # Confusion matrix.
    cm = confusion(y, pred, len(classes))
    out += ["## Confusion matrix (rows = true, cols = predicted)", "",
            "| | " + " | ".join(c[:12] for c in classes) + " |",
            "| --- " * (len(classes) + 1) + "|"]
    for i, cls in enumerate(classes):
        out.append(f"| **{cls[:16]}** | " + " | ".join(str(v) for v in cm[i]) + " |")
    out.append("")

    report_path = args.report or (ROOT / "results" / f"04_eval_{args.run.name}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(out), encoding="utf-8")

    np.savez(
        args.run / "test_predictions.npz",
        logits=results["no TTA"]["logits"],
        probs=probs,
        y=y,
        temperature=temp,
    )

    print("\n".join(out))
    print(f"\nReport -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

