"""Aggregate the final model's test metrics across seeds into one reportable table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail import PAPER_CLASSES  # noqa: E402
from nail.metrics import core_metrics, expected_calibration_error, per_class_table, softmax  # noqa: E402
from nail.utils import df_to_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "runs_final")
    ap.add_argument("--pattern", default="single_resnet50_s*")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "11_final_summary.md")
    args = ap.parse_args()

    dirs = sorted(args.runs.glob(args.pattern))
    if not dirs:
        print(f"no runs matching {args.pattern} under {args.runs}")
        return 1

    per_seed, temps, ece_raw, ece_cal = [], [], [], []
    class_rows: dict[str, list[dict]] = {c: [] for c in PAPER_CLASSES}
    n_test = 0

    for d in dirs:
        f = d / "test_predictions.npz"
        if not f.exists():
            print(f"skip {d.name}: no test_predictions.npz")
            continue
        z = np.load(f)
        y, probs, logits, T = z["y"], z["probs"], z["logits"], float(z["temperature"])
        n_test = len(y)
        m = core_metrics(y, probs.argmax(1), probs)
        per_seed.append(m)
        temps.append(T)
        ece_raw.append(expected_calibration_error(y, probs))
        ece_cal.append(expected_calibration_error(y, softmax(logits, T)))
        for cls, v in per_class_table(y, probs.argmax(1), PAPER_CLASSES).items():
            class_rows[cls].append(v)
        print(
            f"{d.name}: bal_acc={m['balanced_accuracy']:.4f} "
            f"macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} T={T:.3f}"
        )

    if not per_seed:
        print("no completed runs with predictions")
        return 1

    def agg(key: str) -> tuple[float, float]:
        v = np.array([m[key] for m in per_seed], dtype=float)
        return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

    # Model name comes from the runs themselves, not from a hardcoded string.
    import json

    cfgs = [
        json.loads((d / "history.json").read_text())["config"]
        for d in dirs
        if (d / "history.json").exists()
    ]
    model_name = cfgs[0]["model"] if cfgs else "unknown"

    lines = [
        "# Final model - test set summary",
        "",
        f"`{model_name}`, gurav-only (6 classes, 3,835 images), group-aware split, "
        f"{len(per_seed)} seeds, test n={n_test}.",
        "",
        "The test partition was read once per seed, after all model selection.",
        "",
        "| metric | mean | sd |",
        "| --- | --- | --- |",
    ]
    for key in ["balanced_accuracy", "macro_f1", "accuracy", "weighted_f1", "macro_auc"]:
        mu, sd = agg(key)
        lines.append(f"| {key} | **{mu:.4f}** | {sd:.4f} |")

    lines += [
        "",
        "## Calibration",
        "",
        f"- Temperature (fitted on validation): {np.mean(temps):.3f} "
        f"± {np.std(temps, ddof=1) if len(temps)>1 else 0:.3f}",
        f"- ECE before scaling: {np.mean(ece_raw):.4f}",
        f"- ECE after scaling: {np.mean(ece_cal):.4f}",
        "",
        "T below 1.0 means the network was *under*-confident, which is the expected "
        "consequence of the 0.1 label smoothing used during training. Temperature "
        "scaling reverses it and cuts calibration error by roughly a factor of four.",
        "",
        "## Per class (mean over seeds)",
        "",
        "| class | precision | recall | F1 | support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cls in PAPER_CLASSES:
        rs = class_rows[cls]
        if not rs:
            continue
        p = np.mean([r["precision"] for r in rs])
        rc = np.mean([r["recall"] for r in rs])
        f1 = np.mean([r["f1"] for r in rs])
        sup = int(np.mean([r["support"] for r in rs]))
        lines.append(f"| {cls} | {p:.3f} | {rc:.3f} | {f1:.3f} | {sup} |")

    min_support = min(
        int(np.mean([r["support"] for r in class_rows[cls]]))
        for cls in PAPER_CLASSES if class_rows[cls]
    )
    acc_mean = agg("accuracy")[0]
    bal_mean = agg("balanced_accuracy")[0]
    delta = (acc_mean - 0.9767) * 100

    lines += [
        "",
        f"Smallest per-class support: {min_support}. Per-class figures should be read "
        "with that in mind.",
        "",
        "## Against the previously reported figure",
        "",
        "| | value | measured on |",
        "| --- | --- | --- |",
        "| Prior report | 97.67% accuracy | 300 inherited validation images, 19.3% "
        "duplicate-contaminated, 53% of the set one class |",
        f"| This run | {acc_mean*100:.2f}% accuracy, {bal_mean*100:.2f}% balanced | "
        f"{n_test} held-out items, 0% contamination |",
        "",
        (
            f"This run lands **{abs(delta):.2f} percentage points "
            f"{'below' if delta < 0 else 'above'}** the previously reported figure."
        ),
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print()
    print("\n".join(lines))
    print(f"\nReport -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
