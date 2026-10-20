"""Does the model learn nail pathology, or dataset provenance?

The metadata test showed that 65% accuracy is reachable from file size alone,
because most classes are drawn from exactly one source and the two sources have
very different technical signatures. That establishes the shortcut exists. It
does not establish that the CNN uses it - the CNN never sees file size, only
pixels resized to 224x224.

This is the decisive test. Train and evaluate within a SINGLE source, so class
cannot be inferred from provenance at all:

  gurav-only      6 classes, 3,835 images, every one 416x416 JPEG. No source
                  signal exists, so any accuracy here is real image content.
  rasanjana-only  3 classes, 1,465 images. Same logic, smaller.

If gurav-only accuracy stays near the pooled number, the model is learning
something about nails and the source confound was not load-bearing. If it
collapses, the pooled number was substantially provenance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail.evaluate import predict_split  # noqa: E402
from nail.metrics import bootstrap_metrics, per_class_table, softmax  # noqa: E402
from nail.splits import make_splits  # noqa: E402
from nail.train import TrainConfig, classes_in, train_one  # noqa: E402
from nail.utils import df_to_markdown  # noqa: E402


def run_one(df: pd.DataFrame, tag: str, out_root: Path, seed: int, epochs: int,
            model: str) -> dict:
    classes = classes_in(df)
    split_df, report = make_splits(df, test_frac=0.15, val_frac=0.15, seed=seed)
    cfg = TrainConfig(model=model, epochs=epochs, patience=5, seed=seed)
    run_dir = out_root / f"{tag}_s{seed}"

    print(f"\n=== {tag}, seed {seed} ===")
    print(f"  {len(df)} images, {len(classes)} classes: {classes}")
    res = train_one(split_df, cfg, run_dir)
    print(f"  best epoch {res.best_epoch}, val macro-F1 {res.best_metric:.4f}")

    logits, y, _ = predict_split(
        split_df, run_dir / "best.pt", cfg, split="test", classes=classes
    )
    probs = softmax(logits)
    pred = probs.argmax(1)
    ci = bootstrap_metrics(y, pred, probs, n_boot=1000)
    print(f"  test accuracy          {ci['accuracy']}")
    print(f"  test balanced accuracy {ci['balanced_accuracy']}")

    return {
        "tag": tag,
        "seed": seed,
        "n_images": len(df),
        "n_classes": len(classes),
        "n_test": len(y),
        "accuracy": ci["accuracy"],
        "balanced_accuracy": ci["balanced_accuracy"],
        "macro_f1": ci["macro_f1"],
        "per_class": per_class_table(y, pred, classes),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_grouped.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "runs_source")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "07_source.md")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--model", default="resnet18_paper")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["label"].notna()]

    results = []
    for seed in args.seeds:
        for source in ["gurav", "rasanjana"]:
            sub = df[df["source"] == source].copy()
            if sub["label"].nunique() < 2:
                continue
            results.append(
                run_one(sub, f"{source}_only", args.out, seed, args.epochs, args.model)
            )

    lines = [
        "# Source-controlled evaluation",
        "",
        f"Model `{args.model}`, group-aware splits, seeds {args.seeds}.",
        "",
        "Each row trains and tests within a single source, so class cannot be",
        "inferred from provenance.",
        "",
        "| condition | images | classes | test n | accuracy [95% CI] | balanced acc [95% CI] |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['tag']} (seed {r['seed']}) | {r['n_images']} | {r['n_classes']} | "
            f"{r['n_test']} | {r['accuracy']} | {r['balanced_accuracy']} |"
        )

    lines += ["", "## Per class", ""]
    for r in results:
        pc = pd.DataFrame(r["per_class"]).T
        pc["support"] = pc["support"].astype(int)
        lines += [f"### {r['tag']} (seed {r['seed']})", "",
                  df_to_markdown(pc.round(3), index_name="class"), ""]

    lines += [
        "## Reading this",
        "",
        "Compare against the pooled group-aware number in `05_leakage.md`. If the",
        "single-source accuracy is comparable, the model is separating classes on",
        "image content rather than on which dataset the file came from, and the",
        "provenance confound - though real and worth disclosing - is not carrying",
        "the result. If it drops sharply, the pooled figure was substantially",
        "measuring provenance and the paper's central claim does not hold.",
        "",
        "Note that gurav-only is the stronger of the two tests: 3,835 images across",
        "6 classes, all identical 416x416 JPEGs, so there is no technical signature",
        "distinguishing any class from any other.",
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nReport -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
