"""The headline experiment: how much of 97.67% is memorised duplicates?

Two conditions, both using the draft's own architecture and hyperparameters.

  A. The draft's setup verbatim - its 6-class taxonomy, its shipped train/val
     folders. Its 300-image validation set is then scored twice: once on the 58
     images that share a duplicate group with training data, once on the 242 that
     do not. Same model, same val set, so the gap between those two numbers is
     memorisation and nothing else.

  B. A group-aware stratified split over the same pool, which is what the number
     should have been measured on.

Condition A is the clean argument. It has no confound from differing training
sets or class distributions - it is one model, one held-out set, partitioned by
whether the model had already seen a near-copy.
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
from nail.metrics import bootstrap_metrics, core_metrics, softmax  # noqa: E402
from nail.splits import make_splits  # noqa: E402
from nail.train import TrainConfig, classes_in, train_one  # noqa: E402


def evaluate_subset(y: np.ndarray, probs: np.ndarray, mask: np.ndarray) -> dict:
    if mask.sum() == 0:
        return {"n": 0}
    pred = probs.argmax(1)
    return {
        "n": int(mask.sum()),
        "accuracy": float((pred[mask] == y[mask]).mean()),
        **{
            k: v
            for k, v in core_metrics(y[mask], pred[mask]).items()
            if k in {"balanced_accuracy", "macro_f1"}
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest", type=Path, default=ROOT / "data" / "manifest_paper_split.csv"
    )
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "runs_leakage")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "05_leakage.md")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--model", default="resnet18_paper")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    classes = classes_in(df)
    print(f"{len(df)} images, {len(classes)} classes: {classes}")

    rows_a: list[dict] = []
    rows_b: list[dict] = []

    for seed in args.seeds:
        # ---------------------------------------------------------- condition A
        man_a = df.copy()
        man_a["split"] = man_a["split_shipped"]
        cfg = TrainConfig(
            model=args.model, epochs=args.epochs, patience=args.patience, seed=seed
        )
        run_a = args.out / f"A_shipped_s{seed}"
        print(f"\n=== A: draft's shipped split, seed {seed} ===")
        res = train_one(man_a, cfg, run_a)
        print(f"  best epoch {res.best_epoch}, val macro-F1 {res.best_metric:.4f}")

        logits, y, val_rows = predict_split(
            man_a, run_a / "best.pt", cfg, split="val", classes=classes
        )
        probs = softmax(logits)
        pred = probs.argmax(1)

        train_groups = set(man_a.loc[man_a["split"] == "train", "group_id"])
        contaminated = val_rows["group_id"].isin(train_groups).to_numpy()

        overall = {"n": len(y), "accuracy": float((pred == y).mean()),
                   **{k: v for k, v in core_metrics(y, pred).items()
                      if k in {"balanced_accuracy", "macro_f1"}}}
        dirty = evaluate_subset(y, probs, contaminated)
        clean = evaluate_subset(y, probs, ~contaminated)

        rows_a.append({"seed": seed, "subset": "overall", **overall})
        rows_a.append({"seed": seed, "subset": "contaminated", **dirty})
        rows_a.append({"seed": seed, "subset": "clean", **clean})

        print(f"  overall      n={overall['n']:4d}  acc={overall['accuracy']*100:.2f}%")
        print(f"  contaminated n={dirty['n']:4d}  acc={dirty['accuracy']*100:.2f}%")
        print(f"  clean        n={clean['n']:4d}  acc={clean['accuracy']*100:.2f}%")

        np.savez(
            run_a / "val_breakdown.npz",
            y=y, probs=probs, contaminated=contaminated,
        )

        # ---------------------------------------------------------- condition B
        man_b, split_report = make_splits(df, test_frac=0.15, val_frac=0.15, seed=seed)
        cfg_b = TrainConfig(
            model=args.model, epochs=args.epochs, patience=args.patience, seed=seed
        )
        run_b = args.out / f"B_groupaware_s{seed}"
        print(f"\n=== B: group-aware split, seed {seed} ===")
        res_b = train_one(man_b, cfg_b, run_b)
        print(f"  best epoch {res_b.best_epoch}, val macro-F1 {res_b.best_metric:.4f}")

        logits_b, y_b, _ = predict_split(
            man_b, run_b / "best.pt", cfg_b, split="test", classes=classes
        )
        probs_b = softmax(logits_b)
        ci = bootstrap_metrics(y_b, probs_b.argmax(1), probs_b, n_boot=1000)
        rows_b.append(
            {
                "seed": seed,
                "n": len(y_b),
                "accuracy": ci["accuracy"].point,
                "accuracy_lo": ci["accuracy"].lo,
                "accuracy_hi": ci["accuracy"].hi,
                "balanced_accuracy": ci["balanced_accuracy"].point,
                "balanced_accuracy_lo": ci["balanced_accuracy"].lo,
                "balanced_accuracy_hi": ci["balanced_accuracy"].hi,
                "macro_f1": ci["macro_f1"].point,
                "macro_f1_lo": ci["macro_f1"].lo,
                "macro_f1_hi": ci["macro_f1"].hi,
            }
        )
        print(f"  test accuracy {ci['accuracy']}")
        print(f"  test balanced accuracy {ci['balanced_accuracy']}")

    a = pd.DataFrame(rows_a)
    b = pd.DataFrame(rows_b)
    args.out.mkdir(parents=True, exist_ok=True)
    a.to_csv(args.out / "condition_A.csv", index=False)
    b.to_csv(args.out / "condition_B.csv", index=False)

    agg_a = a.groupby("subset").agg(
        n=("n", "first"),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
    ).fillna(0.0)

    lines = [
        "# Leakage experiment",
        "",
        f"Model `{args.model}`, seeds {args.seeds}, max {args.epochs} epochs.",
        "",
        "## Condition A - the draft's own split, partitioned by contamination",
        "",
        "One model, one held-out set. The only difference between the two rows is",
        "whether a near-duplicate of the image was in the training data.",
        "",
        "| subset | n | accuracy |",
        "| --- | --- | --- |",
    ]
    for subset in ["overall", "contaminated", "clean"]:
        if subset in agg_a.index:
            r = agg_a.loc[subset]
            sd = f" ± {r['acc_std']*100:.2f}" if len(args.seeds) > 1 else ""
            lines.append(
                f"| {subset} | {int(r['n'])} | {r['acc_mean']*100:.2f}%{sd} |"
            )

    if {"contaminated", "clean"} <= set(agg_a.index):
        gap = (
            agg_a.loc["contaminated", "acc_mean"] - agg_a.loc["clean", "acc_mean"]
        ) * 100
        lines += [
            "",
            f"**Memorisation gap: {gap:+.2f} percentage points.**",
            "",
            "The contaminated images are not harder or easier by any clinical",
            "measure - they were drawn from the same distribution. The gap is the",
            "model recognising pictures it has already seen.",
        ]

    lines += [
        "",
        "## Condition B - group-aware split, held-out test set",
        "",
        "| seed | n | accuracy [95% CI] | balanced accuracy [95% CI] | macro-F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, r in b.iterrows():
        lines.append(
            f"| {int(r['seed'])} | {int(r['n'])} | "
            f"{r['accuracy']:.3f} [{r['accuracy_lo']:.3f}, {r['accuracy_hi']:.3f}] | "
            f"{r['balanced_accuracy']:.3f} [{r['balanced_accuracy_lo']:.3f}, "
            f"{r['balanced_accuracy_hi']:.3f}] | {r['macro_f1']:.3f} |"
        )

    lines += [
        "",
        "## Against the published claim",
        "",
        "| | accuracy |",
        "| --- | --- |",
        "| Published (draft, Table V) | 97.67% |",
    ]
    if "overall" in agg_a.index:
        lines.append(
            f"| Reproduced on the draft's own split | "
            f"{agg_a.loc['overall', 'acc_mean']*100:.2f}% |"
        )
    if "clean" in agg_a.index:
        lines.append(
            f"| Same model, uncontaminated subset only | "
            f"{agg_a.loc['clean', 'acc_mean']*100:.2f}% |"
        )
    if len(b):
        lines.append(
            f"| Group-aware split, held-out test | {b['accuracy'].mean()*100:.2f}% |"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nReport -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
