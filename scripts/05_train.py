"""Training driver with the three sweeps the paper needs.

  --sweep leakage    train identically on the group-aware and naive splits.
                     The gap is the leakage effect: the paper's headline result.
  --sweep backbones  the same split, several architectures, so the comparison
                     table compares models rather than quoting other papers.
  --sweep ablation   augmentation x balancing x layer4-unfreeze, over seeds.

Every configuration runs under multiple seeds and results are reported as
mean +/- std. Single-run numbers without variance are not reportable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail.train import TrainConfig, train_one  # noqa: E402

# Five models spanning the design space that matters for this comparison:
# the draft's own architecture, a deeper CNN, an efficient CNN, a modern
# conv/transformer-era backbone, and a frozen-feature linear probe as the floor.
# resnet18 (full fine-tune) and mobilenetv3 were dropped as near-duplicates of
# resnet18_paper and efficientnet_b0 respectively; deit_small was dropped as
# redundant with convnext_tiny at this dataset size.
BACKBONES = [
    "resnet18_paper",
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
    "resnet50_linear",
]


def load_manifest(path: Path, split_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if split_col not in df.columns:
        raise SystemExit(f"manifest has no column '{split_col}'; run 04_make_splits.py")
    df = df.copy()
    df["split"] = df[split_col]
    return df


def run_config(
    manifest: pd.DataFrame, cfg: TrainConfig, tag: str, out_root: Path
) -> dict:
    """Train one configuration, or recover it if it already finished.

    Long sweeps get interrupted. Without this, every restart repeats work that
    is already on disk, which on a windowed scheduler means the sweep can fail
    to make progress at all.
    """
    out_dir = out_root / tag
    done = out_dir / "history.json"
    if done.exists():
        stored = json.loads(done.read_text())
        best_epoch = stored["best_epoch"]
        best_metric = stored["best_metric"]
        print(f"\n=== {tag} === (cached)")
        print(f"  best epoch {best_epoch}  val {cfg.select_on} {best_metric:.4f}")
    else:
        print(f"\n=== {tag} ===")
        result = train_one(manifest, cfg, out_dir)
        best_epoch, best_metric = result.best_epoch, result.best_metric
        print(f"  best epoch {best_epoch}  val {cfg.select_on} {best_metric:.4f}")

    return {
        "tag": tag,
        "best_epoch": best_epoch,
        f"val_{cfg.select_on}": best_metric,
        **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
    }


def aggregate(rows: list[dict], group_keys: list[str], metric: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(group_keys)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    agg["std"] = agg["std"].fillna(0.0)
    agg["summary"] = agg.apply(lambda r: f"{r['mean']:.4f} +/- {r['std']:.4f}", axis=1)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_split.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    ap.add_argument(
        "--sweep", choices=["single", "leakage", "backbones", "ablation"], default="single"
    )
    ap.add_argument("--model", default="resnet18_paper")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--balancing", choices=["sampler", "loss_weights", "none"],
                    default="sampler")
    ap.add_argument("--fc-lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--num-workers", type=int, default=4,
                    help="0 disables DataLoader subprocesses; use it if a run hangs")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def base(**kw) -> TrainConfig:
        params = dict(
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            augment=not args.no_augment,
            balancing=args.balancing,
            fc_lr=args.fc_lr,
            backbone_lr=args.backbone_lr,
            patience=args.patience,
            num_workers=args.num_workers,
        )
        params.update(kw)
        return TrainConfig(**params)

    if args.sweep == "single":
        manifest = load_manifest(args.manifest, args.split_col)
        for seed in args.seeds:
            cfg = base(seed=seed)
            rows.append(
                run_config(manifest, cfg, f"single_{args.model}_s{seed}", args.out)
            )
        summary = aggregate(rows, ["cfg_model"], "val_macro_f1")

    elif args.sweep == "leakage":
        for split_col in ["split", "split_naive"]:
            manifest = load_manifest(args.manifest, split_col)
            for seed in args.seeds:
                cfg = base(seed=seed)
                tag = f"leakage_{split_col}_s{seed}"
                row = run_config(manifest, cfg, tag, args.out)
                row["split_col"] = split_col
                rows.append(row)
        summary = aggregate(rows, ["split_col"], "val_macro_f1")

    elif args.sweep == "backbones":
        manifest = load_manifest(args.manifest, args.split_col)
        for model in BACKBONES:
            for seed in args.seeds:
                cfg = base(model=model, seed=seed)
                bs = 16 if model in {"deit_small", "vit_b16", "convnext_tiny"} else args.batch_size
                cfg.batch_size = bs
                rows.append(run_config(manifest, cfg, f"bb_{model}_s{seed}", args.out))
        summary = aggregate(rows, ["cfg_model"], "val_macro_f1")

    else:  # ablation
        manifest = load_manifest(args.manifest, args.split_col)
        grid = itertools.product([True, False], ["sampler", "none"], [True, False])
        for augment, balancing, unfreeze in grid:
            for seed in args.seeds:
                cfg = base(
                    seed=seed, augment=augment, balancing=balancing,
                    unfreeze_layer4=unfreeze,
                )
                tag = f"abl_a{int(augment)}_b{balancing}_u{int(unfreeze)}_s{seed}"
                rows.append(run_config(manifest, cfg, tag, args.out))
        summary = aggregate(
            rows, ["cfg_augment", "cfg_balancing", "cfg_unfreeze_layer4"], "val_macro_f1"
        )

    runs_csv = args.out / f"{args.sweep}_runs.csv"
    pd.DataFrame(rows).to_csv(runs_csv, index=False)
    summary_csv = args.out / f"{args.sweep}_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 60)
    print(f"SWEEP: {args.sweep}   (val macro-F1, mean +/- std over seeds {args.seeds})")
    print("=" * 60)
    print(summary.to_string(index=False))

    if args.sweep == "leakage" and len(summary) == 2:
        try:
            grouped = float(summary.loc[summary["split_col"] == "split", "mean"].iloc[0])
            naive = float(summary.loc[summary["split_col"] == "split_naive", "mean"].iloc[0])
            print(
                f"\nLeakage effect: naive split scores {naive - grouped:+.4f} macro-F1 "
                f"higher than the group-aware split."
            )
            print("That difference is memorised duplicates, not generalisation.")
        except (IndexError, ValueError):
            pass

    print(f"\nRuns    -> {runs_csv}")
    print(f"Summary -> {summary_csv}")
    print("\nNext: python scripts/06_evaluate.py --run <best run dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
