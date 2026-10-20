"""Publication figures, generated from the actual runs.

Replaces the draft's figures, several of which are wrong or absent:

  Fig. 2  promised an accuracy panel and a loss curve; only one plot appears, over
          ~12 epochs, captioned as stopping at epoch 22.
  Fig. 4  legend reports AUC = 1.00 for five classes while the text claims 0.92-0.95.

Everything here is drawn from saved run artefacts, so figure and text cannot drift
apart again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail import PAPER_CLASSES  # noqa: E402
from nail.metrics import confusion, reliability_bins, risk_coverage_curve, softmax  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

SHORT = {
    "Acral_Lentiginous_Melanoma": "ALM",
    "Blue_Finger": "Blue finger",
    "Clubbing": "Clubbing",
    "Healthy_Nail": "Healthy",
    "Onychogryphosis": "Onychogryph.",
    "Pitting": "Pitting",
}


def load_runs(run_dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    hists, preds = [], []
    for d in run_dirs:
        h = d / "history.json"
        p = d / "test_predictions.npz"
        if h.exists():
            hists.append(json.loads(h.read_text()))
        if p.exists():
            z = np.load(p)
            preds.append(
                {"y": z["y"], "probs": z["probs"], "logits": z["logits"],
                 "T": float(z["temperature"])}
            )
    return hists, preds


def fig_training(hists: list[dict], out: Path) -> None:
    """Both panels the draft promised: metric curves and loss, over the true range."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    for i, h in enumerate(hists):
        ep = [r["epoch"] for r in h["history"]]
        axes[0].plot(ep, [r["macro_f1"] for r in h["history"]],
                     label=f"seed {h['config']['seed']}", lw=1.4)
        axes[1].plot(ep, [r["train_loss"] for r in h["history"]],
                     label=f"seed {h['config']['seed']}", lw=1.4)
        best = h["best_epoch"]
        axes[0].axvline(best, ls=":", lw=0.8, alpha=0.5)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("validation macro-F1")
    axes[0].set_title("(a) validation macro-F1", loc="left", fontsize=9)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("training loss")
    axes[1].set_title("(b) training loss", loc="left", fontsize=9)
    axes[0].legend(fontsize=7.5)
    fig.savefig(out)
    plt.close(fig)


def fig_confusion(preds: list[dict], out: Path) -> None:
    cms = [confusion(p["y"], p["probs"].argmax(1), len(PAPER_CLASSES)) for p in preds]
    cm = np.sum(cms, axis=0)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = [SHORT[c] for c in PAPER_CLASSES]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7.5,
                        color="white" if norm[i, j] > 0.55 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045, label="row-normalised")
    ax.set_title(f"pooled over {len(preds)} seeds", loc="left", fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def fig_roc(preds: list[dict], out: Path) -> None:
    """ROC on a log FPR axis, with partial AUC over the clinically useful region.

    On a linear axis every curve here is pinned to the top-left corner at
    AUC 1.000 and the figure carries no information - which is what makes the
    draft's Fig. 4 decorative. A log FPR axis exposes the low-false-positive
    regime, which is the only region a screening tool operates in, and pAUC over
    FPR <= 0.1 separates classes that full AUC cannot.
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    y = np.concatenate([p["y"] for p in preds])
    probs = np.concatenate([p["probs"] for p in preds])
    floor = 1.0 / len(y)
    limit = 0.1
    for i, cls in enumerate(PAPER_CLASSES):
        fpr, tpr, _ = roc_curve((y == i).astype(int), probs[:, i])
        full = auc(fpr, tpr)
        # Integrate to the FPR boundary itself, interpolating the endpoint.
        # sklearn.auc over `fpr <= limit` stops at the largest point that happens
        # to exist, which on a near-perfect curve is ~0 - which is why a flawless
        # classifier was reporting pAUC 0.000.
        keep = fpr <= limit
        fpr_p = np.append(fpr[keep], limit)
        tpr_p = np.append(tpr[keep], np.interp(limit, fpr, tpr))
        pauc = float(np.trapezoid(tpr_p, fpr_p) / limit)
        ax.plot(np.clip(fpr, floor, None), tpr, lw=1.4,
                label=f"{SHORT[cls]}  AUC {full:.4f}  pAUC {pauc:.4f}")
    ax.set_xscale("log")
    ax.set_xlim(floor, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("false positive rate (log scale)")
    ax.set_ylabel("true positive rate")
    ax.axvline(limit, ls=":", lw=0.9, color="crimson", alpha=0.7)
    ax.text(limit * 1.2, 0.55, "pAUC\nboundary", fontsize=6.5, color="crimson")
    ax.legend(fontsize=6.6, loc="lower left",
              title="normalised pAUC over FPR ≤ 0.1", title_fontsize=6.5)
    fig.savefig(out)
    plt.close(fig)


def fig_calibration(preds: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for label, key, ax in [("before scaling", None, axes[0]),
                           ("after scaling", "T", axes[1])]:
        accs, confs, counts = [], [], []
        for p in preds:
            pr = p["probs"] if key is None else softmax(p["logits"], p["T"])
            b = reliability_bins(p["y"], pr, n_bins=10)
            accs.append(b["accuracy"]); confs.append(b["confidence"]); counts.append(b["count"])
        acc = np.nanmean(accs, axis=0); conf = np.nanmean(confs, axis=0)
        cnt = np.sum(counts, axis=0)
        centres = reliability_bins(preds[0]["y"], preds[0]["probs"], n_bins=10)["bin_centre"]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="perfect")
        m = ~np.isnan(acc)
        # Marker area tracks bin population. Without this, bins holding two or
        # three samples are drawn as emphatically as bins holding several hundred,
        # and the eye reads sampling noise as miscalibration.
        frac = cnt / max(cnt.sum(), 1)
        ax.plot(conf[m], acc[m], "-", lw=1.0, color="tab:blue", alpha=0.6)
        ax.scatter(conf[m], acc[m], s=12 + 260 * frac[m], color="tab:blue",
                   alpha=0.85, zorder=3, label="observed (area ∝ bin count)")
        ax2 = ax.twinx()
        ax2.bar(centres, frac, width=0.09, alpha=0.22, color="gray")
        ax2.set_ylim(0, 1); ax2.set_yticks([]); ax2.grid(False)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(label, loc="left", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")
    fig.savefig(out)
    plt.close(fig)


def fig_risk_coverage(preds: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for p in preds:
        rc = risk_coverage_curve(p["y"], softmax(p["logits"], p["T"]))
        ax.plot(rc.coverage, rc.risk * 100, lw=1.3, alpha=0.85)
    for t, style in [(0.7, ":"), (0.9, "--")]:
        cov = np.mean([
            risk_coverage_curve(p["y"], softmax(p["logits"], p["T"]))
            .at_threshold[t]["coverage"] for p in preds
        ])
        ax.axvline(cov, ls=style, lw=0.9, color="crimson", alpha=0.8)
        ax.text(cov, ax.get_ylim()[1] * 0.92, f" τ={t}", fontsize=7, color="crimson")
    ax.set_xlabel("coverage"); ax.set_ylabel("selective error (%)")
    ax.set_xlim(0, 1.005)
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "runs_headline")
    ap.add_argument("--pattern", default="single_efficientnet_b0_s*")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    args = ap.parse_args()

    dirs = sorted(args.runs.glob(args.pattern))
    hists, preds = load_runs(dirs)
    if not preds:
        print(f"no test predictions under {args.runs}/{args.pattern}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    made = []
    fig_training(hists, args.out / "fig2_training.png"); made.append("fig2_training.png")
    fig_confusion(preds, args.out / "fig3_confusion.png"); made.append("fig3_confusion.png")
    fig_roc(preds, args.out / "fig4_roc.png"); made.append("fig4_roc.png")
    fig_calibration(preds, args.out / "fig5_calibration.png"); made.append("fig5_calibration.png")
    fig_risk_coverage(preds, args.out / "fig6_risk_coverage.png"); made.append("fig6_risk_coverage.png")

    print(f"{len(dirs)} runs, {len(preds)} with test predictions")
    for f in made:
        print(f"  {args.out / f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
