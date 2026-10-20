"""Elsevier graphical abstract.

A graphical abstract is a single image summarising the paper's main finding at a
glance. It appears beside the title in ScienceDirect search results and in the
table of contents, so it has to survive being shown small: Elsevier's guidance is
a minimum of 531 x 1328 pixels (height x width) at 300 dpi, legible at roughly
200 x 500 pixels. That means very few words, large type and no fine detail.

This draws the paper's argument left to right: what the audit finds in the public
data, what the protocol does about it, and what the audited model achieves.

Note: write this file with a UTF-8 aware editor. Round-tripping it through
PowerShell's Set-Content mangles the typographic characters below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1a1a1a"
MUTED = "#5a6570"
RED = "#9b1c2e"
TEAL = "#0d5c66"
GREEN = "#1c6b45"
PANEL = "#f4f6f7"
LINE = "#c9d2d7"

DOT = "·"      # middle dot
DASH = "—"     # em dash
TAU = "τ"      # Greek tau


def box(ax, x, y, w, h, fc=PANEL, ec=LINE, lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1))


def arrow(ax, x1, y, x2):
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y), arrowstyle="-|>", mutation_scale=15,
        color=MUTED, linewidth=1.4, zorder=3))


def spaced(s: str) -> str:
    """Poor man's letter-spacing; matplotlib Text has no such property."""
    return " ".join(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "results" / "figures" / "graphical_abstract.png")
    args = ap.parse_args()

    fig = plt.figure(figsize=(13.28, 5.31), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0",
                                facecolor="white", edgecolor="none", zorder=0))

    ax.text(0.5, 0.945, "Auditing public biomedical image collections",
            ha="center", va="center", fontsize=21, color=INK, fontweight="semibold")
    ax.text(0.5, 0.878,
            f"duplication  {DOT}  provenance confounding  {DOT}  shortcut learning",
            ha="center", va="center", fontsize=13, color=MUTED)

    # ---- what the audit finds
    box(ax, 0.030, 0.185, 0.285, 0.62)
    ax.text(0.1725, 0.745, spaced("PUBLIC COLLECTIONS"), ha="center",
            fontsize=11.5, color=RED, fontweight="bold")
    findings = [
        ("65%", "of images are augmented\ncopies of 1,337 originals"),
        ("95.8%", "of a conventional test set\nduplicates training data"),
        ("1.2%", f"is all four perceptual\nsignals could detect"),
    ]
    y = 0.625
    for big, small in findings:
        ax.text(0.068, y, big, ha="left", va="center", fontsize=19,
                color=RED, fontweight="bold")
        ax.text(0.163, y, small, ha="left", va="center", fontsize=10,
                color=INK, linespacing=1.35)
        y -= 0.155
    ax.text(0.1725, 0.225, "all invisible to standard evaluation",
            ha="center", fontsize=9.5, color=MUTED, style="italic")

    arrow(ax, 0.325, 0.495, 0.375)

    # ---- the protocol
    box(ax, 0.385, 0.185, 0.230, 0.62, fc="white", ec=TEAL, lw=1.6)
    ax.text(0.500, 0.745, spaced("AUDIT PROTOCOL"), ha="center",
            fontsize=11.5, color=TEAL, fontweight="bold")
    steps = [
        "Filename provenance,\nnot pixel similarity",
        "Metadata-only\nshortcut probe",
        "Equal-area\nocclusion control",
    ]
    y = 0.625
    for i, s in enumerate(steps, 1):
        ax.text(0.410, y, f"{i}", ha="center", va="center", fontsize=13,
                color="white", fontweight="bold", zorder=4,
                bbox=dict(boxstyle="circle,pad=0.32", facecolor=TEAL, edgecolor="none"))
        ax.text(0.435, y, s, ha="left", va="center", fontsize=10.5,
                color=INK, linespacing=1.35)
        y -= 0.155
    ax.text(0.500, 0.225, "splits made over source photographs",
            ha="center", fontsize=9.5, color=MUTED, style="italic")

    arrow(ax, 0.625, 0.495, 0.675)

    # ---- result
    box(ax, 0.685, 0.185, 0.285, 0.62, fc="#f2f8f4", ec=GREEN, lw=1.6)
    ax.text(0.8275, 0.745, spaced("AUDITED RESULT"), ha="center",
            fontsize=11.5, color=GREEN, fontweight="bold")
    ax.text(0.8275, 0.615, "93.8%", ha="center", va="center", fontsize=40,
            color=GREEN, fontweight="bold")
    ax.text(0.8275, 0.530, f"balanced accuracy  {DOT}  6 classes",
            ha="center", va="center", fontsize=10.5, color=INK)
    ax.text(0.8275, 0.487, "191 independent photographs",
            ha="center", va="center", fontsize=9.5, color=MUTED, style="italic")

    for i, (k, v) in enumerate([
        ("Inflation removed", "5.0 pp"),
        ("OOD detection AUROC", "0.930"),
        ("Augmentation x trunk", "+8.1 pp"),
    ]):
        yy = 0.400 - i * 0.058
        ax.text(0.715, yy, k, ha="left", va="center", fontsize=9.8, color=MUTED)
        ax.text(0.945, yy, v, ha="right", va="center", fontsize=10.5,
                color=INK, fontweight="bold")

    ax.plot([0.030, 0.970], [0.135, 0.135], color=LINE, linewidth=1.0)
    ax.text(0.030, 0.082,
            "The duplication was recoverable from filenames, not from any image "
            f"similarity signal  {DASH}  check provenance before trusting a dedup",
            ha="left", va="center", fontsize=10.5, color=RED)
    ax.text(0.030, 0.035,
            f"Nail disease case study  {DOT}  3,835 images from 1,337 photographs  "
            f"{DOT}  code and split indices released",
            ha="left", va="center", fontsize=9.5, color=MUTED)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"{args.out}  (2656 x 1062 px)")
    print("Elsevier minimum is 1328 x 531; this is 2x that in each dimension.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
