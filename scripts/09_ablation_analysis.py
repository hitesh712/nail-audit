"""Paired main effects and the augmentation x trunk interaction.

The merged-dataset ablation and the clean-dataset ablation disagree on the sign
of the augmentation effect. That is not noise: it is an interaction with how much
of the trunk is trainable, and it needs reporting as such rather than as a single
main effect.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

METRIC = "val_macro_f1"


def paired_effect(df: pd.DataFrame, factor: str, others: list[str]) -> dict:
    piv = df.pivot_table(index=others + ["cfg_seed"], columns=factor, values=METRIC).dropna()
    cols = list(piv.columns)
    d = (piv[cols[1]] - piv[cols[0]]) * 100
    t, p = st.ttest_1samp(d, 0)
    lo, hi = st.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=st.sem(d))
    return {
        "factor": factor,
        "from": cols[0],
        "to": cols[1],
        "effect_pp": d.mean(),
        "ci_lo": lo,
        "ci_hi": hi,
        "p": p,
        "n_pairs": len(d),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "runs_final_abl")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "12_ablation.md")
    ap.add_argument("--label", default="gurav-only, ResNet-50, 12 epochs")
    args = ap.parse_args()

    df = pd.read_csv(args.runs / "ablation_runs.csv")
    lines = [f"# Ablation - {args.label}", "", "## Main effects (paired, percentage points of macro-F1)", "",
             "| factor | change | effect | 95% CI | p | pairs |", "| --- | --- | --- | --- | --- | --- |"]

    for factor, others in [
        ("cfg_unfreeze_layer4", ["cfg_augment", "cfg_balancing"]),
        ("cfg_augment", ["cfg_balancing", "cfg_unfreeze_layer4"]),
        ("cfg_balancing", ["cfg_augment", "cfg_unfreeze_layer4"]),
    ]:
        r = paired_effect(df, factor, others)
        lines.append(
            f"| `{r['factor']}` | {r['from']} → {r['to']} | **{r['effect_pp']:+.2f}** | "
            f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p']:.4f} | {r['n_pairs']} |"
        )

    # The interaction that matters: does augmentation help only when the trunk moves?
    lines += ["", "## Augmentation effect, split by whether the trunk is trainable", "",
              "| trunk | augmentation effect | 95% CI | p | pairs |", "| --- | --- | --- | --- | --- |"]
    for unfrozen in [True, False]:
        sub = df[df["cfg_unfreeze_layer4"] == unfrozen]
        r = paired_effect(sub, "cfg_augment", ["cfg_balancing"])
        label = "trainable" if unfrozen else "frozen"
        lines.append(
            f"| {label} | **{r['effect_pp']:+.2f}** | "
            f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p']:.4f} | {r['n_pairs']} |"
        )

    # Formal interaction test: difference of the two augmentation effects.
    piv = df.pivot_table(
        index=["cfg_balancing", "cfg_seed"],
        columns=["cfg_unfreeze_layer4", "cfg_augment"],
        values=METRIC,
    ).dropna()
    eff_unfrozen = (piv[(True, True)] - piv[(True, False)]) * 100
    eff_frozen = (piv[(False, True)] - piv[(False, False)]) * 100
    diff = eff_unfrozen - eff_frozen
    t, p = st.ttest_1samp(diff, 0)
    lo, hi = st.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=st.sem(diff))
    lines += [
        "",
        f"**Interaction**: the augmentation effect is {diff.mean():+.2f} pp larger when "
        f"the trunk is trainable, 95% CI [{lo:+.2f}, {hi:+.2f}], p={p:.4f} "
        f"({len(diff)} paired).",
        "",
        "## Reading this",
        "",
        "Augmentation and trunk adaptation are not independent. With the trunk frozen,",
        "the classifier can only fit whatever ImageNet features survive, and augmentation",
        "adds variance it has no capacity to absorb - so it hurts. With the trunk",
        "trainable, the network can actually learn the invariances augmentation supplies,",
        "and it helps. Reporting a single main effect for augmentation therefore averages",
        "over two opposite regimes and is misleading in either direction.",
        "",
        "This also explains why the draft's Table VI is not reproducible: a +11.6 point",
        "augmentation gain is only reachable if the comparison baseline differed in",
        "something other than augmentation.",
        "",
        "## Config ranking",
        "",
        "| augment | balancing | trunk | macro-F1 |",
        "| --- | --- | --- | --- |",
    ]
    agg = (
        df.groupby(["cfg_augment", "cfg_balancing", "cfg_unfreeze_layer4"])[METRIC]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    for _, r in agg.iterrows():
        trunk = "trainable" if r["cfg_unfreeze_layer4"] else "frozen"
        lines.append(
            f"| {r['cfg_augment']} | {r['cfg_balancing']} | {trunk} | "
            f"{r['mean']:.4f} +/- {r['std']:.4f} |"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    # Console may be cp1252; never let a reporting print kill the run.
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))
    print(f"\nReport -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
