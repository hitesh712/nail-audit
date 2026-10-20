"""Merge per-seed ablation directories into one runs table.

The sweep was executed one seed at a time because each scheduling window was too
short for all 24 configurations. The analysis in 09_ablation_analysis.py expects a
single ablation_runs.csv containing every run, so concatenate them here and verify
the factorial design is complete before any paired test is computed.

An incomplete factorial silently breaks the pairing: a missing cell drops the whole
matched comparison it belongs to, which shrinks n without saying so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def rows_from(run_dir: Path) -> list[dict]:
    out = []
    for h in sorted(run_dir.rglob("history.json")):
        d = json.loads(h.read_text())
        cfg = dict(d["config"])
        cfg.pop("classes", None)
        out.append({
            "tag": h.parent.name,
            "best_epoch": d["best_epoch"],
            "val_macro_f1": d["best_metric"],
            **{f"cfg_{k}": v for k, v in cfg.items()},
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", type=Path, nargs="+",
                    default=[ROOT / "results" / "runs_sa_abl_s0",
                             ROOT / "results" / "runs_sa_abl_s1",
                             ROOT / "results" / "runs_sa_abl_s2"])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "runs_sa_abl")
    args = ap.parse_args()

    rows: list[dict] = []
    for d in args.dirs:
        found = rows_from(d)
        print(f"{d.name}: {len(found)} runs")
        rows.extend(found)

    if not rows:
        print("no runs found")
        return 1

    df = pd.DataFrame(rows).drop_duplicates(subset="tag")
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "ablation_runs.csv", index=False)

    factors = ["cfg_augment", "cfg_balancing", "cfg_unfreeze_layer4"]
    cells = df.groupby(factors + ["cfg_seed"]).size()
    expected = 1
    for f in factors:
        expected *= df[f].nunique()
    seeds = sorted(df["cfg_seed"].unique())
    expected_total = expected * len(seeds)

    print(f"\n{len(df)} unique runs, {expected} configurations x {len(seeds)} seeds "
          f"= {expected_total} expected")
    if len(df) != expected_total:
        print("INCOMPLETE - the factorial is not full. Paired tests would silently")
        print("drop matched comparisons. Missing cells:")
        full = pd.MultiIndex.from_product(
            [sorted(df[f].unique()) for f in factors] + [seeds],
            names=factors + ["cfg_seed"])
        missing = full.difference(cells.index)
        for m in missing:
            print(f"  {dict(zip(factors + ['cfg_seed'], m))}")
        return 1

    print("Factorial complete.")
    print(f"-> {args.out / 'ablation_runs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
