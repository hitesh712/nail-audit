"""Flag runs whose best epoch sits suspiciously early.

A model that peaks in the first few epochs and then degrades has not converged
slowly - it has diverged, usually because the learning rate is wrong for that
architecture. Reporting such a run as a weak architecture is a hyperparameter
failure dressed as a finding, which is exactly what ConvNeXt-Tiny nearly became
earlier in this project.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+",
                    default=[ROOT / "results" / "runs_sa_bb",
                             ROOT / "results" / "runs_sa_bb2",
                             ROOT / "results" / "runs_source_aware"])
    ap.add_argument("--early", type=int, default=3,
                    help="best epoch at or below this is flagged")
    args = ap.parse_args()

    rows = []
    for root in args.runs:
        for h in sorted(root.rglob("history.json")):
            d = json.loads(h.read_text())
            hist = d["history"]
            best = d["best_epoch"]
            f1s = [r["macro_f1"] for r in hist]
            rows.append({
                "run": h.parent.name,
                "best_epoch": best,
                "epochs_run": len(hist),
                "best_f1": d["best_metric"],
                "final_f1": f1s[-1] if f1s else float("nan"),
                "first_f1": f1s[0] if f1s else float("nan"),
                "suspect": best <= args.early and len(hist) > args.early + 2,
            })

    rows.sort(key=lambda r: r["best_epoch"])
    print(f"{'run':38s} {'best':>5s} {'/run':>5s} {'bestF1':>8s} "
          f"{'finalF1':>8s} {'ep0F1':>7s}  flag")
    print("-" * 85)
    for r in rows:
        flag = "DIVERGED?" if r["suspect"] else ""
        print(f"{r['run']:38s} {r['best_epoch']:5d} {r['epochs_run']:5d} "
              f"{r['best_f1']:8.4f} {r['final_f1']:8.4f} {r['first_f1']:7.4f}  {flag}")

    bad = [r for r in rows if r["suspect"]]
    print()
    if bad:
        print(f"{len(bad)} run(s) peaked at or before epoch {args.early} and then")
        print("declined. Inspect before quoting their numbers; a diverged run is a")
        print("learning-rate failure, not evidence about the architecture.")
    else:
        print("No run peaked suspiciously early.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
