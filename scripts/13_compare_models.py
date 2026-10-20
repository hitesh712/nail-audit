"""Is EfficientNet-B0 genuinely better than ResNet-50, or is 0.56 pp noise?

Both models were evaluated on the identical test partition, so the comparison is
paired at the level of individual images. McNemar's exact test on the discordant
pairs is the right instrument; a t-test over three seed means throws away that
pairing and has almost no power at n=3.

Reported per seed, then pooled, with a bootstrap interval on the accuracy
difference for effect size.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail.metrics import core_metrics  # noqa: E402
from nail.utils import df_to_markdown  # noqa: E402


def load(run: Path):
    z = np.load(run / "test_predictions.npz")
    return z["y"], z["probs"].argmax(1)


def mcnemar(a_ok: np.ndarray, b_ok: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar. b01 = a right/b wrong, b10 = a wrong/b right."""
    b01 = int(np.sum(a_ok & ~b_ok))
    b10 = int(np.sum(~a_ok & b_ok))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    p = float(st.binomtest(b01, n, 0.5).pvalue)
    return b01, b10, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-runs", type=Path,
                    default=ROOT / "results" / "runs_headline")
    ap.add_argument("--a-pattern", default="single_efficientnet_b0_s*")
    ap.add_argument("--a-name", default="EfficientNet-B0")
    ap.add_argument("--b-runs", type=Path, default=ROOT / "results" / "runs_sig")
    ap.add_argument("--b-pattern", default="single_resnet50_s*")
    ap.add_argument("--b-name", default="ResNet-50")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "17_model_comparison.md")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    a_dirs = sorted(d for d in args.a_runs.glob(args.a_pattern)
                    if (d / "test_predictions.npz").exists())
    b_dirs = sorted(d for d in args.b_runs.glob(args.b_pattern)
                    if (d / "test_predictions.npz").exists())
    if not a_dirs or not b_dirs:
        print("missing runs")
        return 1
    k = min(len(a_dirs), len(b_dirs))

    lines = [
        f"# {args.a_name} vs {args.b_name}",
        "",
        "Both trained on the identical split with a matched 40-epoch budget and",
        "evaluated on the identical test partition. Comparison is paired per image.",
        "",
        "## Per seed",
        "",
        "| seed | " + f"{args.a_name} acc | {args.b_name} acc | Δ | "
        "A✓B✗ | A✗B✓ | McNemar p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    all_a_ok, all_b_ok, deltas = [], [], []
    for i in range(k):
        ya, pa = load(a_dirs[i])
        yb, pb = load(b_dirs[i])
        if not np.array_equal(ya, yb):
            print(f"seed {i}: label vectors differ; splits are not aligned")
            return 1
        a_ok, b_ok = pa == ya, pb == yb
        b01, b10, p = mcnemar(a_ok, b_ok)
        d = (a_ok.mean() - b_ok.mean()) * 100
        deltas.append(d)
        all_a_ok.append(a_ok); all_b_ok.append(b_ok)
        lines.append(
            f"| {i} | {a_ok.mean()*100:.2f}% | {b_ok.mean()*100:.2f}% | "
            f"{d:+.2f} | {b01} | {b10} | {p:.4f} |"
        )

    A = np.concatenate(all_a_ok); B = np.concatenate(all_b_ok)
    b01, b10, p_pool = mcnemar(A, B)

    rng = np.random.default_rng(0)
    diffs = np.array([
        (A[idx].mean() - B[idx].mean()) * 100
        for idx in (rng.integers(0, len(A), len(A)) for _ in range(args.n_boot))
    ])
    lo, hi = np.quantile(diffs, [0.025, 0.975])

    lines += [
        "",
        "## Pooled over seeds",
        "",
        f"- {args.a_name}: **{A.mean()*100:.2f}%**",
        f"- {args.b_name}: **{B.mean()*100:.2f}%**",
        f"- Difference: **{(A.mean()-B.mean())*100:+.2f} pp**, "
        f"bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}]",
        f"- Discordant pairs: {args.a_name} correct only **{b01}**, "
        f"{args.b_name} correct only **{b10}**",
        f"- McNemar exact p = **{p_pool:.4f}**",
        "",
    ]

    if p_pool < 0.05:
        lines.append(
            f"The difference is statistically significant. {args.a_name} may be "
            f"reported as the stronger model on this dataset."
        )
    else:
        lines.append(
            f"**The difference is not statistically significant.** The two "
            f"architectures cannot be separated on this test set, and the paper "
            f"should say so rather than presenting {args.a_name} as superior. "
            f"Either would be a defensible choice; report the comparison as a tie "
            f"and pick on secondary grounds such as parameter count or latency."
        )
    n_items = len(all_a_ok[0]) if all_a_ok else 0
    lines += [
        "",
        f"Note that seeds are not independent replicates of the test set - the same",
        f"{n_items} items recur - so the pooled McNemar count is optimistic about",
        "sample size. The per-seed rows are the conservative reading.",
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
