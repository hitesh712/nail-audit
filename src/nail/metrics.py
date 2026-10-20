"""Evaluation: honest headline metrics, bootstrap intervals, calibration, abstention.

Two deliberate departures from the current draft:

* The headline number is **balanced accuracy / macro-F1**, not raw accuracy. On a
  validation set that is 53% one class, raw accuracy mostly measures the prior.
* Every reported number carries a bootstrap 95% interval. With ~9 images in the
  smallest class, a point estimate on its own is not interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


# ---------------------------------------------------------------- point metrics


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """One-vs-rest macro AUC, robust to classes absent from ``y_true``."""
    present = np.unique(y_true)
    if len(present) < 2:
        return float("nan")
    try:
        return float(
            roc_auc_score(
                y_true, y_prob[:, present], multi_class="ovr", average="macro", labels=present
            )
        )
    except ValueError:
        return float("nan")


def core_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_prob is not None:
        out["macro_auc"] = macro_auc(y_true, y_prob)
    return out


def per_class_table(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict[str, dict[str, float]]:
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        class_names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i in labels
    }


# ---------------------------------------------------------------- bootstrap


@dataclass
class Interval:
    point: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Interval]:
    """Percentile bootstrap over samples for every metric in ``core_metrics``."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = core_metrics(y_true, y_pred, y_prob)
    draws: dict[str, list[float]] = {k: [] for k in point}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = core_metrics(
            y_true[idx], y_pred[idx], None if y_prob is None else y_prob[idx]
        )
        for k, v in m.items():
            if not np.isnan(v):
                draws[k].append(v)

    out: dict[str, Interval] = {}
    for k, v in point.items():
        arr = np.asarray(draws[k], dtype=float)
        if arr.size == 0:
            out[k] = Interval(v, float("nan"), float("nan"))
        else:
            out[k] = Interval(
                v,
                float(np.quantile(arr, alpha / 2)),
                float(np.quantile(arr, 1 - alpha / 2)),
            )
    return out


# ---------------------------------------------------------------- calibration


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """Top-label ECE with equal-width confidence bins."""
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.mean()) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> dict[str, np.ndarray]:
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, accs, confs, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        centres.append((lo + hi) / 2)
        counts.append(int(mask.sum()))
        accs.append(float(correct[mask].mean()) if mask.sum() else np.nan)
        confs.append(float(conf[mask].mean()) if mask.sum() else np.nan)
    return {
        "bin_centre": np.array(centres),
        "accuracy": np.array(accs),
        "confidence": np.array(confs),
        "count": np.array(counts),
    }


def fit_temperature(logits: np.ndarray, y_true: np.ndarray, max_iter: int = 200) -> float:
    """Single-parameter temperature scaling, fitted on validation logits by NLL."""
    import torch

    lg = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y_true, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(lg / torch.exp(log_t), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).detach().item())


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- abstention


@dataclass
class RiskCoverage:
    coverage: np.ndarray
    risk: np.ndarray
    thresholds: np.ndarray
    aurc: float
    at_threshold: dict[float, dict[str, float]] = field(default_factory=dict)

    def to_markdown(self, thresholds: tuple[float, ...] = (0.5, 0.7, 0.9)) -> str:
        lines = [
            "## Risk-coverage (selective prediction)",
            "",
            f"- AURC (area under risk-coverage curve, lower is better): **{self.aurc:.4f}**",
            "",
            "| confidence threshold | coverage | selective accuracy |",
            "| --- | --- | --- |",
        ]
        for t in thresholds:
            if t in self.at_threshold:
                d = self.at_threshold[t]
                lines.append(
                    f"| {t:.2f} | {d['coverage']*100:.1f}% | {d['selective_accuracy']*100:.2f}% |"
                )
        lines += [
            "",
            "Reporting accuracy after a confidence filter without stating coverage "
            "overstates performance. The pair must always be quoted together.",
        ]
        return "\n".join(lines)


def risk_coverage_curve(
    y_true: np.ndarray, y_prob: np.ndarray, report_at: tuple[float, ...] = (0.5, 0.7, 0.9)
) -> RiskCoverage:
    """Selective-prediction curve: what accuracy costs how much coverage."""
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    order = np.argsort(-conf)
    correct_sorted = correct[order]
    n = len(y_true)

    coverage = np.arange(1, n + 1) / n
    cum_correct = np.cumsum(correct_sorted)
    selective_acc = cum_correct / np.arange(1, n + 1)
    risk = 1.0 - selective_acc
    aurc = float(np.trapz(risk, coverage))

    at: dict[float, dict[str, float]] = {}
    for t in report_at:
        keep = conf >= t
        at[t] = {
            "coverage": float(keep.mean()),
            "selective_accuracy": float(correct[keep].mean()) if keep.sum() else float("nan"),
            "n_kept": int(keep.sum()),
        }

    return RiskCoverage(
        coverage=coverage,
        risk=risk,
        thresholds=conf[order],
        aurc=aurc,
        at_threshold=at,
    )


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
