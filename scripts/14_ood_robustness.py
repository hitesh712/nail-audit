"""Does the model know when it is looking at unfamiliar data?

No public dataset covers this study's six classes, so true external validation is
not available (see 19_external_search.md). The nearest achievable substitute is an
out-of-distribution check: feed nail photographs from a wholly independent
collection and ask whether confidence falls and the abstention mechanism engages,
or whether the model is confidently wrong.

A clinically deployed screening tool meets unfamiliar input constantly. A model
that answers such input with high confidence is more dangerous than one that is
merely inaccurate, so this is worth measuring even though it is not a substitute
for labelled external data.

The OOD pool is first deduplicated against our own training data - an overlap
would silently turn an OOD test into an in-distribution one.
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
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail import PAPER_CLASSES  # noqa: E402
from nail.dedup import _hamming_matrix, compute_phashes  # noqa: E402
from nail.models import build_model  # noqa: E402
from nail.train import IMAGENET_MEAN, IMAGENET_STD, TrainConfig  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.size": 8, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


class Plain(Dataset):
    def __init__(self, paths: list[str], tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), i


@torch.no_grad()
def predict(model, paths: list[str], tf, device: str, bs: int = 32) -> np.ndarray:
    loader = DataLoader(Plain(paths, tf), batch_size=bs, num_workers=4, shuffle=False)
    out = np.zeros((len(paths), len(PAPER_CLASSES)), dtype=np.float32)
    for xb, idx in loader:
        p = torch.softmax(model(xb.to(device)), 1).cpu().numpy()
        out[idx.numpy()] = p
    return out


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path,
                    default=ROOT / "results" / "runs_headline" / "single_efficientnet_b0_s0")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_final.csv")
    ap.add_argument("--ood-root", type=Path, default=ROOT / "data" / "ood" / "nailseg")
    ap.add_argument("--n-ood", type=int, default=1200)
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "20_ood.md")
    ap.add_argument("--figdir", type=Path, default=ROOT / "results" / "figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hist = json.loads((args.run / "history.json").read_text())
    cfg_d = dict(hist["config"]); cfg_d.pop("classes", None)
    cfg = TrainConfig(**cfg_d)
    model = build_model(cfg.model, num_classes=len(PAPER_CLASSES),
                        unfreeze_layer4=cfg.unfreeze_layer4, dropout=cfg.dropout)
    model.load_state_dict(torch.load(args.run / "best.pt", map_location="cpu"))
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    df = pd.read_csv(args.manifest)
    id_test = df[(df["split"] == "test") & df["label"].notna()]
    train_paths = df[df["split"] == "train"]["path"].tolist()

    # Only the photographs, never the segmentation masks.
    ood_all = [
        str(p) for p in args.ood_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and "mask" not in str(p).lower()
    ]
    rng = np.random.default_rng(args.seed)
    if len(ood_all) > args.n_ood:
        ood_all = list(rng.choice(ood_all, args.n_ood, replace=False))
    print(f"{len(ood_all)} candidate OOD images")

    # Guard against the OOD pool overlapping our training data.
    h_ood = compute_phashes(ood_all)
    h_tr = compute_phashes(train_paths)
    keep = []
    for s in range(0, len(h_ood), 256):
        d = _hamming_matrix(h_ood[s:s + 256], h_tr).min(axis=1)
        keep.extend((d > 6).tolist())
    keep = np.array(keep)
    n_overlap = int((~keep).sum())
    ood = [p for p, k in zip(ood_all, keep) if k]
    print(f"removed {n_overlap} overlapping with training; {len(ood)} remain")

    p_id = predict(model, id_test["path"].tolist(), tf, device)
    p_ood = predict(model, ood, tf, device)

    c_id, c_ood = p_id.max(1), p_ood.max(1)
    e_id, e_ood = entropy(p_id), entropy(p_ood)

    y = np.r_[np.ones(len(c_id)), np.zeros(len(c_ood))]
    auroc_conf = roc_auc_score(y, np.r_[c_id, c_ood])
    auroc_ent = roc_auc_score(y, -np.r_[e_id, e_ood])

    rows = []
    for t in (0.5, 0.7, 0.9, 0.95):
        rows.append({
            "threshold": t,
            "id_coverage_pct": round(100 * float((c_id >= t).mean()), 1),
            "ood_accepted_pct": round(100 * float((c_ood >= t).mean()), 1),
        })
    cov = pd.DataFrame(rows).set_index("threshold")

    pred_dist = pd.Series(
        [PAPER_CLASSES[i] for i in p_ood.argmax(1)]
    ).value_counts(normalize=True).mul(100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    bins = np.linspace(0, 1, 41)
    axes[0].hist(c_id, bins=bins, alpha=0.6, density=True, label="in-distribution test")
    axes[0].hist(c_ood, bins=bins, alpha=0.6, density=True, label="independent nails (OOD)")
    axes[0].set_xlabel("max softmax confidence"); axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7)
    axes[1].hist(e_id, bins=40, alpha=0.6, density=True, label="in-distribution test")
    axes[1].hist(e_ood, bins=40, alpha=0.6, density=True, label="independent nails (OOD)")
    axes[1].set_xlabel("predictive entropy (nats)"); axes[1].set_ylabel("density")
    axes[1].legend(fontsize=7)
    args.figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figdir / "fig8_ood.png")
    plt.close(fig)

    lines = [
        "# Out-of-distribution behaviour",
        "",
        f"Model `{cfg.model}` (seed {cfg.seed}). In-distribution: {len(c_id)} held-out",
        f"test images. OOD: {len(ood)} nail photographs from an independent public",
        f"collection (nail segmentation dataset, CC0), after removing {n_overlap}",
        "images that perceptually matched our training data.",
        "",
        "## Does confidence fall on unfamiliar data?",
        "",
        "| | in-distribution | OOD |",
        "| --- | --- | --- |",
        f"| mean max-softmax | {c_id.mean():.3f} | {c_ood.mean():.3f} |",
        f"| median max-softmax | {np.median(c_id):.3f} | {np.median(c_ood):.3f} |",
        f"| mean entropy (nats) | {e_id.mean():.3f} | {e_ood.mean():.3f} |",
        "",
        f"- OOD detection AUROC, max-softmax: **{auroc_conf:.3f}**",
        f"- OOD detection AUROC, entropy: **{auroc_ent:.3f}**",
        "",
        "## What the abstention threshold actually does",
        "",
        "| threshold | in-distribution retained | OOD wrongly accepted |",
        "| --- | --- | --- |",
    ]
    for t, r in cov.iterrows():
        lines.append(
            f"| {t:.2f} | {r['id_coverage_pct']:.1f}% | {r['ood_accepted_pct']:.1f}% |"
        )

    lines += [
        "",
        "## What the model predicts on OOD input",
        "",
        "| class | share of OOD predictions |",
        "| --- | --- |",
    ]
    for cls, pct in pred_dist.items():
        lines.append(f"| {cls} | {pct:.1f}% |")

    verdict = (
        "The model separates familiar from unfamiliar input well"
        if auroc_conf >= 0.8 else
        "The model separates familiar from unfamiliar input only weakly"
        if auroc_conf >= 0.65 else
        "**The model cannot tell familiar from unfamiliar input**"
    )
    lines += [
        "",
        "## Reading this",
        "",
        f"{verdict} (AUROC {auroc_conf:.3f} from max-softmax alone). "
        "A confidence threshold set for in-distribution abstention will admit the "
        f"OOD share shown above; at the draft's 0.7 it admits "
        f"{cov.loc[0.7, 'ood_accepted_pct']:.1f}% of them.",
        "",
        "Two caveats. These OOD images are unlabelled, so no accuracy can be computed",
        "on them - only confidence behaviour. And they are still nails, making this a",
        "*near*-OOD test; a deployed system would also meet non-nail input, which",
        "would be a harder test than this one.",
        "",
        "This is not a substitute for labelled external validation. It is the",
        "strongest evidence available given that no public dataset covers these six",
        "classes.",
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
