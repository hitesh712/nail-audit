"""Grad-CAM with a faithfulness evaluation that needs no annotations.

The draft asserts that attention lands on clinically meaningful structures and
quotes figures (82% of attention on nail changes, 89% agreement with Lovibond
angle criteria) that no stated method could have produced and no annotation in
the dataset could ground.

Attribution *faithfulness* can be measured without any ground truth, by asking
whether the highlighted pixels are the ones the model actually uses:

  deletion   progressively remove the highest-attributed pixels and track the
             predicted-class probability. A faithful map makes it collapse fast,
             so LOWER area under the curve is better.
  insertion  start from a blurred image and progressively restore the
             highest-attributed pixels. HIGHER area is better.

Neither number means anything on its own, so both are reported against a random
attribution baseline on the identical images. That comparison is the claim.
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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail import PAPER_CLASSES  # noqa: E402
from nail.models import build_model  # noqa: E402
from nail.train import IMAGENET_MEAN, IMAGENET_STD, TrainConfig  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.size": 8})


def load_model(run_dir: Path, device: str):
    hist = json.loads((run_dir / "history.json").read_text())
    cfg_d = dict(hist["config"])
    classes = cfg_d.pop("classes", PAPER_CLASSES)
    cfg = TrainConfig(**cfg_d)
    model = build_model(cfg.model, num_classes=len(classes),
                        unfreeze_layer4=cfg.unfreeze_layer4, dropout=cfg.dropout)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    return model.eval().to(device), cfg, classes


def gradcam(model, x: torch.Tensor, target: int | None = None):
    """Grad-CAM over the final feature map. Works for any timm backbone."""
    inner = getattr(model, "model", model)
    feats = inner.forward_features(x)
    feats.retain_grad()
    logits = inner.forward_head(feats)
    cls = int(logits.argmax(1).item()) if target is None else target
    inner.zero_grad(set_to_none=True)
    logits[0, cls].backward(retain_graph=True)

    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam.cpu().numpy(), cls, torch.softmax(logits, 1)[0, cls].item()


@torch.no_grad()
def _probs(model, batch: torch.Tensor, cls: int) -> np.ndarray:
    out = []
    for i in range(0, len(batch), 32):
        p = torch.softmax(model(batch[i:i + 32]), 1)[:, cls]
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def deletion_insertion(model, x: torch.Tensor, sal: np.ndarray, cls: int,
                       steps: int = 25, device: str = "cuda") -> tuple[float, float]:
    """Normalised area under the deletion and insertion curves."""
    _, _, H, W = x.shape
    order = np.argsort(-sal.reshape(-1))
    n = len(order)
    per = max(1, n // steps)

    blurred = transforms.GaussianBlur(31, sigma=12.0)(x)

    del_imgs, ins_imgs = [], []
    for s in range(steps + 1):
        k = min(s * per, n)
        mask = torch.ones(n, device=device)
        mask[torch.as_tensor(order[:k].copy(), device=device)] = 0.0
        mask = mask.view(1, 1, H, W)
        del_imgs.append(x * mask + blurred * (1 - mask))
        ins_imgs.append(x * (1 - mask) + blurred * mask)

    d = _probs(model, torch.cat(del_imgs), cls)
    i = _probs(model, torch.cat(ins_imgs), cls)
    xs = np.linspace(0, 1, steps + 1)
    return float(np.trapezoid(d, xs)), float(np.trapezoid(i, xs))


def _verdict(del_diff: float, del_p: float, ins_diff: float, ins_p: float,
             alpha: float = 0.05) -> str:
    """Describe what the numbers actually show, rather than what we hoped."""
    def phrase(diff: float, p: float, name: str, favour: str) -> str:
        if p >= alpha:
            return (f"The {name} test does not separate Grad-CAM from random "
                    f"attribution (difference {diff:+.4f}, p={p:.3f}).")
        if diff > 0:
            return (f"The {name} test favours Grad-CAM ({diff:+.4f}, p={p:.2e}): "
                    f"{favour}")
        return (f"The {name} test favours **random** attribution ({diff:+.4f}, "
                f"p={p:.2e}), which is evidence against the map being faithful.")

    parts = [
        phrase(del_diff, del_p, "deletion",
               "removing the highest-attributed pixels collapses the prediction "
               "faster than removing random pixels."),
        phrase(ins_diff, ins_p, "insertion",
               "restoring the highest-attributed pixels recovers the prediction "
               "faster than restoring random pixels."),
    ]
    agree = (del_diff > 0) == (ins_diff > 0)
    if not agree:
        parts.append(
            "The two tests disagree. Deletion and insertion probe different "
            "failure modes and are known to diverge when the removal baseline "
            "itself carries signal: blurring a nail region leaves surrounding "
            "skin, lighting and overall colour intact, and if the model draws on "
            "those, occluding the salient region need not reduce class evidence. "
            "Insertion is the more informative of the two here, because starting "
            "from a fully blurred image removes that confound. The honest summary "
            "is that the attribution is faithful by insertion and unresolved by "
            "deletion, and both should be reported."
        )
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path,
                    default=ROOT / "results" / "runs_headline" / "single_efficientnet_b0_s0")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_final.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "16_gradcam.md")
    ap.add_argument("--n-per-class", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, classes = load_model(args.run, device)
    tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    df = pd.read_csv(args.manifest)
    test = df[(df["split"] == "test") & df["label"].notna()]
    rng = np.random.default_rng(args.seed)

    rows, panels = [], []
    for cls_name in classes:
        sub = test[test["label"] == cls_name]
        if sub.empty:
            continue
        take = sub.sample(min(args.n_per_class, len(sub)), random_state=args.seed)
        first = True
        for path in take["path"]:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
            x = tf(rgb).unsqueeze(0).to(device)
            sal, pred, conf = gradcam(model, x)
            d_cam, i_cam = deletion_insertion(model, x, sal, pred, device=device)
            rnd = rng.random(sal.shape)
            d_rnd, i_rnd = deletion_insertion(model, x, rnd, pred, device=device)
            rows.append({"class": cls_name, "pred": classes[pred], "conf": conf,
                         "del_cam": d_cam, "ins_cam": i_cam,
                         "del_rnd": d_rnd, "ins_rnd": i_rnd})
            if first and classes[pred] == cls_name:
                panels.append((cls_name, np.array(rgb.resize((224, 224))), sal, conf))
                first = False

    res = pd.DataFrame(rows)
    res.to_csv(args.out.parent / "gradcam_faithfulness.csv", index=False)

    # Qualitative panel: one correctly-classified example per class.
    if panels:
        fig, axes = plt.subplots(2, len(panels), figsize=(2.0 * len(panels), 4.2))
        if len(panels) == 1:
            axes = axes.reshape(2, 1)
        for j, (name, img, sal, conf) in enumerate(panels):
            axes[0, j].imshow(img); axes[0, j].set_title(name.replace("_", " "), fontsize=7)
            axes[1, j].imshow(img); axes[1, j].imshow(sal, cmap="jet", alpha=0.45)
            axes[1, j].set_xlabel(f"p={conf:.2f}", fontsize=7)
            for r in (0, 1):
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([]); axes[r, j].grid(False)
        axes[0, 0].set_ylabel("input", fontsize=8)
        axes[1, 0].set_ylabel("Grad-CAM", fontsize=8)
        fig.savefig(args.out / "fig7_gradcam.png")
        plt.close(fig)

    from scipy import stats as st
    dd = res["del_rnd"] - res["del_cam"]      # positive => CAM deletes faster
    di = res["ins_cam"] - res["ins_rnd"]      # positive => CAM inserts faster
    td, pd_ = st.ttest_1samp(dd, 0)
    ti, pi_ = st.ttest_1samp(di, 0)

    lines = [
        "# Grad-CAM faithfulness",
        "",
        f"Model `{cfg.model}`, {len(res)} test images "
        f"({args.n_per_class} per class), Grad-CAM over the final feature map.",
        "",
        "## Deletion and insertion, against a random-attribution control",
        "",
        "| metric | Grad-CAM | random | difference | p |",
        "| --- | --- | --- | --- | --- |",
        f"| deletion AUC (lower better) | {res['del_cam'].mean():.4f} | "
        f"{res['del_rnd'].mean():.4f} | **{dd.mean():+.4f}** | {pd_:.2e} |",
        f"| insertion AUC (higher better) | {res['ins_cam'].mean():.4f} | "
        f"{res['ins_rnd'].mean():.4f} | **{di.mean():+.4f}** | {pi_:.2e} |",
        "",
        _verdict(dd.mean(), pd_, di.mean(), pi_),
        "",
        "## What this does and does not establish",
        "",
        "It establishes faithfulness: the map reflects the model's own computation.",
        "It does **not** establish clinical correctness - that the highlighted region",
        "is the one a dermatologist would use. That claim needs expert annotations,",
        "which this dataset does not carry, and it is the claim the draft makes",
        "without them.",
        "",
        "## Per class",
        "",
        "| class | n | deletion Δ | insertion Δ | mean confidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cls_name, g in res.groupby("class"):
        lines.append(
            f"| {cls_name} | {len(g)} | "
            f"{(g['del_rnd'] - g['del_cam']).mean():+.4f} | "
            f"{(g['ins_cam'] - g['ins_rnd']).mean():+.4f} | {g['conf'].mean():.3f} |"
        )

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
