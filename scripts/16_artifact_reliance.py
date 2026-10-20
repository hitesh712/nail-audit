"""Is the model reading nail pathology, or augmentation artefacts?

Qualitative Grad-CAM panels showed two things at once: the source images carry
visible synthetic augmentation (rotation with black corner fill, composited
patches, injected noise), and attention repeatedly lands on image borders and
corners rather than on the nail.

Border activation alone proves nothing - Grad-CAM at a 7x7 feature map has poor
spatial resolution and border effects are common. So we test reliance directly, by
occluding regions and measuring what the prediction actually loses:

  border occlusion   mask an outer ring, where the rotation fill and frame sit.
  centre occlusion   mask a centred region of THE SAME AREA, where the nail sits.

If the model is reading pathology, occluding the centre should hurt far more than
occluding the border. If the two are comparable - or the border matters more - the
reported accuracy is substantially artefactual.

Equal-area matching is the point of the design: any occlusion damages accuracy, so
an unmatched test would prove nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail import PAPER_CLASSES  # noqa: E402
from nail.models import build_model  # noqa: E402
from nail.train import IMAGENET_MEAN, IMAGENET_STD, TrainConfig  # noqa: E402

C2I = {c: i for i, c in enumerate(PAPER_CLASSES)}


def border_mask(size: int, frac: float) -> torch.Tensor:
    """1 inside the kept centre, 0 in the outer ring of thickness `frac`*size/2."""
    m = torch.ones(size, size)
    t = int(round(size * frac / 2))
    if t > 0:
        m[:t, :] = 0; m[-t:, :] = 0; m[:, :t] = 0; m[:, -t:] = 0
    return m


def centre_mask_matching(size: int, ring_area: int) -> torch.Tensor:
    """0 inside a centred square of the same pixel area as the ring, 1 elsewhere."""
    side = int(round(np.sqrt(ring_area)))
    side = min(side, size)
    m = torch.ones(size, size)
    lo = (size - side) // 2
    m[lo:lo + side, lo:lo + side] = 0
    return m


@torch.no_grad()
def accuracy(model, tensors: torch.Tensor, labels: np.ndarray, device: str,
             mask: torch.Tensor | None = None, bs: int = 64) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(tensors), bs):
        xb = tensors[i:i + bs].clone()
        if mask is not None:
            xb = xb * mask.view(1, 1, *mask.shape)
        pred = model(xb.to(device)).argmax(1).cpu().numpy()
        correct += int((pred == labels[i:i + bs]).sum())
    return correct / len(labels)


def dark_fraction(t: torch.Tensor) -> float:
    """Share of pixels that are near-black in un-normalised space (rotation fill)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    raw = t * std + mean
    return float((raw.mean(0) < 0.06).float().mean().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path,
                    default=ROOT / "results" / "runs_headline" / "single_efficientnet_b0_s0")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_final.csv")
    ap.add_argument("--report", type=Path, default=ROOT / "results" / "23_artifacts.md")
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
    test = df[(df["split"] == "test") & df["label"].notna()]
    tensors = torch.stack([
        tf(Image.open(p).convert("RGB")) for p in test["path"]
    ])
    labels = np.array([C2I[c] for c in test["label"]])
    S = cfg.image_size

    dark = np.array([dark_fraction(t) for t in tensors])

    base = accuracy(model, tensors, labels, device)
    rows = []
    for frac in (0.10, 0.20, 0.30):
        bm = border_mask(S, frac)
        ring_area = int((bm == 0).sum().item())
        cm = centre_mask_matching(S, ring_area)
        rows.append({
            "occluded_frac_of_area": round(ring_area / (S * S), 3),
            "border_occluded_acc": accuracy(model, tensors, labels, device, bm),
            "centre_occluded_acc": accuracy(model, tensors, labels, device, cm),
        })
    res = pd.DataFrame(rows)
    res["border_drop_pp"] = (base - res["border_occluded_acc"]) * 100
    res["centre_drop_pp"] = (base - res["centre_occluded_acc"]) * 100

    lines = [
        "# Reliance on augmentation artefacts",
        "",
        f"Model `{cfg.model}`, test set n={len(labels)}. Unoccluded accuracy "
        f"**{base*100:.2f}%**.",
        "",
        "## Are the source images synthetically augmented?",
        "",
        f"- Mean share of near-black pixels per image: **{dark.mean()*100:.1f}%**",
        f"- Images with >5% near-black pixels: "
        f"**{(dark > 0.05).mean()*100:.1f}%** of the test set",
        f"- Images with >15% near-black pixels: **{(dark > 0.15).mean()*100:.1f}%**",
        "",
        "Large black regions are the fill left by rotating an image without cropping.",
        "Their prevalence indicates the distributed images are augmented derivatives",
        "rather than raw photographs.",
        "",
        "## Equal-area occlusion test",
        "",
        "| area occluded | border occluded | centre occluded | border drop | centre drop |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, r in res.iterrows():
        lines.append(
            f"| {r['occluded_frac_of_area']*100:.0f}% | "
            f"{r['border_occluded_acc']*100:.2f}% | {r['centre_occluded_acc']*100:.2f}% | "
            f"{r['border_drop_pp']:.2f} pp | {r['centre_drop_pp']:.2f} pp |"
        )

    worst = res.iloc[-1]
    if worst["border_drop_pp"] >= worst["centre_drop_pp"]:
        verdict = (
            "**Occluding the border costs at least as much as occluding an "
            "equal area at the centre, where the nail is.** The model is relying "
            "substantially on regions that contain no nail, which on this data means "
            "augmentation fill and framing. The reported accuracy cannot be "
            "attributed to nail pathology alone."
        )
    elif worst["centre_drop_pp"] > 3 * worst["border_drop_pp"]:
        verdict = (
            "Occluding the centre is far more damaging than occluding an equal "
            "area at the border, so the model is predominantly reading the nail "
            "region. Border attention in Grad-CAM overlays reflects the coarse "
            "resolution of the final feature map rather than genuine reliance."
        )
    else:
        verdict = (
            "Centre occlusion is more damaging than border occlusion, but not "
            "decisively so. The model reads the nail region primarily while drawing "
            "some signal from image framing. This should be disclosed, and the "
            "sensitivity reported."
        )

    lines += ["", "## Verdict", "", verdict, "",
              "Any occlusion damages accuracy, so the two conditions are matched on",
              "occluded area. The comparison between them, not either number alone,",
              "carries the evidence."]

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
