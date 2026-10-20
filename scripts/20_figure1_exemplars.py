"""Figure 1: class exemplars.

Selection is not arbitrary. Many images in this collection carry heavy rotation
fill, and a figure made from those would misrepresent the data to a reader while
also being hard to read. We therefore rank candidates by three criteria and pick
the best per class:

  * low near-black fraction  - shows an actual nail rather than a black frame;
  * correctly classified     - no point illustrating a class with a failure case;
  * high confidence          - a representative rather than a borderline example.

The selection rule is reported in the caption so the figure is not mistaken for a
random sample. Images come from the test partition only, so nothing shown here
influenced training.
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
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nail import PAPER_CLASSES  # noqa: E402
from nail.models import build_model  # noqa: E402
from nail.train import IMAGENET_MEAN, IMAGENET_STD, TrainConfig  # noqa: E402

PRETTY = {
    "Acral_Lentiginous_Melanoma": "Acral lentiginous\nmelanoma",
    "Blue_Finger": "Blue finger",
    "Clubbing": "Clubbing",
    "Healthy_Nail": "Healthy nail",
    "Onychogryphosis": "Onychogryphosis",
    "Pitting": "Pitting",
}
C2I = {c: i for i, c in enumerate(PAPER_CLASSES)}


def black_fraction(img: Image.Image) -> float:
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return float((a.mean(axis=2) < 0.06).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path,
                    default=ROOT / "results" / "runs_headline" / "single_efficientnet_b0_s0")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest_final.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures" / "fig1_exemplars.png")
    ap.add_argument("--per-class", type=int, default=3, help="columns per class")
    ap.add_argument("--candidates", type=int, default=40)
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

    chosen: dict[str, list[tuple[str, float, float]]] = {}
    for cls in PAPER_CLASSES:
        sub = test[test["label"] == cls]
        scored = []
        for p in sub["path"].tolist():
            with Image.open(p) as im:
                rgb = im.convert("RGB")
                bf = black_fraction(rgb)
                x = tf(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(x), 1)[0]
            pred = int(prob.argmax().item())
            scored.append((p, bf, float(prob[C2I[cls]].item()), pred == C2I[cls]))
        ok = [s for s in scored if s[3]]
        # Prefer little black fill, then high confidence.
        ok.sort(key=lambda s: (s[1], -s[2]))
        chosen[cls] = [(p, bf, cf) for p, bf, cf, _ in ok[: args.per_class]]
        print(f"{cls}: {len(ok)}/{len(scored)} correct; "
              f"best black-fill {ok[0][1]*100:.1f}%, conf {ok[0][2]:.3f}")

    # Classes as columns: landscape, fits a double-column journal figure.
    ncol = len(PAPER_CLASSES)
    nrow = args.per_class
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.45 * ncol, 1.60 * nrow))
    if nrow == 1:
        axes = axes.reshape(1, -1)
    for c, cls in enumerate(PAPER_CLASSES):
        for r in range(nrow):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("#999999")
            if r < len(chosen[cls]):
                p, bf, cf = chosen[cls][r]
                with Image.open(p) as im:
                    ax.imshow(im.convert("RGB").resize((224, 224)))
            if r == 0:
                ax.set_title(PRETTY[cls].replace("\n", " "), fontsize=7.2, pad=4)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Fig. 1. Representative images from each of the six classes, drawn from the "
        "test partition. For each class the images shown are those carrying the least "
        "rotation fill among correctly classified examples, so that nail morphology "
        "rather than augmentation artefact is visible; they are not a random sample. "
        "Characteristic features are nonetheless apparent: the longitudinal "
        "pigmented band of acral lentiginous melanoma, the bulbous distal phalanx of "
        "clubbing, and the punctate depressions of pitting. The figure also "
        "illustrates the synthetic augmentation present throughout the distributed "
        "collection - visible rotation, composited rectangular patches, and colour "
        "and noise perturbation - which motivates the occlusion control reported in "
        "Section 5.6. Images are from the Nail Disease Detection dataset [9], "
        "reproduced under CC BY-NC-SA 4.0."
    )
    (args.out.parent / "fig1_caption.txt").write_text(caption, encoding="utf-8")
    print(f"\n{args.out}")
    print(f"\nCaption written to {args.out.parent / 'fig1_caption.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
