"""Inference on a held-out split, with optional test-time augmentation.

TTA is a flag rather than an assumption: the draft lists five TTA transforms and
never shows what they contribute. Run it both ways and report the difference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from .models import build_model
from .train import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NailDataset,
    TrainConfig,
    classes_in,
)


def _norm(size: int) -> list:
    return [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def tta_transforms(size: int = 224) -> list[transforms.Compose]:
    """Five deterministic views: identity, h-flip, v-flip, and +/-10 degrees."""
    return [
        transforms.Compose(_norm(size)),
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), *_norm(size)]),
        transforms.Compose([transforms.RandomVerticalFlip(p=1.0), *_norm(size)]),
        transforms.Compose([transforms.RandomRotation((10, 10)), *_norm(size)]),
        transforms.Compose([transforms.RandomRotation((-10, -10)), *_norm(size)]),
    ]


@torch.no_grad()
def predict_split(
    manifest: pd.DataFrame,
    checkpoint: Path,
    cfg: TrainConfig,
    split: str = "test",
    use_tta: bool = False,
    device: str | None = None,
    classes: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (logits, y_true, rows) for the requested split.

    With TTA the returned logits are averaged over the five views in probability
    space, then converted back to log space so downstream temperature scaling
    still behaves sensibly.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = manifest[(manifest["split"] == split) & manifest["label"].notna()].reset_index(
        drop=True
    )
    if rows.empty:
        raise ValueError(f"split '{split}' is empty")

    classes = classes or classes_in(manifest)
    c2i = {c: i for i, c in enumerate(classes)}

    model = build_model(
        cfg.model,
        num_classes=len(classes),
        unfreeze_layer4=cfg.unfreeze_layer4,
        dropout=cfg.dropout,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval().to(device)

    views = tta_transforms(cfg.image_size) if use_tta else [
        transforms.Compose(_norm(cfg.image_size))
    ]

    prob_sum = None
    y_true = None
    for tf in views:
        loader = DataLoader(
            NailDataset(rows, tf, c2i),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )
        logits, targets = [], []
        for xb, yb in loader:
            logits.append(model(xb.to(device)).float().cpu())
            targets.append(yb)
        lg = torch.cat(logits)
        probs = torch.softmax(lg, dim=1).numpy()
        prob_sum = probs if prob_sum is None else prob_sum + probs
        y_true = torch.cat(targets).numpy()

    mean_probs = prob_sum / len(views)
    logits_out = np.log(np.clip(mean_probs, 1e-12, None))
    return logits_out, y_true, rows
