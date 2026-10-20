"""Training loop with the ablation switches wired in as first-class arguments.

Everything the draft claims as a contribution - augmentation, class balancing,
layer4 unfreezing, TTA - is a flag here, so the ablation table can be produced by
sweeping flags under fixed seeds instead of being asserted.

Model selection happens on validation. The test split is never read by this
module; ``evaluate.py`` touches it once, at the end.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from . import CORRECTED_CLASSES, PAPER_CLASSES
from .metrics import core_metrics
from .models import build_model


def classes_in(manifest: pd.DataFrame) -> list[str]:
    """Class list for a manifest, ordered canonically so indices are stable.

    The taxonomy is a property of the data, not a global: the corrected 8-class
    and the draft's 6-class manifests both flow through the same code.
    """
    present = set(manifest["label"].dropna().unique())
    for canonical in (PAPER_CLASSES, CORRECTED_CLASSES):
        if present <= set(canonical):
            return [c for c in canonical if c in present]
    return sorted(present)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- config


@dataclass
class TrainConfig:
    model: str = "resnet18_paper"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    fc_lr: float = 1e-3
    backbone_lr: float = 5e-4
    weight_decay: float = 1e-3
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    patience: int = 7
    seed: int = 42
    # ablation switches
    augment: bool = True
    balancing: str = "sampler"  # "sampler" | "loss_weights" | "none"
    unfreeze_layer4: bool = True
    dropout: float = 0.5
    # runtime
    # Data loading, not the GPU, is the bottleneck here: JPEG decode plus PIL
    # transforms on a single thread held a 4050 to ~2.4 batches/sec against a
    # GPU-bound ceiling around 15-20. Workers are safe on Windows because every
    # entry point in scripts/ is guarded by `if __name__ == "__main__"`.
    num_workers: int = 4
    amp: bool = True
    select_on: str = "macro_f1"  # metric used for early stopping


# ---------------------------------------------------------------- data


def build_transforms(cfg: TrainConfig, train: bool) -> transforms.Compose:
    if not train or not cfg.augment:
        return transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class NailDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tf: transforms.Compose,
        class_to_idx: dict[str, int] | None = None,
    ):
        if class_to_idx is None:
            class_to_idx = {c: i for i, c in enumerate(classes_in(df))}
        self.class_to_idx = class_to_idx
        self.paths = df["path"].tolist()
        self.labels = [class_to_idx[c] for c in df["label"].tolist()]
        self.tf = tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.labels[i]


def make_loaders(
    manifest: pd.DataFrame, cfg: TrainConfig, classes: list[str] | None = None
) -> tuple[DataLoader, DataLoader, np.ndarray]:
    train_df = manifest[(manifest["split"] == "train") & manifest["label"].notna()]
    val_df = manifest[(manifest["split"] == "val") & manifest["label"].notna()]
    if train_df.empty or val_df.empty:
        raise ValueError("train or val split is empty; run the split step first")

    classes = classes or classes_in(manifest)
    c2i = {c: i for i, c in enumerate(classes)}

    train_ds = NailDataset(train_df, build_transforms(cfg, train=True), c2i)
    val_ds = NailDataset(val_df, build_transforms(cfg, train=False), c2i)

    counts = np.bincount(train_ds.labels, minlength=len(classes)).astype(float)
    counts[counts == 0] = 1.0

    sampler = None
    shuffle = True
    if cfg.balancing == "sampler":
        w = 1.0 / counts
        sample_weights = [w[y] for y in train_ds.labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle = False

    extra = (
        {"persistent_workers": True, "prefetch_factor": 4}
        if cfg.num_workers > 0
        else {}
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        **extra,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        **extra,
    )
    return train_loader, val_loader, counts


# ---------------------------------------------------------------- loop


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def collect_logits(
    model: nn.Module, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, targets = [], []
    for xb, yb in loader:
        out = model(xb.to(device, non_blocking=True))
        logits.append(out.float().cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(logits), np.concatenate(targets)


@dataclass
class TrainResult:
    best_epoch: int
    best_metric: float
    history: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)


def train_one(
    manifest: pd.DataFrame,
    cfg: TrainConfig,
    out_dir: Path,
    device: str | None = None,
) -> TrainResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    classes = classes_in(manifest)
    train_loader, val_loader, counts = make_loaders(manifest, cfg, classes)
    model = build_model(
        cfg.model,
        num_classes=len(classes),
        unfreeze_layer4=cfg.unfreeze_layer4,
        dropout=cfg.dropout,
    ).to(device)

    class_weights = None
    if cfg.balancing == "loss_weights":
        w = counts.sum() / (len(counts) * counts)
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=cfg.label_smoothing
    )

    groups = (
        model.param_groups(cfg.fc_lr, cfg.backbone_lr)
        if hasattr(model, "param_groups")
        else [{"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.fc_lr}]
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=max(1, cfg.warmup_epochs)
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs - cfg.warmup_epochs)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[max(1, cfg.warmup_epochs)]
    )

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")

    best_metric, best_epoch, bad_epochs = -np.inf, -1, 0
    history: list[dict] = []

    for epoch in range(cfg.epochs):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device == "cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)
        scheduler.step()

        val_logits, val_y = collect_logits(model, val_loader, device)
        val_pred = val_logits.argmax(1)
        m = core_metrics(val_y, val_pred)
        row = {"epoch": epoch, "train_loss": running / max(seen, 1), **m}
        history.append(row)

        score = m[cfg.select_on]
        if score > best_metric:
            best_metric, best_epoch, bad_epochs = score, epoch, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
            np.savez(out_dir / "val_logits.npz", logits=val_logits, y=val_y)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    result = TrainResult(
        best_epoch=best_epoch,
        best_metric=float(best_metric),
        history=history,
        config={**asdict(cfg), "classes": classes},
    )
    (out_dir / "history.json").write_text(json.dumps(asdict(result), indent=2))
    return result
