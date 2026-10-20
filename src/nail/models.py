"""Backbone zoo.

``resnet18_paper`` reproduces the architecture described in the draft exactly
(layer4 + a 512-unit head with dropout 0.5, everything else frozen) so that the
comparison table compares architectures on *our* split rather than quoting other
papers' numbers on other people's data.

``*_linear`` variants freeze the trunk entirely and train only the classifier.
That is the floor: any fine-tuned model that fails to beat its own linear probe
by a clear margin has not learned anything the ImageNet features did not already
provide.
"""

from __future__ import annotations

import torch
import torch.nn as nn

TIMM_BACKBONES = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3_large": "mobilenetv3_large_100",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "vit_b16": "vit_base_patch16_224",
}


class PaperResNet18(nn.Module):
    """ResNet-18 with the draft's head: Linear(512) -> ReLU -> Dropout(0.5) -> Linear.

    Only ``layer4`` and the head receive gradients, matching Section IV-B.
    """

    def __init__(self, num_classes: int = 6, dropout: float = 0.5, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self.set_trainable(unfreeze_layer4=True)

    def set_trainable(self, unfreeze_layer4: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.fc.parameters():
            p.requires_grad = True
        if unfreeze_layer4:
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

    def param_groups(self, fc_lr: float = 1e-3, layer4_lr: float = 5e-4) -> list[dict]:
        """Layer-specific learning rates, as in Table IV."""
        return [
            {"params": [p for p in self.backbone.fc.parameters() if p.requires_grad],
             "lr": fc_lr},
            {"params": [p for p in self.backbone.layer4.parameters() if p.requires_grad],
             "lr": layer4_lr},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TimmClassifier(nn.Module):
    """Any timm backbone with a fresh classifier head."""

    def __init__(
        self,
        name: str,
        num_classes: int = 6,
        pretrained: bool = True,
        freeze_trunk: bool = False,
        dropout: float = 0.0,
        unfreeze_last_block: bool = True,
    ):
        super().__init__()
        import timm

        self.model = timm.create_model(
            TIMM_BACKBONES.get(name, name),
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
        self.freeze_trunk = freeze_trunk
        if freeze_trunk:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.get_classifier().parameters():
                p.requires_grad = True
        elif not unfreeze_last_block:
            # Ablation arm: head only, trunk frozen including the last block.
            # Mirrors what `unfreeze_layer4=False` means for PaperResNet18 so the
            # factor is comparable across architectures.
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.get_classifier().parameters():
                p.requires_grad = True
        # unfreeze_last_block=True with freeze_trunk=False is a full fine-tune,
        # which is the default and what the backbone comparison measured.
        self.last_block = self._find_last_block()

    def _find_last_block(self) -> nn.Module | None:
        """The final residual/conv stage, where one exists (resnet family)."""
        for attr in ("layer4", "stages"):
            block = getattr(self.model, attr, None)
            if block is not None:
                return block[-1] if attr == "stages" else block
        return None

    def param_groups(self, fc_lr: float = 1e-3, backbone_lr: float = 5e-4) -> list[dict]:
        head_ids = {id(p) for p in self.model.get_classifier().parameters()}
        head = [p for p in self.model.parameters() if id(p) in head_ids and p.requires_grad]
        trunk = [
            p for p in self.model.parameters() if id(p) not in head_ids and p.requires_grad
        ]
        groups = [{"params": head, "lr": fc_lr}]
        if trunk:
            groups.append({"params": trunk, "lr": backbone_lr})
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(
    name: str,
    num_classes: int = 6,
    pretrained: bool = True,
    unfreeze_layer4: bool = True,
    dropout: float = 0.5,
) -> nn.Module:
    """Factory. ``name`` may be ``resnet18_paper``, any key in TIMM_BACKBONES,
    or ``<backbone>_linear`` for a frozen-trunk linear probe."""
    if name == "resnet18_paper":
        model = PaperResNet18(num_classes, dropout=dropout, pretrained=pretrained)
        model.set_trainable(unfreeze_layer4=unfreeze_layer4)
        return model

    if name.endswith("_linear"):
        return TimmClassifier(
            name[: -len("_linear")],
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_trunk=True,
        )

    return TimmClassifier(
        name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        unfreeze_last_block=unfreeze_layer4,
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
