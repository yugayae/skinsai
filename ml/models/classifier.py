"""Skin lesion neural network classifier architecture with calibration."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SkinClassifier(nn.Module):
    """Deep CNN classifier for skin lesion diagnosis with temperature calibration."""

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 6,
        pretrained: bool = True,
        drop_rate: float = 0.3,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Create backbone via timm
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Temperature scaling parameter (initialized to 1.0)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unscaled raw logits."""
        return self.model(x)

    def forward_calibrated(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning temperature-scaled logits."""
        logits = self.forward(x)
        return logits / torch.clamp(self.temperature, min=0.01)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning calibrated class probabilities."""
        calibrated_logits = self.forward_calibrated(x)
        return F.softmax(calibrated_logits, dim=-1)

    def calibrate_temperature(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        max_iter: int = 50,
    ) -> float:
        """Optimize temperature parameter T on validation set using NLL loss."""
        dev = self.temperature.device
        val_logits = val_logits.to(dev)
        val_targets = val_targets.to(dev)
        self.temperature.requires_grad = True
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            scaled_logits = val_logits / torch.clamp(self.temperature, min=0.01)
            loss = nll_criterion(scaled_logits, val_targets)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        with torch.no_grad():
            self.temperature.clamp_(min=0.05, max=10.0)
        return float(self.temperature.item())
