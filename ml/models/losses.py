"""Loss functions for class-imbalanced skin lesion classification."""

from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    class_counts: Dict[str, int],
    class_order: List[str],
    power: float = 0.5,
) -> torch.Tensor:
    """Compute balanced class weights based on inverse frequency.

    weight_c = 1 / (N_c ** power), normalized so mean(weights) == 1.0.
    """
    weights = []
    for cls_name in class_order:
        cnt = max(1, class_counts.get(cls_name, 1))
        w = 1.0 / (cnt ** power)
        weights.append(w)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights so their average is 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor


class ClassWeightedFocalLoss(nn.Module):
    """Focal Loss with per-class alpha weighting to handle severe class imbalance."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:

        inputs: Raw logits (batch_size, num_classes)
        targets: Target class indices (batch_size,)
        """
        # Compute cross entropy with label smoothing
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        # Compute p_t (probability of ground truth class)
        p = torch.softmax(inputs, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Modulating focal factor: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
