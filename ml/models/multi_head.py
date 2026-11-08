"""Multi-Head Skin Lesion Classifier for decoupled clinical triage and differential subtyping.

Architecture:
1. Shared Vision Backbone (e.g. Meta DINOv2 Small, EfficientNet-B0):
   Extracts high-dimensional latent representation z = f(x).
2. Head 1 (Binary Oncology Triage):
   Predicts Malignant (BCC, SCC, MEL) vs Benign (NEV, SEK, general benign lesions).
   Optimized for maximum sensitivity (minimal false negatives on cancer).
3. Head 2 (Differential Subtyping):
   Predicts the 6 canonical skin disease classes (ACK, BCC, MEL, NEV, SCC, SEK).
4. Masked Multi-Task Loss:
   Supports training Head 1 on weakly-labeled / general benign images (e.g. 1,219 ISIC
   benign close-up images) without forcing arbitrary 6-class subtype targets.
5. Independent Temperature Calibration:
   Provides independent temperature parameters for Head 1 and Head 2.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CLASS_ORDER = ["ack", "bcc", "mel", "nev", "scc", "sek"]
MALIGNANT_CLASSES = {"bcc", "scc", "mel"}
BENIGN_CLASSES = {"nev", "sek"}
PREMALIGNANT_CLASSES = {"ack"}


class MaskedClassWeightedFocalLoss(nn.Module):
    """Focal Loss with per-class alpha weighting that safely handles ignore_index=-1."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        ignore_index: int = -1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute masked class-weighted focal loss.

        Args:
            inputs: Logits of shape (batch_size, num_classes).
            targets: Target class indices of shape (batch_size,).

        Returns:
            Computed scalar loss tensor.
        """
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        valid_inputs = inputs[valid_mask]
        valid_targets = targets[valid_mask]

        ce_loss = F.cross_entropy(
            valid_inputs,
            valid_targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        p = torch.softmax(valid_inputs, dim=-1)
        p_t = p.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_dev = self.alpha.to(valid_inputs.device)
            alpha_t = alpha_dev.gather(0, valid_targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiHeadLoss(nn.Module):
    """Multi-task loss combining Binary Triage Loss and Differential Subtype Loss."""

    def __init__(
        self,
        triage_weight: float = 1.0,
        subtype_weight: float = 1.0,
        triage_criterion: Optional[nn.Module] = None,
        subtype_criterion: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.triage_weight = triage_weight
        self.subtype_weight = subtype_weight
        self.triage_loss = triage_criterion or MaskedClassWeightedFocalLoss(gamma=2.0)
        self.subtype_loss = subtype_criterion or MaskedClassWeightedFocalLoss(gamma=2.0)

    def forward(
        self,
        triage_logits: torch.Tensor,
        subtype_logits: torch.Tensor,
        triage_targets: torch.Tensor,
        subtype_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute weighted combination of triage and subtype losses.

        Returns:
            Tuple of (total_loss, loss_triage, loss_subtype).
        """
        loss_t = self.triage_loss(triage_logits, triage_targets)
        loss_s = self.subtype_loss(subtype_logits, subtype_targets)

        total_loss = self.triage_weight * loss_t + self.subtype_weight * loss_s
        return total_loss, loss_t, loss_s


class MultiHeadProbe(nn.Module):
    """Lightweight decoupled multi-head classifier operating on pre-extracted embeddings.

    Ideal for fast training/fine-tuning on top of frozen foundation backbones.
    """

    def __init__(
        self,
        in_features: int = 384,
        num_subtypes: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_subtypes = num_subtypes

        # Shared projection trunk
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Head 1: Binary Oncology Triage (2 logits: [Benign, Malignant])
        self.triage_head = nn.Linear(hidden_dim, 2)

        # Head 2: 6-Class Differential Subtypes
        self.subtype_head = nn.Linear(hidden_dim, num_subtypes)

        # Independent temperature scaling parameters
        self.temperature_triage = nn.Parameter(torch.ones(1) * 1.0)
        self.temperature_subtype = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (triage_logits, subtype_logits)."""
        h = self.trunk(features)
        triage_logits = self.triage_head(h)
        subtype_logits = self.subtype_head(h)
        return triage_logits, subtype_logits

    def forward_calibrated(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with temperature scaling applied to both heads."""
        triage_logits, subtype_logits = self.forward(features)
        t_triage = torch.clamp(self.temperature_triage, min=0.01)
        t_subtype = torch.clamp(self.temperature_subtype, min=0.01)
        return triage_logits / t_triage, subtype_logits / t_subtype

    def predict_proba(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (triage_probs, subtype_probs)."""
        cal_triage, cal_subtype = self.forward_calibrated(features)
        triage_probs = F.softmax(cal_triage, dim=-1)
        subtype_probs = F.softmax(cal_subtype, dim=-1)
        return triage_probs, subtype_probs

    def calibrate_temperatures(
        self,
        val_features: torch.Tensor,
        val_triage_targets: torch.Tensor,
        val_subtype_targets: torch.Tensor,
        max_iter: int = 50,
    ) -> Tuple[float, float]:
        """Independently calibrate temperatures on validation data using L-BFGS."""
        with torch.no_grad():
            raw_triage, raw_subtype = self.forward(val_features)

        # Calibrate Head 1 (Triage) on samples with valid triage target
        valid_t = val_triage_targets >= 0
        if valid_t.any():
            self.temperature_triage.requires_grad = True
            opt_t = optim.LBFGS([self.temperature_triage], lr=0.01, max_iter=max_iter)
            ce_t = nn.CrossEntropyLoss()

            def eval_t():
                opt_t.zero_grad()
                scaled = raw_triage[valid_t] / torch.clamp(self.temperature_triage, min=0.01)
                loss = ce_t(scaled, val_triage_targets[valid_t])
                loss.backward()
                return loss

            opt_t.step(eval_t)
            with torch.no_grad():
                self.temperature_triage.clamp_(min=0.05, max=10.0)

        # Calibrate Head 2 (Subtypes) on samples with valid subtype target
        valid_s = val_subtype_targets >= 0
        if valid_s.any():
            self.temperature_subtype.requires_grad = True
            opt_s = optim.LBFGS([self.temperature_subtype], lr=0.01, max_iter=max_iter)
            ce_s = nn.CrossEntropyLoss()

            def eval_s():
                opt_s.zero_grad()
                scaled = raw_subtype[valid_s] / torch.clamp(self.temperature_subtype, min=0.01)
                loss = ce_s(scaled, val_subtype_targets[valid_s])
                loss.backward()
                return loss

            opt_s.step(eval_s)
            with torch.no_grad():
                self.temperature_subtype.clamp_(min=0.05, max=10.0)

        return float(self.temperature_triage.item()), float(self.temperature_subtype.item())


class MultiHeadSkinClassifier(nn.Module):
    """End-to-end Multi-Head Classifier with shared vision backbone and dual heads."""

    def __init__(
        self,
        backbone_name: str = "vit_small_patch14_dinov2",
        pretrained: bool = True,
        num_subtypes: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        # Create feature extractor backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # removes linear head, returns pooled embeddings
        )

        in_features = self.backbone.num_features

        # Multi-head probe
        self.probe = MultiHeadProbe(
            in_features=in_features,
            num_subtypes=num_subtypes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract normalized backbone features and pass through multi-head probe."""
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        # Normalize features like DINOv2 / CLIP
        features = features / features.norm(dim=-1, keepdim=True)
        return self.probe(features)

    def forward_calibrated(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through temperature-calibrated probe heads."""
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features = features / features.norm(dim=-1, keepdim=True)
        return self.probe.forward_calibrated(features)

    def predict_proba(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute softmax probabilities for triage and subtype heads."""
        cal_triage, cal_subtype = self.forward_calibrated(x)
        return F.softmax(cal_triage, dim=-1), F.softmax(cal_subtype, dim=-1)

    def predict_clinical_consensus(
        self,
        x: torch.Tensor,
        malignancy_tau: float = 0.20,
    ) -> Dict[str, Union[float, bool, str, Dict[str, float]]]:
        """Compute joint clinical consensus between Head 1 and Head 2."""
        triage_probs, subtype_probs = self.predict_proba(x)

        p_malignant_triage = float(triage_probs[0, 1])

        # Head 2 subtyping probabilities
        sub_probs = {cls_name: float(subtype_probs[0, i]) for i, cls_name in enumerate(CLASS_ORDER)}

        # Head 2 derived malignancy probability (sum of bcc + scc + mel)
        p_malignant_subtype = sub_probs["bcc"] + sub_probs["scc"] + sub_probs["mel"]

        # Consensus score (conservative maximum for patient safety)
        consensus_malignant_score = max(p_malignant_triage, p_malignant_subtype)

        # Agreement flag
        both_referral = (p_malignant_triage >= malignancy_tau) and (p_malignant_subtype >= malignancy_tau)
        both_safe = (p_malignant_triage < malignancy_tau) and (p_malignant_subtype < malignancy_tau)
        heads_agree = both_referral or both_safe

        return {
            "p_malignant_triage": round(p_malignant_triage, 4),
            "p_malignant_subtype": round(p_malignant_subtype, 4),
            "consensus_malignant_score": round(consensus_malignant_score, 4),
            "heads_agree": heads_agree,
            "referral_triggered": consensus_malignant_score >= malignancy_tau,
            "subtype_probabilities": {k: round(v, 4) for k, v in sub_probs.items()},
        }
