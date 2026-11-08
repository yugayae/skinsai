"""Multimodal Skin Lesion Classifier combining Computer Vision with Clinical Metadata.

Architecture (Late Fusion):
1. Vision Stream:
   Meta DINOv2 Small backbone extracts 384-dimensional latent embedding z_vis.
2. Clinical Tabular Stream:
   Clinical metadata (age, sex, anatomical location, diameter >= 7mm) is encoded
   into a 12-dimensional vector and processed via a Tabular MLP into z_clin (64-dim).
3. Late Fusion & Trunk:
   z_fused = [z_vis, z_clin] in R^448.
4. Dual Decoupled Heads:
   - Head 1: Binary Oncology Triage (Malignant vs Benign).
   - Head 2: Differential 6-Class Subtyping (ACK, BCC, MEL, NEV, SCC, SEK).
"""

from __future__ import annotations

from typing import Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TabularClinicalEncoder(nn.Module):
    """Encodes 12-dimensional clinical questionnaire features into a dense representation."""

    def __init__(self, in_features: int = 12, out_features: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_features),
            nn.GELU(),
            nn.BatchNorm1d(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode tabular features into latent embedding.

        Args:
            x: Tabular feature tensor of shape (batch_size, clin_dim).

        Returns:
            Encoded embedding of shape (batch_size, out_features).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class MultimodalLateFusionProbe(nn.Module):
    """Multimodal probe operating on pre-extracted vision embeddings and clinical tabular vectors."""

    def __init__(
        self,
        vis_dim: int = 384,
        clin_dim: int = 12,
        clin_embed_dim: int = 64,
        hidden_dim: int = 256,
        num_subtypes: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vis_dim = vis_dim
        self.clin_dim = clin_dim
        self.clin_embed_dim = clin_embed_dim

        # Tabular clinical encoder
        self.clin_encoder = TabularClinicalEncoder(
            in_features=clin_dim,
            out_features=clin_embed_dim,
            dropout=dropout,
        )

        # Fused projection trunk
        fused_dim = vis_dim + clin_embed_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim),
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

    def forward(self, vis_feat: torch.Tensor, clin_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (triage_logits, subtype_logits)."""
        z_clin = self.clin_encoder(clin_feat)
        z_fused = torch.cat([vis_feat, z_clin], dim=-1)
        h = self.trunk(z_fused)
        triage_logits = self.triage_head(h)
        subtype_logits = self.subtype_head(h)
        return triage_logits, subtype_logits

    def forward_calibrated(self, vis_feat: torch.Tensor, clin_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass scaled by learned temperature parameters."""
        t_logits, s_logits = self.forward(vis_feat, clin_feat)
        t_triage = torch.clamp(self.temperature_triage, min=0.01)
        t_subtype = torch.clamp(self.temperature_subtype, min=0.01)
        return t_logits / t_triage, s_logits / t_subtype

    def predict_proba(self, vis_feat: torch.Tensor, clin_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute calibrated softmax probabilities for triage and subtype."""
        cal_triage, cal_subtype = self.forward_calibrated(vis_feat, clin_feat)
        return F.softmax(cal_triage, dim=-1), F.softmax(cal_subtype, dim=-1)

    def calibrate_temperatures(
        self,
        val_vis: torch.Tensor,
        val_clin: torch.Tensor,
        val_triage_targets: torch.Tensor,
        val_subtype_targets: torch.Tensor,
        max_iter: int = 50,
    ) -> Tuple[float, float]:
        """Fit temperature parameters on validation features via NLL loss minimization."""
        with torch.no_grad():
            raw_triage, raw_subtype = self.forward(val_vis, val_clin)

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


class MultimodalSkinClassifier(nn.Module):
    """End-to-end Multimodal Classifier with visual backbone, tabular encoder, and dual heads."""

    def __init__(
        self,
        backbone_name: str = "vit_small_patch14_dinov2",
        pretrained: bool = True,
        clin_dim: int = 12,
        clin_embed_dim: int = 64,
        hidden_dim: int = 256,
        num_subtypes: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        vis_dim = self.backbone.num_features

        self.probe = MultimodalLateFusionProbe(
            vis_dim=vis_dim,
            clin_dim=clin_dim,
            clin_embed_dim=clin_embed_dim,
            hidden_dim=hidden_dim,
            num_subtypes=num_subtypes,
            dropout=dropout,
        )

    def forward(self, img: torch.Tensor, clin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract backbone features and run late fusion probe."""
        feat = self.backbone(img)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return self.probe(feat, clin)

    def predict_proba(self, img: torch.Tensor, clin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute calibrated probabilities from raw image and clinical features."""
        feat = self.backbone(img)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return self.probe.predict_proba(feat, clin)
