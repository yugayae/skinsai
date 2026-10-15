"""Two-level hierarchical skin lesion classifier for risk triage and subtype identification.

Level 1 (Trinary Risk Triage):
  - malignant: [bcc, scc, mel]
  - pre_malignant: [ack]
  - benign: [nev, sek]

Level 2 (Specific Subtyping):
  - If malignant -> [bcc, scc, mel]
  - If benign -> [nev, sek]
  - If pre_malignant -> [ack]
"""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.classifier import SkinClassifier

# Clinical hierarchy mapping
RISK_CATEGORIES = ["benign", "pre_malignant", "malignant"]

CLASS_TO_RISK = {
    "nev": "benign",
    "sek": "benign",
    "ack": "pre_malignant",
    "bcc": "malignant",
    "scc": "malignant",
    "mel": "malignant",
}

RISK_TO_CLASSES = {
    "benign": ["nev", "sek"],
    "pre_malignant": ["ack"],
    "malignant": ["bcc", "scc", "mel"],
}


class HierarchicalSkinClassifier(nn.Module):
    """Two-stage hierarchical classifier optimizing both clinical risk triage and subtyping."""

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        # Level 1: Risk Triage Head (3 classes: benign, pre_malignant, malignant)
        self.level1_model = SkinClassifier(
            backbone_name=backbone_name,
            num_classes=3,
            pretrained=pretrained,
            drop_rate=drop_rate,
        )

        # Level 2A: Malignant Subtype Head (3 classes: bcc, scc, mel)
        self.malignant_head = SkinClassifier(
            backbone_name=backbone_name,
            num_classes=3,
            pretrained=pretrained,
            drop_rate=drop_rate,
        )

        # Level 2B: Benign Subtype Head (2 classes: nev, sek)
        self.benign_head = SkinClassifier(
            backbone_name=backbone_name,
            num_classes=2,
            pretrained=pretrained,
            drop_rate=drop_rate,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns:

        (level1_logits, malignant_sub_logits, benign_sub_logits)
        """
        l1_logits = self.level1_model(x)
        mal_logits = self.malignant_head(x)
        ben_logits = self.benign_head(x)
        return l1_logits, mal_logits, ben_logits

    def predict_hierarchical_proba(self, x: torch.Tensor) -> Dict[str, float]:
        """Compute joint calibrated probabilities across all 6 classes via probability tree:

        P(class) = P(category) * P(class | category)
        """
        l1_logits, mal_logits, ben_logits = self.forward(x)

        l1_probs = F.softmax(l1_logits, dim=-1)[0]
        mal_sub_probs = F.softmax(mal_logits, dim=-1)[0]
        ben_sub_probs = F.softmax(ben_logits, dim=-1)[0]

        p_benign = float(l1_probs[0])
        p_premalignant = float(l1_probs[1])
        p_malignant = float(l1_probs[2])

        # Joint probabilities
        joint_probs = {
            # Benign branch
            "nev": round(p_benign * float(ben_sub_probs[0]), 4),
            "sek": round(p_benign * float(ben_sub_probs[1]), 4),
            # Pre-malignant branch
            "ack": round(p_premalignant, 4),
            # Malignant branch
            "bcc": round(p_malignant * float(mal_sub_probs[0]), 4),
            "scc": round(p_malignant * float(mal_sub_probs[1]), 4),
            "mel": round(p_malignant * float(mal_sub_probs[2]), 4),
        }

        # Normalize sum to 1.0
        total = sum(joint_probs.values())
        if total > 0:
            return {k: round(v / total, 4) for k, v in joint_probs.items()}
        return joint_probs
