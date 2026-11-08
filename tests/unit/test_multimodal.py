"""Unit tests for Multimodal Late Fusion architecture and TabularClinicalEncoder."""

import torch

from ml.models.multimodal import (
    MultimodalLateFusionProbe,
    MultimodalSkinClassifier,
    TabularClinicalEncoder,
)


def test_tabular_clinical_encoder():
    encoder = TabularClinicalEncoder(in_features=12, out_features=64)
    encoder.eval()

    # Batch of 4 clinical vectors
    x = torch.randn(4, 12)
    with torch.no_grad():
        out = encoder(x)

    assert out.shape == (4, 64)


def test_multimodal_late_fusion_probe():
    probe = MultimodalLateFusionProbe(
        vis_dim=384,
        clin_dim=12,
        clin_embed_dim=64,
        hidden_dim=128,
        num_subtypes=6,
    )
    probe.eval()

    vis = torch.randn(4, 384)
    clin = torch.randn(4, 12)

    with torch.no_grad():
        t_logits, s_logits = probe(vis, clin)
        t_probs, s_probs = probe.predict_proba(vis, clin)

    assert t_logits.shape == (4, 2)
    assert s_logits.shape == (4, 6)
    assert t_probs.shape == (4, 2)
    assert s_probs.shape == (4, 6)
    assert torch.allclose(t_probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(s_probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_multimodal_skin_classifier():
    model = MultimodalSkinClassifier(
        backbone_name="efficientnet_b0",
        pretrained=False,
        clin_dim=12,
        clin_embed_dim=32,
        hidden_dim=64,
        num_subtypes=6,
    )
    model.eval()

    img = torch.randn(2, 3, 224, 224)
    clin = torch.randn(2, 12)

    with torch.no_grad():
        t_probs, s_probs = model.predict_proba(img, clin)

    assert t_probs.shape == (2, 2)
    assert s_probs.shape == (2, 6)
