"""Unit tests for Multi-Head Skin Lesion Classifier, Probe, Loss, and Consensus."""

import torch

from ml.models.multi_head import (
    MaskedClassWeightedFocalLoss,
    MultiHeadLoss,
    MultiHeadProbe,
    MultiHeadSkinClassifier,
)


def test_masked_focal_loss():
    loss_fn = MaskedClassWeightedFocalLoss(gamma=2.0, ignore_index=-1)
    logits = torch.randn(4, 6, requires_grad=True)
    # Target with -1 (ignored)
    targets = torch.tensor([0, -1, 3, 5], dtype=torch.long)

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    # Gradient for the ignored sample should be 0
    assert torch.allclose(logits.grad[1], torch.zeros(6))


def test_multi_head_probe_forward_and_proba():
    probe = MultiHeadProbe(in_features=384, num_subtypes=6, hidden_dim=128)
    probe.eval()

    x = torch.randn(4, 384)
    with torch.no_grad():
        t_logits, s_logits = probe(x)
        t_probs, s_probs = probe.predict_proba(x)

    assert t_logits.shape == (4, 2)
    assert s_logits.shape == (4, 6)
    assert t_probs.shape == (4, 2)
    assert s_probs.shape == (4, 6)

    # Check probabilities sum to 1.0 across rows
    assert torch.allclose(t_probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(s_probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_multi_head_loss_with_masking():
    loss_fn = MultiHeadLoss(triage_weight=1.0, subtype_weight=1.0)

    t_logits = torch.randn(4, 2, requires_grad=True)
    s_logits = torch.randn(4, 6, requires_grad=True)

    # Sample 0: Full labels (triage=1, subtype=1 [bcc])
    # Sample 1: Only triage label (triage=0 [benign], subtype=-1 [masked])
    # Sample 2: Only subtype label (triage=-1 [ack], subtype=0 [ack])
    # Sample 3: Full labels (triage=0, subtype=3 [nev])
    t_targets = torch.tensor([1, 0, -1, 0], dtype=torch.long)
    s_targets = torch.tensor([1, -1, 0, 3], dtype=torch.long)

    total_loss, l_t, l_s = loss_fn(t_logits, s_logits, t_targets, s_targets)

    assert total_loss.dim() == 0
    assert total_loss.item() > 0.0
    assert l_t.item() > 0.0
    assert l_s.item() > 0.0

    total_loss.backward()
    assert t_logits.grad is not None
    assert s_logits.grad is not None


def test_multi_head_classifier_consensus():
    # Test classifier consensus inference logic using MultiHeadSkinClassifier with pretrained=False
    model = MultiHeadSkinClassifier(
        backbone_name="efficientnet_b0",
        pretrained=False,
        num_subtypes=6,
        hidden_dim=64,
    )
    model.eval()

    dummy_img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        res = model.predict_clinical_consensus(dummy_img, malignancy_tau=0.20)

    assert "p_malignant_triage" in res
    assert "p_malignant_subtype" in res
    assert "consensus_malignant_score" in res
    assert "heads_agree" in res
    assert "referral_triggered" in res
    assert "subtype_probabilities" in res
    assert len(res["subtype_probabilities"]) == 6
    assert 0.0 <= res["consensus_malignant_score"] <= 1.0
