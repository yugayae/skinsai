"""Unit tests for models, losses, and transforms."""

from pathlib import Path
import torch
from PIL import Image

from ml.models.classifier import SkinClassifier
from ml.models.dataset import SkinLesionDataset, get_canonical_transforms
from ml.models.losses import ClassWeightedFocalLoss, compute_class_weights


def test_compute_class_weights():
    counts = {"bcc": 800, "ack": 700, "mel": 50}
    order = ["bcc", "ack", "mel"]
    weights = compute_class_weights(counts, order, power=0.5)

    assert weights.shape == (3,)
    # Mel should have much higher weight than bcc
    assert weights[2] > weights[0]
    # Mean of weights should be close to 1.0
    assert abs(weights.mean().item() - 1.0) < 1e-5


def test_focal_loss_forward():
    criterion = ClassWeightedFocalLoss(gamma=2.0)
    logits = torch.randn(4, 6, requires_grad=True)
    targets = torch.tensor([0, 2, 4, 1], dtype=torch.long)

    loss = criterion(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0

    # Backward pass check
    loss.backward()
    assert logits.grad is not None


def test_skin_classifier_forward():
    # Use uninitialized backbone for fast unit test
    model = SkinClassifier(
        backbone_name="efficientnet_b0",
        num_classes=6,
        pretrained=False,
    )
    model.eval()

    dummy_x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_x)
        probs = model.predict_proba(dummy_x)

    assert logits.shape == (2, 6)
    assert probs.shape == (2, 6)
    # Check probabilities sum to 1.0
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(2), atol=1e-5)


def test_dataset_item_loading(tmp_path: Path):
    # Create dummy image
    img_path = tmp_path / "TEST_001.jpg"
    img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    img.save(img_path)

    items = [{"image_id": "TEST_001", "class_id": "bcc", "image_path": str(img_path)}]
    class_to_idx = {"bcc": 0, "mel": 1}

    ds = SkinLesionDataset(
        manifest_items=items,
        class_to_idx=class_to_idx,
        transform=get_canonical_transforms(is_train=False, input_size=(224, 224)),
    )

    assert len(ds) == 1
    tensor, target, img_id = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert target == 0
    assert img_id == "TEST_001"
