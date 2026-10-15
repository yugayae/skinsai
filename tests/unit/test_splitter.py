"""Unit tests for DatasetSplitter and GroupLeakageError."""

import pytest
from ml.datasets.manifest import DatasetItem, DatasetManifest
from ml.datasets.splitter import DatasetSplitter, GroupLeakageError


def create_synthetic_manifest(num_patients: int = 50, images_per_patient: int = 3) -> DatasetManifest:
    items = []
    classes = ["bcc", "mel", "scc", "ack", "sek", "nev"]
    for p_idx in range(num_patients):
        p_id = f"PAT_{p_idx:03d}"
        assigned_cls = classes[p_idx % len(classes)]
        for img_idx in range(images_per_patient):
            img_id = f"IMG_{p_idx}_{img_idx}"
            les_id = f"LES_{p_idx}_{img_idx // 2}"
            items.append(
                DatasetItem(
                    image_id=img_id,
                    profile="smartphone_clinical",
                    class_id=assigned_cls,
                    raw_diagnosis=assigned_cls,
                    license="CC-BY",
                    license_scope="commercial_ok",
                    patient_id=p_id,
                    lesion_id=les_id,
                )
            )
    return DatasetManifest(profile="smartphone_clinical", items=items)


def test_splitter_zero_patient_leakage():
    manifest = create_synthetic_manifest(num_patients=60, images_per_patient=2)
    splitter = DatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    split_manifest = splitter.split(manifest)

    # Verify every item has a split
    for item in split_manifest.items:
        assert item.split in ("train", "val", "test")

    # Gather patient sets
    train_patients = {it.patient_id for it in split_manifest.items if it.split == "train"}
    val_patients = {it.patient_id for it in split_manifest.items if it.split == "val"}
    test_patients = {it.patient_id for it in split_manifest.items if it.split == "test"}

    # Strict assertion of zero patient leakage
    assert len(train_patients & val_patients) == 0
    assert len(train_patients & test_patients) == 0
    assert len(val_patients & test_patients) == 0

    # Gather lesion sets
    train_lesions = {it.lesion_id for it in split_manifest.items if it.split == "train"}
    val_lesions = {it.lesion_id for it in split_manifest.items if it.split == "val"}
    test_lesions = {it.lesion_id for it in split_manifest.items if it.split == "test"}

    # Strict assertion of zero lesion leakage
    assert len(train_lesions & val_lesions) == 0
    assert len(train_lesions & test_lesions) == 0
    assert len(val_lesions & test_lesions) == 0


def test_splitter_leakage_detector_raises():
    manifest = create_synthetic_manifest(num_patients=10, images_per_patient=2)
    splitter = DatasetSplitter()
    split_manifest = splitter.split(manifest)

    # Intentionally corrupt one item to introduce leakage
    corrupted_items = list(split_manifest.items)
    # Find a patient in val
    val_item = next(it for it in corrupted_items if it.split == "val")
    # Change it to train so patient is now in both train and val
    corrupted_items[0] = corrupted_items[0].model_copy(
        update={"patient_id": val_item.patient_id, "split": "train"}
    )
    bad_manifest = DatasetManifest(profile=manifest.profile, items=corrupted_items)

    with pytest.raises(GroupLeakageError, match="Patient leakage"):
        DatasetSplitter.verify_no_leakage(bad_manifest)
