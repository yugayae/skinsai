"""Unit tests for DatasetItem and DatasetManifest."""

from pathlib import Path

from ml.datasets.manifest import DatasetItem, DatasetManifest


def test_dataset_item_roundtrip():
    item = DatasetItem(
        image_id="ISIC_0000001",
        profile="smartphone_clinical",
        class_id="bcc",
        raw_diagnosis="Basal cell carcinoma",
        license="CC-BY",
        license_scope="commercial_ok",
        patient_id="PAT_001",
        lesion_id="LES_001",
        source_url="https://example.com/img1.jpg",
        width=500,
        height=400,
        split="train",
        metadata_extra={"age": 55, "site": "face"},
    )

    row = item.to_row()
    assert row["image_id"] == "ISIC_0000001"
    assert row["patient_id"] == "PAT_001"
    assert row["split"] == "train"

    restored = DatasetItem.from_row(row)
    assert restored.image_id == item.image_id
    assert restored.patient_id == item.patient_id
    assert restored.class_id == item.class_id
    assert restored.width == item.width
    assert restored.metadata_extra == {"age": 55, "site": "face"}


def test_manifest_csv_json_roundtrip(tmp_path: Path):
    items = [
        DatasetItem(
            image_id=f"ISIC_{i:07d}",
            profile="smartphone_clinical",
            class_id="bcc" if i % 2 == 0 else "mel",
            raw_diagnosis="Basal cell carcinoma" if i % 2 == 0 else "Melanoma",
            license="CC-BY",
            license_scope="commercial_ok",
            patient_id=f"PAT_{i // 2}",
            lesion_id=f"LES_{i}",
            split="train" if i < 3 else "val",
        )
        for i in range(5)
    ]

    manifest = DatasetManifest(
        profile="smartphone_clinical",
        items=items,
    )

    assert len(manifest) == 5

    # CSV roundtrip
    csv_path = tmp_path / "manifest.csv"
    manifest.save_csv(csv_path)
    assert csv_path.exists()

    loaded_csv = DatasetManifest.from_csv(csv_path, profile="smartphone_clinical")
    assert len(loaded_csv) == 5
    assert loaded_csv.items[0].image_id == "ISIC_0000000"

    # JSON roundtrip
    json_path = tmp_path / "manifest.json"
    manifest.save_json(json_path)
    assert json_path.exists()

    loaded_json = DatasetManifest.from_json(json_path)
    assert len(loaded_json) == 5
    assert loaded_json.compute_manifest_hash() == manifest.compute_manifest_hash()


def test_manifest_summary_and_filter():
    items = [
        DatasetItem(
            image_id="IMG_1",
            profile="dermoscopy",
            class_id="mel",
            raw_diagnosis="Melanoma",
            license="CC-0",
            license_scope="commercial_ok",
            patient_id="P1",
            split="train",
        ),
        DatasetItem(
            image_id="IMG_2",
            profile="dermoscopy",
            class_id="mel",
            raw_diagnosis="Melanoma",
            license="CC-0",
            license_scope="commercial_ok",
            patient_id="P2",
            split="val",
        ),
        DatasetItem(
            image_id="IMG_3",
            profile="dermoscopy",
            class_id="nv",
            raw_diagnosis="Nevus",
            license="CC-BY",
            license_scope="commercial_ok",
            patient_id="P3",
            split="test",
        ),
    ]

    manifest = DatasetManifest(profile="dermoscopy", items=items)
    summary = manifest.get_summary()

    assert summary["total_images"] == 3
    assert summary["unique_patients"] == 3
    assert summary["splits"]["train"] == 1
    assert summary["splits"]["val"] == 1
    assert summary["splits"]["test"] == 1
    assert summary["classes"]["mel"] == 2
    assert summary["classes"]["nv"] == 1

    train_sub = manifest.filter_split("train")
    assert len(train_sub) == 1
    assert train_sub.items[0].image_id == "IMG_1"
