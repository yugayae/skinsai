"""Unit tests for DatasetDownloader with mocked responses and disk quota safety."""

from pathlib import Path

from ml.datasets.downloader import DatasetDownloader
from ml.datasets.manifest import DatasetItem, DatasetManifest


def test_downloader_disk_quota_rejection(tmp_path: Path):
    # Set high min_free_space_bytes so check fails
    downloader = DatasetDownloader(output_dir=tmp_path, min_free_space_bytes=10**15)
    item = DatasetItem(
        image_id="TEST_001",
        profile="smartphone_clinical",
        class_id="bcc",
        raw_diagnosis="BCC",
        license="CC-BY",
        license_scope="commercial_ok",
        source_url="https://example.com/test.jpg",
    )
    manifest = DatasetManifest(profile="smartphone_clinical", items=[item])

    _, stats = downloader.download_manifest(manifest)
    assert stats.downloaded == 0
    assert len(stats.errors) > 0
    assert "disk space below threshold" in stats.errors[0]


def test_downloader_skips_existing_files(tmp_path: Path):
    # Pre-create the file
    existing_file = tmp_path / "TEST_EXISTS.jpg"
    existing_file.write_bytes(b"EXISTING_IMAGE_DATA_12345")

    downloader = DatasetDownloader(output_dir=tmp_path, min_free_space_bytes=1024)
    item = DatasetItem(
        image_id="TEST_EXISTS",
        profile="smartphone_clinical",
        class_id="bcc",
        raw_diagnosis="BCC",
        license="CC-BY",
        license_scope="commercial_ok",
        source_url="https://example.com/test_exists.jpg",
    )
    manifest = DatasetManifest(profile="smartphone_clinical", items=[item])

    updated_manifest, stats = downloader.download_manifest(manifest)
    assert stats.skipped_existing == 1
    assert stats.downloaded == 0
    assert stats.failed == 0
    assert updated_manifest.items[0].image_path == str(existing_file)
    assert updated_manifest.items[0].sha256 is not None
