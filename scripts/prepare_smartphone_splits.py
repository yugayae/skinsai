#!/usr/bin/env python3
"""Parse PAD-UFES-20 raw metadata and produce stratified, zero-leakage patient-level splits."""

import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.manifest import DatasetItem, DatasetManifest  # noqa: E402
from ml.datasets.splitter import DatasetSplitter  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

DIAG_TO_CLASS = {
    "Basal cell carcinoma": "bcc",
    "Squamous cell carcinoma, NOS": "scc",
    "Solar or actinic keratosis": "ack",
    "Seborrheic keratosis": "sek",
    "Melanoma, NOS": "mel",
    "Nevus": "nev",
}


def build_smartphone_clinical_splits():
    metadata_path = Path("data/metadata/pad_ufes_20.json")
    if not metadata_path.exists():
        print(f"[-] Error: {metadata_path} not found.")
        sys.exit(1)

    print(f"[*] Loading PAD-UFES-20 metadata from {metadata_path}...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"[*] Total raw records: {len(records)}")

    items = []
    for r in records:
        img_id = r["isic_id"]
        clin = r.get("metadata", {}).get("clinical", {})
        acq = r.get("metadata", {}).get("acquisition", {})
        files = r.get("files", {})

        raw_diag = clin.get("diagnosis_3") or clin.get("diagnosis_2") or "Unknown"
        class_id = DIAG_TO_CLASS.get(raw_diag)
        if not class_id:
            print(f"[!] Warning: unknown diagnosis '{raw_diag}' for {img_id}, skipping.")
            continue

        patient_id = clin.get("patient_id")
        lesion_id = clin.get("lesion_id")
        source_url = files.get("full", {}).get("url")
        w = acq.get("pixels_x")
        h = acq.get("pixels_y")

        extra = {
            "anatom_site_1": clin.get("anatom_site_1"),
            "anatom_site_2": clin.get("anatom_site_2"),
            "diagnosis_confirm_type": clin.get("diagnosis_confirm_type"),
            "age_approx": clin.get("age_approx"),
            "file_size_bytes": files.get("full", {}).get("size"),
        }

        item = DatasetItem(
            image_id=img_id,
            profile="smartphone_clinical",
            class_id=class_id,
            raw_diagnosis=raw_diag,
            license=r.get("copyright_license", "CC-BY"),
            license_scope="commercial_ok",
            patient_id=patient_id,
            lesion_id=lesion_id,
            source_url=source_url,
            width=w,
            height=h,
            metadata_extra=extra,
        )
        items.append(item)

    print(f"[*] Processed {len(items)} items mapped to 6 target classes.")

    raw_manifest = DatasetManifest(profile="smartphone_clinical", items=items)

    # Run Group-Level Splitter (70% train, 15% val, 15% test)
    splitter = DatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    split_manifest = splitter.split(raw_manifest)

    # Output directory
    out_dir = Path("data/splits/smartphone_clinical")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full manifest
    split_manifest.save_csv(out_dir / "manifest.csv")
    split_manifest.save_json(out_dir / "manifest.json")

    # Save individual split CSVs
    train_manifest = split_manifest.filter_split("train")
    val_manifest = split_manifest.filter_split("val")
    test_manifest = split_manifest.filter_split("test")

    train_manifest.save_csv(out_dir / "train.csv")
    val_manifest.save_csv(out_dir / "val.csv")
    test_manifest.save_csv(out_dir / "test.csv")

    summary = split_manifest.get_summary()
    print("\n" + "=" * 60)
    print(" SMARTPHONE CLINICAL DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total Images    : {summary['total_images']}")
    print(f"Unique Patients : {summary['unique_patients']}")
    print(f"Unique Lesions  : {summary['unique_lesions']}")
    print(f"Manifest Hash   : {summary['manifest_hash'][:16]}...")
    print("\nSplits breakdown (images):")
    for s_name, s_count in summary["splits"].items():
        pct = (s_count / summary["total_images"]) * 100
        print(f"  - {s_name:<6}: {s_count:>5} images ({pct:>5.1f}%)")

    print("\nClass breakdown across whole dataset:")
    for c_id, c_count in summary["classes"].items():
        pct = (c_count / summary["total_images"]) * 100
        print(f"  - {c_id:<6}: {c_count:>5} images ({pct:>5.1f}%)")

    # Per-split class breakdown
    for split_name, sub_m in [("Train", train_manifest), ("Val", val_manifest), ("Test", test_manifest)]:
        sub_sum = sub_m.get_summary()
        print(f"\n{split_name} class breakdown ({sub_sum['total_images']} images, {sub_sum['unique_patients']} patients):")
        for c_id, c_count in sub_sum["classes"].items():
            pct = (c_count / sub_sum["total_images"]) * 100
            print(f"  - {c_id:<6}: {c_count:>4} ({pct:>5.1f}%)")

    print(f"\n[+] Saved split manifests to: {out_dir}")


if __name__ == "__main__":
    build_smartphone_clinical_splits()
