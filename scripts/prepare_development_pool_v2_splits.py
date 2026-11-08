#!/usr/bin/env python3
"""Prepare Unified Development Pool v2 (PAD-UFES-20 + ISIC Close-up + MED-NODE).

Standard: CLAIM 2024 / STARD-AI / IEC 62304
- Group split strictly by patient_id
- 0% patient and lesion data leakage
- Train (80%) / Validation (20%)
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Set

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_CLASSES = ["bcc", "scc", "ack", "sek", "mel", "nev"]
MALIGNANT_CLASSES = {"bcc", "scc", "mel"}
BENIGN_CLASSES = {"nev", "sek"}


def prepare_development_pool_v2():
    print("=" * 80)
    print(" SkinAI Unified Development Pool v2 Split Pipeline (with MED-NODE)")
    print(" Standard: CLAIM 2024 / STARD-AI / IEC 62304 — Zero Patient Data Leakage")
    print(" Split Ratio: 80% Train / 20% Validation (Grouped by patient_id)")
    print("=" * 80)

    # 1. Base manifest (PAD-UFES-20 + ISIC Close-up)
    manifest_base = PROJECT_ROOT / "data" / "metadata" / "isic_clinical_manifest.csv"
    if not manifest_base.exists():
        print(f"[-] Error: {manifest_base} not found.")
        sys.exit(1)

    df_base = pd.read_csv(manifest_base, keep_default_na=False)
    print(f"[*] Base metadata records: {len(df_base)}")

    pad_raw_dir = PROJECT_ROOT / "data" / "raw" / "pad_ufes_20"
    isic_raw_dir = PROJECT_ROOT / "data" / "raw" / "isic_clinical"

    items = []
    missing_files = 0

    for _, row in df_base.iterrows():
        img_id = str(row["image_id"]).strip()
        is_pad = bool(row["is_existing_pad_ufes"])
        c_id = str(row["class_id"]).strip()
        raw_diag = str(row["raw_diagnosis"]).strip()

        if is_pad:
            img_path = pad_raw_dir / f"{img_id}.png"
            if not img_path.exists():
                img_path = pad_raw_dir / f"{img_id}.jpg"
            source_ds = "pad_ufes_20"
        else:
            img_path = isic_raw_dir / f"{img_id}.jpg"
            source_ds = "isic_clinical"

        if not img_path.exists():
            missing_files += 1
            continue

        patient_id = str(row.get("patient_id", "")).strip() or f"anon_patient_{img_id}"
        lesion_id = str(row.get("lesion_id", "")).strip() or f"anon_lesion_{img_id}"

        if c_id in MALIGNANT_CLASSES:
            triage_target = 1
        elif c_id in BENIGN_CLASSES or c_id == "unmapped":
            triage_target = 0
        else:
            triage_target = -1

        subtype_target = c_id if c_id in TARGET_CLASSES else "unmapped"

        items.append({
            "image_id": img_id,
            "patient_id": patient_id,
            "lesion_id": lesion_id,
            "source_dataset": source_ds,
            "class_id": c_id,
            "subtype_target": subtype_target,
            "triage_target": triage_target,
            "raw_diagnosis": raw_diag,
            "age_approx": row.get("age_approx", ""),
            "sex": row.get("sex", ""),
            "anatom_site_1": row.get("anatom_site_1", ""),
            "file_path": str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "license": row.get("license", "CC-BY"),
            "license_scope": "commercial_ok",
        })

    # 2. Add MED-NODE manifest
    manifest_mednode = PROJECT_ROOT / "data" / "metadata" / "mednode_manifest.csv"
    if not manifest_mednode.exists():
        print(f"[-] Error: {manifest_mednode} not found.")
        sys.exit(1)

    df_mednode = pd.read_csv(manifest_mednode, keep_default_na=False)
    print(f"[*] MED-NODE metadata records: {len(df_mednode)}")

    for _, row in df_mednode.iterrows():
        img_p = PROJECT_ROOT / row["file_path"]
        if not img_p.exists():
            missing_files += 1
            continue

        items.append({
            "image_id": row["image_id"],
            "patient_id": row["patient_id"],
            "lesion_id": row["lesion_id"],
            "source_dataset": "mednode",
            "class_id": row["subtype_target"],
            "subtype_target": row["subtype_target"],
            "triage_target": int(row["triage_target"]),
            "raw_diagnosis": f"MED-NODE {row['category']} (biopsy confirmed)",
            "age_approx": "",
            "sex": "",
            "anatom_site_1": "",
            "file_path": row["file_path"],
            "license": row["license"],
            "license_scope": row["license_scope"],
        })

    print(f"[+] Total physically verified images: {len(items)} (missing files skipped: {missing_files})")

    # 3. Patient Grouping
    patient_to_images = defaultdict(list)
    for it in items:
        patient_to_images[it["patient_id"]].append(it)

    print(f"[*] Total unique patient clusters: {len(patient_to_images)}")

    # Class distribution across patients
    def get_patient_dominant_class(p_items: List[Dict]) -> str:
        for target_c in ["mel", "scc", "bcc", "ack", "sek", "nev"]:
            if any(x["subtype_target"] == target_c for x in p_items):
                return target_c
        return "unmapped"

    patients_by_class = defaultdict(list)
    for p_id, p_items in patient_to_images.items():
        dom_c = get_patient_dominant_class(p_items)
        patients_by_class[dom_c].append(p_id)

    print("\n[*] Patient clusters by dominant diagnostic class:")
    for c, p_list in sorted(patients_by_class.items()):
        total_imgs = sum(len(patient_to_images[p]) for p in p_list)
        print(f"    - {c:<10}: {len(p_list):>4} patients ({total_imgs:>4} images)")

    # 4. Stratified Allocation
    random.seed(42)
    train_patients: Set[str] = set()
    val_patients: Set[str] = set()

    for c, p_list in patients_by_class.items():
        p_shuffled = list(p_list)
        random.shuffle(p_shuffled)
        n_val = max(1, int(round(len(p_shuffled) * 0.20)))
        val_subset = set(p_shuffled[:n_val])
        train_subset = set(p_shuffled[n_val:])

        train_patients.update(train_subset)
        val_patients.update(val_subset)

    # 5. Build Splits
    train_records = []
    val_records = []

    for it in items:
        if it["patient_id"] in train_patients:
            train_records.append(it)
        elif it["patient_id"] in val_patients:
            val_records.append(it)
        else:
            raise ValueError(f"Unassigned patient: {it['patient_id']}")

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    # 6. Strict Verification of Zero Leakage
    train_pt_set = set(train_df["patient_id"])
    val_pt_set = set(val_df["patient_id"])
    overlap_pt = train_pt_set & val_pt_set

    train_ls_set = set(train_df["lesion_id"])
    val_ls_set = set(val_df["lesion_id"])
    overlap_ls = train_ls_set & val_ls_set

    print("\n" + "=" * 80)
    print(" VERIFICATION OF SPLIT INTEGRITY (CLAIM 2024 / STARD-AI Standards)")
    print("=" * 80)
    print(f" Train: {len(train_df)} images ({len(train_pt_set)} patients, {len(train_df)/len(items)*100:.1f}%)")
    print(f" Val  : {len(val_df)} images ({len(val_pt_set)} patients, {len(val_df)/len(items)*100:.1f}%)")
    print(f" Patient ID Overlap (Train vs Val): {len(overlap_pt)} (MUST BE 0)")
    print(f" Lesion ID Overlap  (Train vs Val): {len(overlap_ls)} (MUST BE 0)")

    if len(overlap_pt) != 0 or len(overlap_ls) != 0:
        print("[-] CRITICAL ERROR: Patient/Lesion leakage detected!")
        sys.exit(1)

    print("\n Diagnostic Subtype Distribution in Unified Development Pool v2:")
    print(f" {'Subtype':<10} | {'Train':<8} | {'Val':<8} | {'Total':<8} | {'Val Ratio':<8}")
    print("-" * 55)
    for c in ["mel", "bcc", "scc", "ack", "sek", "nev", "unmapped"]:
        n_tr = sum(train_df["subtype_target"] == c)
        n_va = sum(val_df["subtype_target"] == c)
        n_tot = n_tr + n_va
        v_pct = (n_va / max(n_tot, 1)) * 100
        print(f" {c:<10} | {n_tr:<8} | {n_va:<8} | {n_tot:<8} | {v_pct:5.1f}%")

    print("\n Triage Target Distribution (1=Malignant, 0=Benign, -1=Masked ACK):")
    for t, name in [(1, "Malignant"), (0, "Benign"), (-1, "Masked ACK")]:
        n_tr = sum(train_df["triage_target"] == t)
        n_va = sum(val_df["triage_target"] == t)
        print(f" {name:<12}: Train={n_tr:>4}, Val={n_va:>4}, Total={n_tr+n_va:>4}")

    # 7. Save Artifacts
    out_dir = PROJECT_ROOT / "data" / "splits" / "development_pool_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_manifest = pd.DataFrame(items)
    all_manifest["split"] = all_manifest["patient_id"].apply(lambda p: "train" if p in train_pt_set else "val")

    all_manifest.to_csv(out_dir / "manifest.csv", index=False)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)

    metadata_info = {
        "version": "2.0.0",
        "description": "Unified Development Pool v2: PAD-UFES-20 + ISIC Close-up + MED-NODE",
        "total_images": len(items),
        "total_patients": len(patient_to_images),
        "train_images": len(train_df),
        "train_patients": len(train_pt_set),
        "val_images": len(val_df),
        "val_patients": len(val_pt_set),
        "patient_overlap": len(overlap_pt),
        "lesion_overlap": len(overlap_ls),
        "melanoma_total": int(sum(all_manifest["subtype_target"] == "mel")),
        "melanoma_train": int(sum(train_df["subtype_target"] == "mel")),
        "melanoma_val": int(sum(val_df["subtype_target"] == "mel")),
        "sources": {
            "pad_ufes_20": int(sum(all_manifest["source_dataset"] == "pad_ufes_20")),
            "isic_clinical": int(sum(all_manifest["source_dataset"] == "isic_clinical")),
            "mednode": int(sum(all_manifest["source_dataset"] == "mednode")),
        },
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(metadata_info, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Manifests saved to {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    prepare_development_pool_v2()
