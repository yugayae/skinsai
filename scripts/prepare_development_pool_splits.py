#!/usr/bin/env python3
"""Prepare unified Development Pool (PAD-UFES-20 + ISIC Close-up) Train/Validation Splits.

Strictly enforces zero patient-level and lesion-level data leakage between Train (80%)
and Validation (20%) using Stratified Group allocation by patient_id.
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


def prepare_development_pool():
    print("=" * 80)
    print(" SkinAI Unified Development Pool (Train / Validation) Split Pipeline")
    print(" Standard: CLAIM 2024 / STARD-AI / IEC 62304 — Zero Patient Data Leakage")
    print(" Split Ratio: 80% Train / 20% Validation (Grouped by patient_id)")
    print("=" * 80)

    manifest_csv = PROJECT_ROOT / "data" / "metadata" / "isic_clinical_manifest.csv"
    if not manifest_csv.exists():
        print(f"[-] Error: {manifest_csv} not found.")
        sys.exit(1)

    df = pd.read_csv(manifest_csv, keep_default_na=False)
    print(f"[*] Loaded metadata records: {len(df)}")

    pad_raw_dir = PROJECT_ROOT / "data" / "raw" / "pad_ufes_20"
    isic_raw_dir = PROJECT_ROOT / "data" / "raw" / "isic_clinical"

    items = []
    missing_files = 0

    for _, row in df.iterrows():
        img_id = str(row["image_id"]).strip()
        is_pad = bool(row["is_existing_pad_ufes"])
        c_id = str(row["class_id"]).strip()
        raw_diag = str(row["raw_diagnosis"]).strip()

        # Locate physical file
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

        patient_id = str(row.get("patient_id", "")).strip()
        if not patient_id:
            patient_id = f"anon_patient_{img_id}"

        lesion_id = str(row.get("lesion_id", "")).strip()
        if not lesion_id:
            lesion_id = f"anon_lesion_{img_id}"

        # Target targets
        # Triage: 1=Malignant, 0=Benign, -1=Masked (ACK)
        if c_id in MALIGNANT_CLASSES:
            triage_target = 1
        elif c_id in BENIGN_CLASSES or c_id == "unmapped":
            triage_target = 0
        else:
            triage_target = -1

        subtype_target = c_id if c_id in TARGET_CLASSES else "unmapped"

        item = {
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
        }
        items.append(item)

    print(f"[+] Physical images confirmed: {len(items)} (missing: {missing_files})")

    # 1. Group items strictly by patient_id
    patient_groups: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        patient_groups[it["patient_id"]].append(it)

    print(f"[*] Total unique patient clusters: {len(patient_groups)}")

    # 2. Determine dominant class for stratified allocation
    def get_patient_dominant_class(p_items: List[dict]) -> str:
        classes = [x["class_id"] for x in p_items]
        # Prioritize rare critical classes if present
        for priority_cls in ["mel", "scc", "bcc", "ack", "sek", "nev", "unmapped"]:
            if priority_cls in classes:
                return priority_cls
        return Counter(classes).most_common(1)[0][0]

    rng = random.Random(42)
    class_to_patients: Dict[str, List[str]] = defaultdict(list)
    p_ids = list(patient_groups.keys())
    rng.shuffle(p_ids)

    for p_id in p_ids:
        dom_cls = get_patient_dominant_class(patient_groups[p_id])
        class_to_patients[dom_cls].append(p_id)

    train_patients: Set[str] = set()
    val_patients: Set[str] = set()

    # 3. Stratified allocation: 80% Train, 20% Val per dominant class
    for cls_name, cls_p_ids in sorted(class_to_patients.items()):
        rng.shuffle(cls_p_ids)
        n = len(cls_p_ids)
        n_train = max(1, int(round(n * 0.80))) if n >= 2 else 1
        n_val = n - n_train
        if n >= 2 and n_val == 0:
            n_train = n - 1
            n_val = 1

        t_pids = cls_p_ids[:n_train]
        v_pids = cls_p_ids[n_train:]

        train_patients.update(t_pids)
        val_patients.update(v_pids)

    # 4. Strict Zero-Leakage Verification
    overlap_p = train_patients & val_patients
    if overlap_p:
        raise ValueError(f"CRITICAL ERROR: Patient leakage detected! Overlap: {len(overlap_p)}")

    train_lesions: Set[str] = set()
    val_lesions: Set[str] = set()

    for it in items:
        p_id = it["patient_id"]
        l_id = it["lesion_id"]
        if p_id in train_patients:
            it["split"] = "train"
            train_lesions.add(l_id)
        elif p_id in val_patients:
            it["split"] = "val"
            val_lesions.add(l_id)
        else:
            raise ValueError(f"Patient {p_id} unassigned!")

    overlap_l = train_lesions & val_lesions
    if overlap_l:
        raise ValueError(f"CRITICAL ERROR: Lesion leakage detected! Overlap: {len(overlap_l)}")

    print("[+] Zero patient and lesion leakage strictly verified (0 overlap).")

    # 5. Output Directory & Files
    out_dir = PROJECT_ROOT / "data" / "splits" / "development_pool"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(items)
    out_df.to_csv(out_dir / "manifest.csv", index=False)

    train_df = out_df[out_df["split"] == "train"].copy()
    val_df = out_df[out_df["split"] == "val"].copy()

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)

    manifest_meta = {
        "version": "1.0.0",
        "description": "SkinAI Development Pool (PAD-UFES-20 + ISIC Close-up Permissive)",
        "split_ratio": {"train": 0.80, "val": 0.20},
        "seed": 42,
        "total_images": len(items),
        "total_patients": len(patient_groups),
        "train_images": len(train_df),
        "val_images": len(val_df),
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_meta, f, indent=2, ensure_ascii=False)

    # 6. Summary Reporting (Aviation / Medical Standards)
    print("\n" + "=" * 80)
    print(" DEVELOPMENT POOL SPLIT SUMMARY (CLAIM 2024 / STARD-AI Rigor)")
    print("=" * 80)
    print(f" Total Confirmed Images   : {len(items)}")
    print(f" Total Unique Patients    : {len(patient_groups)}")
    print(f" Train Split              : {len(train_df)} images ({len(train_patients)} patients, {len(train_df)/len(items)*100:.1f}%)")
    print(f" Validation Split         : {len(val_df)} images ({len(val_patients)} patients, {len(val_df)/len(items)*100:.1f}%)")

    print("\n[+] Breakdown by Diagnostic Class across Splits:")
    classes = ["mel", "bcc", "scc", "ack", "sek", "nev", "unmapped"]
    header = f"{'Class':<12} | {'Train Images':<14} | {'Val Images':<14} | {'Total Images':<14} | {'Train %':<10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    t_counts = train_df["class_id"].value_counts()
    v_counts = val_df["class_id"].value_counts()

    for c in classes:
        t_c = t_counts.get(c, 0)
        v_c = v_counts.get(c, 0)
        tot_c = t_c + v_c
        pct = (t_c / tot_c * 100) if tot_c > 0 else 0
        print(f"{c:<12} | {t_c:<14} | {v_c:<14} | {tot_c:<14} | {pct:>6.1f}%")
    print("-" * len(header))

    print("\n[+] Oncology Triage Breakdown (Head 1 Targets):")
    t_triage = train_df["triage_target"].value_counts()
    v_triage = val_df["triage_target"].value_counts()
    print(f" Malignant (1)   : Train={t_triage.get(1, 0)}, Val={v_triage.get(1, 0)} (Total={t_triage.get(1, 0) + v_triage.get(1, 0)})")
    print(f" Benign (0)      : Train={t_triage.get(0, 0)}, Val={v_triage.get(0, 0)} (Total={t_triage.get(0, 0) + v_triage.get(0, 0)})")
    print(f" Masked ACK (-1) : Train={t_triage.get(-1, 0)}, Val={v_triage.get(-1, 0)} (Total={t_triage.get(-1, 0) + v_triage.get(-1, 0)})")
    print("=" * 80)
    print(f"[+] Splits successfully generated in: {out_dir}")


if __name__ == "__main__":
    prepare_development_pool()
