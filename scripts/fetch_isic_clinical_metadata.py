#!/usr/bin/env python3
"""Fetch, parse, and harmonize metadata for all permissive ISIC clinical close-up images."""

from collections import Counter
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
headers = {"User-Agent": "SkinAI/1.0"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def map_diagnosis(diag_str: str) -> str | None:
    """Map ISIC diagnosis text to SkinAI 6-class taxonomy."""
    if not diag_str or diag_str.lower() in ("none", "null", "unknown", "nan"):
        return None
    d = diag_str.lower().strip()

    # Melanoma
    if "melanoma" in d:
        return "mel"

    # Basal cell carcinoma
    if "basal cell" in d or "bcc" in d:
        return "bcc"

    # Squamous cell carcinoma
    if "squamous cell" in d or "scc" in d:
        return "scc"

    # Actinic keratosis
    if "actinic" in d or "solar keratosis" in d or "akiec" in d:
        return "ack"

    # Seborrheic keratosis / benign keratosis
    if "seborrheic" in d or "solar lentigo" in d or "bkl" in d:
        return "sek"

    # Nevus
    if "nevus" in d or "nevi" in d or "naevus" in d or "mole" in d:
        return "nev"

    return None


def fetch_all_metadata():
    print("=" * 70)
    print(" Fetching Permissive ISIC Archive 'clinical: close-up' Metadata")
    print(" Target: copyright_license in ('CC-0', 'CC-BY')")
    print("=" * 70)

    # Load local existing manifest to detect overlap
    local_manifest_path = PROJECT_ROOT / "data" / "splits" / "smartphone_clinical" / "manifest.csv"
    existing_ids = set()
    if local_manifest_path.exists():
        df_local = pd.read_csv(local_manifest_path, keep_default_na=False)
        existing_ids = set(df_local["image_id"].astype(str))
    print(f"[*] Local PAD-UFES-20 image IDs in workspace: {len(existing_ids)}")

    base_query = 'image_type:"clinical: close-up" AND (copyright_license:"CC-0" OR copyright_license:"CC-BY")'
    url = f"https://api.isic-archive.com/api/v2/images/search/?query={urllib.parse.quote(base_query)}&limit=100"

    all_records = []
    page = 1
    t0 = time.time()

    while url:
        print(f"    Fetching page {page:02d}... (Accumulated: {len(all_records)} images)", end="\r")
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"\n[-] Error on page {page}: {e}. Retrying once after 3s...")
            time.sleep(3)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

        results = data.get("results", [])
        if not results:
            break

        for item in results:
            isic_id = item.get("isic_id")
            meta_clin = item.get("metadata", {}).get("clinical", {})
            meta_acq = item.get("metadata", {}).get("acquisition", {})

            # Extract diagnoses hierarchy
            d1 = meta_clin.get("diagnosis_1", "")
            d2 = meta_clin.get("diagnosis_2", "")
            d3 = meta_clin.get("diagnosis_3", "")
            diag_legacy = meta_clin.get("diagnosis", "")

            # Prioritize d3, then d2, then legacy
            raw_diag = d3 or d2 or diag_legacy or d1
            mapped_class = map_diagnosis(d3) or map_diagnosis(d2) or map_diagnosis(diag_legacy)

            confirm_type = meta_clin.get("diagnosis_confirm_type", "unknown")
            patient_id = meta_clin.get("patient_id") or f"ISIC_anon_{isic_id}"
            lesion_id = meta_clin.get("lesion_id") or f"lesion_{isic_id}"
            license_val = meta_acq.get("copyright_license", item.get("copyright_license", "CC-BY"))

            all_records.append({
                "image_id": isic_id,
                "profile": "smartphone_clinical",
                "class_id": mapped_class or "unmapped",
                "raw_diagnosis": raw_diag,
                "diagnosis_1": d1,
                "diagnosis_2": d2,
                "diagnosis_3": d3,
                "diagnosis_confirm_type": confirm_type,
                "license": license_val,
                "license_scope": "commercial_ok",
                "patient_id": patient_id,
                "lesion_id": lesion_id,
                "width": meta_acq.get("pixels_x"),
                "height": meta_acq.get("pixels_y"),
                "fitzpatrick_skin_type": meta_clin.get("fitzpatrick_skin_type", ""),
                "anatom_site_1": meta_clin.get("anatom_site_1", ""),
                "anatom_site_2": meta_clin.get("anatom_site_2", ""),
                "age_approx": meta_clin.get("age_approx"),
                "sex": meta_clin.get("sex", ""),
                "source_url": f"https://isic-archive.s3.amazonaws.com/images/{isic_id}.jpg",
                "is_existing_pad_ufes": isic_id in existing_ids,
            })

        url = data.get("next")
        page += 1
        time.sleep(0.1)

    print(f"\n[+] Total permissive close-up records fetched: {len(all_records)} (Time: {time.time()-t0:.1f}s)")

    df = pd.DataFrame(all_records)

    # Breakdown analysis
    print("\n" + "=" * 70)
    print(" Audit Breakdown of Permissive ISIC Close-up Collection")
    print("=" * 70)

    overlap_count = int(df["is_existing_pad_ufes"].sum())
    new_count = len(df) - overlap_count
    print(f"[*] Already in PAD-UFES-20 : {overlap_count} images")
    print(f"[*] Brand New ISIC Images  : {new_count} images")

    print("\n[*] Distribution across Target 6 Classes (All 3,597 records):")
    class_counts = Counter(df["class_id"])
    for c in ["bcc", "scc", "ack", "sek", "mel", "nev", "unmapped"]:
        print(f"    - {c:<10}: {class_counts[c]:>5} ({class_counts[c]/len(df)*100:.1f}%)")

    new_df = df[~df["is_existing_pad_ufes"]]
    print(f"\n[*] NEW Images ONLY ({len(new_df)} total):")
    new_class_counts = Counter(new_df["class_id"])
    for c in ["bcc", "scc", "ack", "sek", "mel", "nev", "unmapped"]:
        print(f"    - {c:<10}: {new_class_counts[c]:>5} ({new_class_counts[c]/max(len(new_df),1)*100:.1f}%)")

    # Confirmation types
    print("\n[*] Diagnostic Confirmation Types:")
    for k, v in Counter(df["diagnosis_confirm_type"]).most_common(5):
        print(f"    - {k:<30}: {v:>5}")

    # Save metadata
    out_dir = PROJECT_ROOT / "data" / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "isic_clinical_manifest.csv"
    json_path = out_dir / "isic_clinical_manifest.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    print(f"\n[+] Full manifest saved to: {csv_path}")
    return df


if __name__ == "__main__":
    fetch_all_metadata()
