#!/usr/bin/env python3
"""Compute exact class distributions for:
1. Profile B: PAD-UFES-20 (Collection 406, CC-BY)
2. Profile A: Permissive Dermoscopy (CC-0 and CC-BY sub-cohort)
"""

import json
import sys
import urllib.request
import urllib.parse
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
headers = {"User-Agent": "SkinAI/1.0"}


def audit_pad_ufes_distribution():
    print("=" * 60)
    print(" AUDITING PAD-UFES-20 (Collection 406) CLASSES")
    print("=" * 60)
    url = "https://api.isic-archive.com/api/v2/images/search/?collections=406&limit=2500"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = data.get("results", [])
    print(f"Total retrieved images: {len(results)}")

    diag_counter = Counter()
    patient_ids = set()
    lesion_ids = set()

    for item in results:
        clin = item.get("metadata", {}).get("clinical", {})
        d3 = clin.get("diagnosis_3") or clin.get("diagnosis_2") or "Unknown"
        diag_counter[d3] += 1
        if clin.get("patient_id"):
            patient_ids.add(clin.get("patient_id"))
        if clin.get("lesion_id"):
            lesion_ids.add(clin.get("lesion_id"))

    print(f"Unique Patients: {len(patient_ids)}")
    print(f"Unique Lesions : {len(lesion_ids)}")
    print("\nClass breakdown:")
    for diag, count in diag_counter.most_common():
        pct = (count / len(results)) * 100
        print(f"  - {diag:<35}: {count:>5} ({pct:>5.1f}%)")


def audit_permissive_dermoscopy_distribution():
    print("\n" + "=" * 60)
    print(" AUDITING PERMISSIVE DERMOSCOPY (CC-0 / CC-BY) COLLECTIONS")
    print("=" * 60)
    # Check top dermoscopy collections with CC-0 / CC-BY licenses
    # Query collections API
    url = "https://api.isic-archive.com/api/v2/collections/?limit=100"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        cols_data = json.loads(resp.read().decode("utf-8"))

    print("Checking collections...")
    for col in cols_data.get("results", []):
        col_id = col.get("id")
        col_name = col.get("name")
        # Sample one image to see license & modality
        sample_url = f"https://api.isic-archive.com/api/v2/images/search/?collections={col_id}&limit=1"
        try:
            r = urllib.request.Request(sample_url, headers=headers)
            with urllib.request.urlopen(r) as s_resp:
                sd = json.loads(s_resp.read().decode("utf-8"))
                cnt = sd.get("count", 0)
                if cnt > 100 and sd.get("results"):
                    first = sd["results"][0]
                    lic = first.get("copyright_license")
                    mod = first.get("metadata", {}).get("acquisition", {}).get("image_type")
                    if lic in ["CC-0", "CC-BY"] and mod == "dermoscopic":
                        print(f"  [+] Collection {col_id:>4} ({col_name[:40]:<40}): {cnt:>6} images | {lic} | {mod}")
        except Exception:
            pass


if __name__ == "__main__":
    audit_pad_ufes_distribution()
    audit_permissive_dermoscopy_distribution()
