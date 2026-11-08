#!/usr/bin/env python3
"""Detailed audit of ISIC Archive clinical images and diagnostic breakdown."""

import json
import sys
import urllib.parse
import urllib.request
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
headers = {"User-Agent": "SkinAI/1.0"}


def get_json(url):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print("=" * 70)
    print(" ISIC Archive Exact Diagnostic Breakdown for 'clinical: close-up'")
    print("=" * 70)

    # 1. Check all image types under CC-0 and CC-BY
    image_types = [
        "clinical: close-up",
        "clinical: overview",
        "TBP: tile: 3D",
        "TBP: tile: 2D",
        "dermoscopic"
    ]

    print("\n[1] Permissive Volumes (CC-0 + CC-BY) by Modality:")
    for itype in image_types:
        q = f'image_type:"{itype}" AND (copyright_license:"CC-0" OR copyright_license:"CC-BY")'
        url = f"https://api.isic-archive.com/api/v2/images/search/?query={urllib.parse.quote(q)}&limit=1"
        try:
            d = get_json(url)
            print(f"    - {itype:<22}: {d.get('count', 0):>7} images")
        except Exception as e:
            print(f"    - {itype:<22}: Error {e}")

    # Also check CC-BY-NC close-up
    q_nc = 'image_type:"clinical: close-up" AND copyright_license:"CC-BY-NC"'
    url_nc = f"https://api.isic-archive.com/api/v2/images/search/?query={urllib.parse.quote(q_nc)}&limit=1"
    d_nc = get_json(url_nc)
    print(f"    - clinical: close-up (CC-BY-NC): {d_nc.get('count', 0):>5} images (research_only)")

    # 2. Sample 100 images from the 3,597 permissive close-up images to inspect diagnoses
    print("\n[2] Sampling 150 images from the 3,597 permissive close-up images:")
    q_sample = 'image_type:"clinical: close-up" AND (copyright_license:"CC-0" OR copyright_license:"CC-BY")'
    sample_url = f"https://api.isic-archive.com/api/v2/images/search/?query={urllib.parse.quote(q_sample)}&limit=150"
    sample_data = get_json(sample_url)
    results = sample_data.get("results", [])
    print(f"    Fetched {len(results)} sample records.")

    diag1_counts = Counter()
    diag3_counts = Counter()
    benign_mal = Counter()

    for item in results:
        meta = item.get("metadata", {}).get("clinical", {})
        d1 = meta.get("diagnosis_1", "None")
        d3 = meta.get("diagnosis_3", meta.get("diagnosis_2", meta.get("diagnosis", "None")))
        bm = meta.get("benign_malignant", meta.get("diagnosis_1", "Unknown"))
        diag1_counts[d1] += 1
        diag3_counts[d3] += 1
        benign_mal[bm] += 1

    print("\n    Breakdown by high-level diagnosis_1 in sample:")
    for k, v in diag1_counts.most_common():
        print(f"      * {k:<30}: {v:>3} ({v/len(results)*100:.1f}%)")

    print("\n    Breakdown by detailed diagnosis_3 in sample:")
    for k, v in diag3_counts.most_common(12):
        print(f"      * {k:<30}: {v:>3} ({v/len(results)*100:.1f}%)")

    # 3. Check search query using diagnosis_3 or diagnosis_1
    print("\n[3] Testing exact search queries on ISIC API:")
    test_queries = [
        ("Malignant", 'image_type:"clinical: close-up" AND (copyright_license:"CC-0" OR copyright_license:"CC-BY") AND diagnosis_1:"Malignant"'),
        ("Benign", 'image_type:"clinical: close-up" AND (copyright_license:"CC-0" OR copyright_license:"CC-BY") AND diagnosis_1:"Benign"'),
        ("Melanoma (diag 3)", 'image_type:"clinical: close-up" AND (copyright_license:"CC-0" OR copyright_license:"CC-BY") AND diagnosis_3:"Melanoma"'),
        ("BCC (diag 3)", 'image_type:"clinical: close-up" AND (copyright_license:"CC-0" OR copyright_license:"CC-BY") AND diagnosis_3:"Basal cell carcinoma"'),
        ("SCC (diag 3)", 'image_type:"clinical: close-up" AND (copyright_license:"CC-0" OR copyright_license:"CC-BY") AND diagnosis_3:"Squamous cell carcinoma"'),
        ("Nevus (diag 3)", 'image_type:"clinical: close-up" AND (copyright_license:"CC-0" OR copyright_license:"CC-BY") AND diagnosis_3:"Nevus"'),
        ("Actinic Keratosis (diag 3)", 'image_type:"clinical: close-up" AND (copyright_license:"CC-0" OR copyright_license:"CC-BY") AND diagnosis_3:"Solar or actinic keratosis"'),
        ("Seborrheic Keratosis (diag 3)", 'image_type:"clinical: close-up" AND (copyright_license:"CC-0" OR copyright_license:"CC-BY") AND diagnosis_3:"Seborrheic keratosis"'),
    ]

    for label, q in test_queries:
        url = f"https://api.isic-archive.com/api/v2/images/search/?query={urllib.parse.quote(q)}&limit=1"
        try:
            d = get_json(url)
            print(f"    - {label:<32}: {d.get('count', 0):>6} images")
        except Exception as e:
            print(f"    - {label:<32}: Error {e}")


if __name__ == "__main__":
    main()
