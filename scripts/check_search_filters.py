#!/usr/bin/env python3
"""Detailed inspection of ISIC search endpoint and collection 406 (PAD-UFES-20)."""

import json
import sys
import urllib.request
import urllib.parse

sys.stdout.reconfigure(encoding="utf-8")
headers = {"User-Agent": "SkinAI/1.0"}


def inspect_pad_ufes_in_isic():
    print("=== PAD-UFES-20 in ISIC Archive (Collection 406) ===")
    url = "https://api.isic-archive.com/api/v2/images/search/?collections=406&limit=5"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"Total images in PAD-UFES-20: {data.get('count')}")
        for i, item in enumerate(data.get("results", [])):
            meta = item.get("metadata", {}).get("clinical", {})
            acq = item.get("metadata", {}).get("acquisition", {})
            print(f"  [{i+1}] isic_id: {item.get('isic_id')}, "
                  f"license: {item.get('copyright_license')}, "
                  f"diagnosis: {meta.get('diagnosis')}, "
                  f"benign_malignant: {meta.get('benign_malignant')}, "
                  f"patient_id: {meta.get('patient_id')}, "
                  f"lesion_id: {meta.get('lesion_id')}, "
                  f"image_type: {acq.get('image_type')}")


def inspect_search_filters():
    print("\n=== Testing ISIC Search Query Filters ===")
    # Test query by copyright_license
    queries = [
        "copyright_license:\"CC-0\"",
        "copyright_license:\"CC-BY\"",
        "copyright_license:\"CC-BY-NC\"",
        "diagnosis:\"melanoma\"",
        "image_type:\"dermoscopic\"",
        "image_type:\"clinical\""
    ]
    for q in queries:
        q_enc = urllib.parse.quote(q)
        url = f"https://api.isic-archive.com/api/v2/images/search/?query={q_enc}&limit=1"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                print(f"  Query: {q:<35} -> Count: {data.get('count')}")
        except Exception as e:
            print(f"  Query: {q:<35} -> Error: {e}")


if __name__ == "__main__":
    inspect_pad_ufes_in_isic()
    inspect_search_filters()
