#!/usr/bin/env python3
"""Inspect collection 406 (PAD-UFES-20) and Mendeley Data API files."""

import json
import urllib.request


def check_isic_collection_406():
    print("=== Checking ISIC Collection 406 ===")
    headers = {"User-Agent": "SkinAI/1.0"}
    url = "https://api.isic-archive.com/api/v2/collections/406/"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"Name: {data.get('name')}")
            print(f"Description: {data.get('description')}")
            print(f"DOI: {data.get('doi')}")
            print(f"Public: {data.get('public')}")
            print(f"Num Images: {data.get('images_count', data.get('num_images', 'N/A'))}")
    except Exception as e:
        print(f"Collection 406 check error: {e}")

    # Inspect first 3 images from collection 406
    try:
        url = "https://api.isic-archive.com/api/v2/images/?collection=406&limit=3"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"Total images in collection 406 query: {data.get('count')}")
            for img in data.get("results", []):
                meta = img.get("metadata", {}).get("clinical", {})
                acq = img.get("metadata", {}).get("acquisition", {})
                print(f"  ISIC ID: {img.get('isic_id')}, License: {img.get('copyright_license')}, "
                      f"Diag: {meta.get('diagnosis')}, Patient: {meta.get('patient_id')}, "
                      f"Modality: {acq.get('image_type')}")
    except Exception as e:
        print(f"Images in collection 406 check error: {e}")


def check_mendeley_files():
    print("\n=== Checking Mendeley Public API files ===")
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://data.mendeley.com/public-api/datasets/zr7vgbcyr2/files?version=1"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"Mendeley files returned: {len(data)}")
            for item in data:
                print(f"  File: {item.get('filename')}, Size: {item.get('size')} bytes, Download: {item.get('content_details', {}).get('download_url')}")
    except Exception as e:
        print(f"Mendeley check error: {e}")


if __name__ == "__main__":
    check_isic_collection_406()
    check_mendeley_files()
