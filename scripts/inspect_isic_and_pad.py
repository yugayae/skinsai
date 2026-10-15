#!/usr/bin/env python3
"""Detailed inspection of ISIC API collections and PAD-UFES-20 data access."""

import json
import urllib.request
import urllib.parse


def check_isic():
    print("--- ISIC API Collections & Endpoints ---")
    headers = {"User-Agent": "SkinAI/1.0"}

    # Check collections
    try:
        url = "https://api.isic-archive.com/api/v2/collections/?limit=20"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"Collections count: {data.get('count')}")
            for col in data.get("results", []):
                print(f"  ID {col.get('id')}: {col.get('name')}")
    except Exception as e:
        print(f"Collections failed: {e}")

    # Check search parameters schema
    try:
        url = "https://api.isic-archive.com/api/v2/images/search/"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"Search endpoint status: {data.get('count', 'N/A')}")
    except Exception as e:
        print(f"Search endpoint check: {e}")


def check_pad_ufes_20():
    print("\n--- PAD-UFES-20 Dataset Access ---")
    # Mendeley Data public page for PAD-UFES-20
    # DOI: 10.17632/zr7vgbcyr2.1
    # Try Mendeley public download URL
    mendeley_direct = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"
    print("Testing Mendeley S3 / API links...")

    # Also check if metadata.csv is hosted or available
    urls_to_test = [
        mendeley_direct,
        "https://data.mendeley.com/public-api/datasets/zr7vgbcyr2/files?version=1",
        "https://raw.githubusercontent.com/paaatcha/PAD-UFES-20/master/metadata.csv",
        "https://raw.githubusercontent.com/paaatcha/PAD-UFES-20/main/metadata.csv",
        "https://huggingface.co/datasets/pacheco/PAD-UFES-20"
    ]

    headers = {"User-Agent": "Mozilla/5.0"}
    for u in urls_to_test:
        try:
            req = urllib.request.Request(u, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"  [+] {u} -> Status: {resp.status}, Content-Length: {resp.headers.get('Content-Length')}")
        except Exception as e:
            print(f"  [-] {u} -> Error: {e}")


if __name__ == "__main__":
    check_isic()
    check_pad_ufes_20()
