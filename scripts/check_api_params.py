#!/usr/bin/env python3
"""Inspect collection 406 endpoints and Mendeley API response structure."""

import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")


def check_isic_406():
    headers = {"User-Agent": "SkinAI/1.0"}
    # Check collection details
    url = "https://api.isic-archive.com/api/v2/collections/406/"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        col = json.loads(resp.read().decode("utf-8"))
        print(f"Collection: ID {col.get('id')}, Name: {col.get('name')}")
        print(f"Images count in collection: {col.get('num_images') or col.get('images_count') or col.get('count')}")
        print(f"Keys in collection object: {list(col.keys())}")

    # Check search images in collections
    # Test endpoints
    test_urls = [
        "https://api.isic-archive.com/api/v2/images/?collections=406&limit=2",
        "https://api.isic-archive.com/api/v2/images/?collections__id=406&limit=2",
        "https://api.isic-archive.com/api/v2/images/search/?collections=406&limit=2",
        "https://api.isic-archive.com/api/v2/images/?collections=406"
    ]
    for u in test_urls:
        try:
            r = urllib.request.Request(u, headers=headers)
            with urllib.request.urlopen(r) as resp:
                d = json.loads(resp.read().decode("utf-8"))
                print(f"URL: {u} -> count={d.get('count')}")
        except Exception as e:
            print(f"URL: {u} -> error {e}")


def check_mendeley():
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://data.mendeley.com/public-api/datasets/zr7vgbcyr2/files?version=1"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        print(f"Mendeley raw type: {type(data)}")
        if isinstance(data, list):
            for i, x in enumerate(data[:5]):
                print(f"  Item {i}: {type(x)}, {x}")
        elif isinstance(data, dict):
            print(f"  Dict keys: {list(data.keys())}")
            print(f"  Dict sample: {data}")
        else:
            print(f"  Content preview: {raw[:200]}")


if __name__ == "__main__":
    check_isic_406()
    check_mendeley()
