#!/usr/bin/env python3
"""Inspect schema fields for clinical diagnosis and query dermoscopic CC-0/CC-BY."""

import json
import sys
import urllib.request
import urllib.parse

sys.stdout.reconfigure(encoding="utf-8")
headers = {"User-Agent": "SkinAI/1.0"}

# Test search query combining license and dermoscopic modality
queries = [
    'copyright_license:"CC-0" AND image_type:"dermoscopic"',
    'copyright_license:"CC-BY" AND image_type:"dermoscopic"',
    'copyright_license:"CC-BY-NC" AND image_type:"dermoscopic"'
]

for q in queries:
    q_enc = urllib.parse.quote(q)
    url = f"https://api.isic-archive.com/api/v2/images/search/?query={q_enc}&limit=1"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        d = json.loads(resp.read().decode("utf-8"))
        print(f"Permissive dermoscopic [{q}]: {d.get('count')} images")

# Find an image that has non-null diagnosis to see the exact field name
url = "https://api.isic-archive.com/api/v2/images/search/?query=benign_malignant:malignant&limit=1"
try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        d = json.loads(resp.read().decode("utf-8"))
        print(f"benign_malignant:malignant count = {d.get('count')}")
        if d.get("results"):
            item = d["results"][0]
            print("Full metadata of malignant sample:")
            print(json.dumps(item.get("metadata"), indent=2))
except Exception as e:
    print(f"Error checking malignant sample: {e}")
