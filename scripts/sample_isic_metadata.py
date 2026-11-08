#!/usr/bin/env python3
"""Fetch a sample CC-0 dermoscopic image metadata from ISIC."""

import json
import urllib.request
import urllib.parse

headers = {"User-Agent": "SkinAI/1.0"}
q_enc = urllib.parse.quote('copyright_license:"CC-0" AND image_type:"dermoscopic"')
url = f"https://api.isic-archive.com/api/v2/images/search/?query={q_enc}&limit=5"
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    for item in data.get("results", []):
        print(f"ISIC ID: {item.get('isic_id')}")
        print("Metadata:", json.dumps(item.get("metadata"), indent=2))
        print("-" * 40)
