#!/usr/bin/env python3
"""Inspect collection 406 items in ISIC Archive."""

import json
import urllib.request

headers = {"User-Agent": "SkinAI/1.0"}
url = "https://api.isic-archive.com/api/v2/images/search/?collections=406&limit=10"
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    for item in data.get("results", []):
        meta = item.get("metadata", {}).get("clinical", {})
        print(f"ISIC ID: {item.get('isic_id')}, License: {item.get('copyright_license')}")
        print("  Clinical:", meta)
        print("-" * 40)
