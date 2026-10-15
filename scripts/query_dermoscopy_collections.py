#!/usr/bin/env python3
"""Query ISIC collections for CC-0 / CC-BY dermoscopy."""

import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
headers = {"User-Agent": "SkinAI/1.0"}

# Get first 50 collections
url = "https://api.isic-archive.com/api/v2/collections/?limit=100"
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))

print(f"Total collections: {data['count']}")
# Check collections with > 500 images
large_cols = [c for c in data["results"] if c["public"]]
print(f"Total public collections: {len(large_cols)}")

for c in large_cols:
    col_id = c["id"]
    col_name = c["name"]
    s_url = f"https://api.isic-archive.com/api/v2/images/search/?collections={col_id}&limit=1"
    try:
        s_req = urllib.request.Request(s_url, headers=headers)
        with urllib.request.urlopen(s_req) as s_resp:
            s_data = json.loads(s_resp.read().decode("utf-8"))
            cnt = s_data.get("count", 0)
            if cnt >= 200:
                sample = s_data["results"][0] if s_data.get("results") else {}
                lic = sample.get("copyright_license")
                mod = sample.get("metadata", {}).get("acquisition", {}).get("image_type")
                print(f"  Col {col_id:>4}: {cnt:>6} imgs | Lic: {str(lic):<10} | Mod: {str(mod):<12} | Name: {col_name[:40]}")
    except Exception as e:
        print(f"  Col {col_id:>4}: error {e}")
