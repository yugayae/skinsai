#!/usr/bin/env python3
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
headers = {"User-Agent": "SkinAI/1.0"}

for cid in [294, 470, 482, 70, 485, 399]:
    url = f"https://api.isic-archive.com/api/v2/images/search/?collections={cid}&limit=10"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        pts = sum(1 for it in data["results"] if it.get("metadata", {}).get("clinical", {}).get("patient_id"))
        les = sum(1 for it in data["results"] if it.get("metadata", {}).get("clinical", {}).get("lesion_id"))
        d3 = [it.get("metadata", {}).get("clinical", {}).get("diagnosis_3") for it in data["results"]]
        print(f"Col {cid:>4}: total={data.get('count', 0):>6} | pts={pts}/10 | les={les}/10 | diag_sample={d3[:3]}")
    except Exception as e:
        print(f"Col {cid}: error {e}")
