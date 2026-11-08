#!/usr/bin/env python3
"""Fetch metadata for PAD-UFES-20 (Collection 406) from ISIC Archive API.

Downloads metadata with proper pagination (limit=250) and saves to data/metadata/pad_ufes_20.json.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def fetch_pad_ufes_metadata(output_path: Path):
    print("=" * 60)
    print(" Fetching PAD-UFES-20 metadata from ISIC Archive API (Collection 406)")
    print("=" * 60, flush=True)

    headers = {"User-Agent": "SkinAI-Dataset-Engine/1.0"}
    base_url = "https://api.isic-archive.com/api/v2/images/search/?collections=406&limit=250"
    all_records = []
    next_url = base_url
    page = 1

    start_time = time.time()
    while next_url:
        print(f"[*] Fetching page {page}... ", end="", flush=True)
        req = urllib.request.Request(next_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = data.get("results", [])
        all_records.extend(results)
        print(f"received {len(results)} items (total: {len(all_records)})", flush=True)

        next_url = data.get("next")
        page += 1

    elapsed = time.time() - start_time
    print(f"\n[+] Finished in {elapsed:.2f}s. Total records: {len(all_records)}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"[+] Saved metadata to {output_path.resolve()}", flush=True)
    return all_records


if __name__ == "__main__":
    out_file = Path("data/metadata/pad_ufes_20.json")
    fetch_pad_ufes_metadata(out_file)
