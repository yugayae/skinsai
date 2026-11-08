#!/usr/bin/env python3
"""Download Stanford DDI Dataset from official Azure Blob Storage using SAS Token."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "ddi"

BASE_URL = "https://aimistanforddatasets01.blob.core.windows.net/ddidiversedermatologyimages"
SAS_QUERY = "sv=2019-02-02&sr=c&sig=sS82bwsyjbkmeO6n9SZLTMWPv6nC4l1YKNqAcZkBRPs%3D&st=2026-10-23T15%3A30%3A18Z&se=2026-11-22T15%3A35%3A18Z&sp=rl"


def download_blob(blob_name: str) -> Tuple[str, bool, int]:
    blob_url = f"{BASE_URL}/{blob_name}?{SAS_QUERY}"
    dest_path = TARGET_DIR / blob_name
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return blob_name, True, dest_path.stat().st_size

    try:
        req = urllib.request.Request(blob_url, headers={"User-Agent": "SkinAI-Downloader/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        return blob_name, True, len(data)
    except Exception as e:
        print(f"[-] Error downloading {blob_name}: {e}")
        return blob_name, False, 0


def main():
    print("=" * 80)
    print(" Downloading Stanford DDI Dataset from Azure Blob Storage")
    print(f" Target Directory: {TARGET_DIR}")
    print("=" * 80)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 1. List Blobs
    print("[*] Querying container blob listing...")
    list_url = f"{BASE_URL}?restype=container&comp=list&{SAS_QUERY}"
    req = urllib.request.Request(list_url, headers={"User-Agent": "SkinAI-Downloader/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_content = resp.read()

    root = ET.fromstring(xml_content)
    blobs = [b.find("Name").text for b in root.findall(".//Blob")]
    print(f"[+] Discovered {len(blobs)} files in container.\n")

    # 2. Multithreaded Download
    t0 = time.time()
    total_bytes = 0
    success_count = 0

    print(f"[*] Downloading {len(blobs)} files with 16 parallel workers...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_blob, b): b for b in blobs}
        for i, fut in enumerate(as_completed(futures), 1):
            name, ok, size = fut.result()
            if ok:
                success_count += 1
                total_bytes += size
            if i % 100 == 0 or i == len(blobs):
                print(f"    Progress: {i}/{len(blobs)} files ({total_bytes / (1024*1024):.1f} MB, {time.time()-t0:.1f}s)...")

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"[+] Download complete: {success_count}/{len(blobs)} files downloaded successfully.")
    print(f"    Total Volume: {total_bytes / (1024*1024):.2f} MB in {elapsed:.1f}s ({total_bytes / (1024*1024) / max(elapsed, 0.1):.2f} MB/s)")
    print(f"    Metadata CSV: {TARGET_DIR / 'ddi_metadata.csv'} (exists: {(TARGET_DIR / 'ddi_metadata.csv').exists()})")
    print(f"    Image files: {len(list(TARGET_DIR.glob('*.png')))} PNG files verified.")
    print("=" * 80)


if __name__ == "__main__":
    main()
