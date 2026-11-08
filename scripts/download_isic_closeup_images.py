#!/usr/bin/env python3
"""Download the 1,299 new permissive ISIC clinical close-up images from AWS S3."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import sys
import time
import urllib.request
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_one(row_dict: dict, out_dir: Path) -> dict:
    image_id = row_dict["image_id"]
    source_url = row_dict["source_url"]
    dest_path = out_dir / f"{image_id}.jpg"

    # Skip if exists and valid
    if dest_path.exists() and dest_path.stat().st_size > 0:
        hasher = hashlib.sha256()
        hasher.update(dest_path.read_bytes())
        return {
            "image_id": image_id,
            "status": "skipped_existing",
            "size_bytes": dest_path.stat().st_size,
            "sha256": hasher.hexdigest(),
        }

    temp_path = dest_path.with_suffix(".tmp")
    headers = {"User-Agent": "SkinAI/1.0"}
    req = urllib.request.Request(source_url, headers=headers)

    try:
        hasher = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=20) as resp, open(temp_path, "wb") as f:
            while chunk := resp.read(64 * 1024):
                f.write(chunk)
                hasher.update(chunk)
        temp_path.replace(dest_path)
        return {
            "image_id": image_id,
            "status": "downloaded",
            "size_bytes": dest_path.stat().st_size,
            "sha256": hasher.hexdigest(),
        }
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return {
            "image_id": image_id,
            "status": "failed",
            "size_bytes": 0,
            "error": str(e),
        }


def main():
    print("=" * 70)
    print(" Downloading New ISIC Clinical Close-up Images (CC-0 / CC-BY)")
    print("=" * 70)

    manifest_path = PROJECT_ROOT / "data" / "metadata" / "isic_clinical_manifest.csv"
    if not manifest_path.exists():
        print(f"[-] Manifest not found: {manifest_path}")
        sys.exit(1)

    df = pd.read_csv(manifest_path, keep_default_na=False)
    # Filter only new images (not in PAD-UFES-20)
    new_df = df[~df["is_existing_pad_ufes"]].copy()
    print(f"[*] Total new images to download: {len(new_df)}")

    out_dir = PROJECT_ROOT / "data" / "raw" / "isic_clinical"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Target directory: {out_dir}")

    items = new_df.to_dict("records")
    results = []
    t0 = time.time()
    success_count = 0
    fail_count = 0
    total_bytes = 0

    max_workers = 8
    print(f"[*] Starting multi-threaded download (workers={max_workers})...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, item, out_dir): item for item in items}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            results.append(res)
            if res["status"] in ("downloaded", "skipped_existing"):
                success_count += 1
                total_bytes += res.get("size_bytes", 0)
            else:
                fail_count += 1

            if i % 100 == 0 or i == len(items):
                speed_mb = (total_bytes / (1024 * 1024)) / max(time.time() - t0, 0.1)
                print(
                    f"    Progress: {i:>4}/{len(items)} | Downloaded: {success_count} | Failed: {fail_count} | Size: {total_bytes / (1024*1024):.1f} MB ({speed_mb:.1f} MB/s)",
                    end="\r" if i < len(items) else "\n"
                )

    print("\n" + "=" * 70)
    print(" Download Summary:")
    print(f"[*] Total processed  : {len(results)}")
    print(f"[*] Success (ready)  : {success_count}")
    print(f"[*] Failed           : {fail_count}")
    print(f"[*] Total footprint  : {total_bytes / (1024*1024):.2f} MB")
    print(f"[*] Time elapsed     : {time.time() - t0:.1f}s")
    print("=" * 70)

    # Save download log with sha256
    res_df = pd.DataFrame(results)
    log_path = PROJECT_ROOT / "data" / "metadata" / "isic_clinical_download_log.csv"
    res_df.to_csv(log_path, index=False)
    print(f"[+] Download log saved: {log_path}")


if __name__ == "__main__":
    main()
