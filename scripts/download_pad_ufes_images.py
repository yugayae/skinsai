#!/usr/bin/env python3
"""Download PAD-UFES-20 images using DatasetDownloader with safety checks."""

from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.downloader import DatasetDownloader  # noqa: E402
from ml.datasets.manifest import DatasetManifest  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main():
    manifest_path = Path("data/splits/smartphone_clinical/manifest.json")
    if not manifest_path.exists():
        print(f"[-] Manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"[*] Loading manifest from {manifest_path}...")
    manifest = DatasetManifest.from_json(manifest_path)
    print(f"[*] Items to download: {len(manifest.items)}")

    output_dir = Path("data/raw/pad_ufes_20")
    downloader = DatasetDownloader(
        output_dir=output_dir,
        min_free_space_bytes=5 * 1024 * 1024 * 1024,  # 5 GB safety reserve
        max_workers=8,
        timeout=30,
    )

    is_safe, free_bytes = downloader.check_disk_space()
    print(f"[*] Disk space check: {free_bytes / (1024**3):.2f} GB free on target drive.")
    if not is_safe:
        print("[-] Insufficient disk space!")
        sys.exit(1)

    print("[*] Starting concurrent download (8 workers)...")
    start_t = time.time()
    last_print = [time.time()]

    def on_progress(completed: int, total: int, current_id: str):
        now = time.time()
        if now - last_print[0] > 3.0 or completed == total:
            last_print[0] = now
            pct = (completed / total) * 100
            print(f"    Progress: {completed:>4}/{total} ({pct:>5.1f}%) | Current: {current_id}")

    updated_manifest, stats = downloader.download_manifest(
        manifest, progress_callback=on_progress
    )

    elapsed = time.time() - start_t
    print("\n" + "=" * 60)
    print(" DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total items      : {stats.total_items}")
    print(f"Downloaded       : {stats.downloaded}")
    print(f"Skipped (exists) : {stats.skipped_existing}")
    print(f"Failed           : {stats.failed}")
    print(f"Total downloaded : {stats.total_bytes_downloaded / (1024*1024):.2f} MB")
    print(f"Elapsed time     : {elapsed:.2f} s")
    if stats.errors:
        print(f"[!] Errors ({len(stats.errors)}): {stats.errors[:5]}")

    # Save updated manifest with local paths
    updated_manifest.save_json(manifest_path)
    updated_manifest.save_csv(Path("data/splits/smartphone_clinical/manifest.csv"))
    print("[+] Updated manifest saved with local image paths.")


if __name__ == "__main__":
    main()
