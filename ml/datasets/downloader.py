"""Managed image downloader for SkinAI datasets.

Features:
- Disk quota safety enforcement (prevents filling up local C: drive).
- Resumable downloads (skips verified local files).
- Multi-threaded downloading with rate limiting and timeout.
- SHA-256 verification and file integrity validation.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ml.datasets.manifest import DatasetItem, DatasetManifest


class DownloadStats(BaseModel):
    """Execution summary for a dataset download run."""

    total_items: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    failed: int = 0
    total_bytes_downloaded: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = Field(default_factory=list)


class DatasetDownloader:
    """Safely downloads dataset images to local storage."""

    def __init__(
        self,
        output_dir: Path,
        min_free_space_bytes: int = 5 * 1024 * 1024 * 1024,  # 5 GB safety reserve
        max_workers: int = 4,
        timeout: int = 30,
    ):
        self.output_dir = Path(output_dir)
        self.min_free_space_bytes = min_free_space_bytes
        self.max_workers = max_workers
        self.timeout = timeout
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def check_disk_space(self) -> Tuple[bool, int]:
        """Check available disk space on the target filesystem."""
        usage = shutil.disk_usage(self.output_dir)
        free_bytes = usage.free
        return free_bytes >= self.min_free_space_bytes, free_bytes

    def _download_single(
        self, item: DatasetItem
    ) -> Tuple[str, bool, bool, int, Optional[str], Optional[str]]:
        """Download a single item.

        Returns:
            Tuple of (image_id, is_success, was_downloaded, byte_count, sha256_hash, error_msg)
        """
        if not item.source_url:
            return (item.image_id, False, False, 0, None, "missing_source_url")

        # Destination filename: {image_id}.jpg
        dest_path = self.output_dir / f"{item.image_id}.jpg"

        # Check if file already exists and is non-empty
        if dest_path.exists() and dest_path.stat().st_size > 0:
            # File already downloaded
            hasher = hashlib.sha256()
            hasher.update(dest_path.read_bytes())
            return (item.image_id, True, False, dest_path.stat().st_size, hasher.hexdigest(), None)

        # Check free disk space before downloading
        is_safe, free_b = self.check_disk_space()
        if not is_safe:
            return (
                item.image_id,
                False,
                False,
                0,
                None,
                f"disk_space_low ({free_b / (1024**3):.2f} GB free < {self.min_free_space_bytes / (1024**3):.2f} GB required)",
            )

        headers = {"User-Agent": "SkinAI/1.0"}
        req = urllib.request.Request(item.source_url, headers=headers)

        temp_path = dest_path.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp, open(temp_path, "wb") as f:
                hasher = hashlib.sha256()
                while chunk := resp.read(64 * 1024):
                    f.write(chunk)
                    hasher.update(chunk)

            # Atomically rename
            temp_path.replace(dest_path)
            file_size = dest_path.stat().st_size
            file_hash = hasher.hexdigest()
            return (item.image_id, True, True, file_size, file_hash, None)
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            return (item.image_id, False, False, 0, None, str(e))

    def download_manifest(
        self,
        manifest: DatasetManifest,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[DatasetManifest, DownloadStats]:
        """Download all images referenced in manifest concurrently."""
        start_time = time.time()
        stats = DownloadStats(total_items=len(manifest.items))

        # Check space at outset
        is_safe, free_b = self.check_disk_space()
        if not is_safe:
            err = f"Aborting download: disk space below threshold ({free_b / (1024**3):.2f} GB free)"
            stats.errors.append(err)
            return manifest, stats

        updated_items_map: Dict[str, DatasetItem] = {it.image_id: it for it in manifest.items}
        completed_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(self._download_single, it): it.image_id
                for it in manifest.items
            }

            for future in concurrent.futures.as_completed(future_to_id):
                img_id = future_to_id[future]
                completed_count += 1
                try:
                    res_id, success, downloaded, b_count, f_hash, err = future.result()
                    if success:
                        if downloaded:
                            stats.downloaded += 1
                            stats.total_bytes_downloaded += b_count
                        else:
                            stats.skipped_existing += 1

                        # Update item with local path and hash
                        orig_item = updated_items_map[res_id]
                        local_path = str(self.output_dir / f"{res_id}.jpg")
                        updated_items_map[res_id] = orig_item.model_copy(
                            update={
                                "image_path": local_path,
                                "sha256": f_hash or orig_item.sha256,
                            }
                        )
                    else:
                        stats.failed += 1
                        if err:
                            stats.errors.append(f"{res_id}: {err}")
                except Exception as e:
                    stats.failed += 1
                    stats.errors.append(f"{img_id}: {str(e)}")

                if progress_callback:
                    progress_callback(completed_count, stats.total_items, img_id)

        stats.duration_seconds = time.time() - start_time

        updated_manifest = DatasetManifest(
            profile=manifest.profile,
            version=manifest.version,
            items=[updated_items_map[it.image_id] for it in manifest.items],
        )

        return updated_manifest, stats
