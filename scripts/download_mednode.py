"""Download and extract MED-NODE dataset into data/raw/mednode."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import ssl
import sys
import urllib.request
import zipfile

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "mednode"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

MEDNODE_URL = "https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/complete_mednode_dataset.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print(" Downloading and Ingesting MED-NODE Dataset")
    print(" Source: University Medical Center Groningen (UMCG), Netherlands")
    print("=" * 80)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download
    print(f"[*] Fetching archive from {MEDNODE_URL}...")
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(MEDNODE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as resp:
        zip_bytes = resp.read()

    print(f"[+] Downloaded {len(zip_bytes):,} bytes ({len(zip_bytes)/(1024*1024):.2f} MB).")
    zip_sha256 = hashlib.sha256(zip_bytes).hexdigest()
    print(f"[*] Archive SHA-256: {zip_sha256}")

    # 2. Extract
    print(f"[*] Extracting files to {TARGET_DIR}...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for member in z.namelist():
            # Skip directory entries
            if member.endswith("/"):
                continue
            # Members are like complete_mednode_dataset/melanoma/136733.jpg
            parts = member.split("/")
            if len(parts) >= 3:
                category = parts[1].lower()  # 'melanoma' or 'naevus'
                filename = parts[2]
                dest_cat_dir = TARGET_DIR / category
                dest_cat_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_cat_dir / filename
                with z.open(member) as src, open(dest_file, "wb") as dst:
                    dst.write(src.read())

    # 3. Audit extracted files
    records = []
    for cat in ["melanoma", "naevus"]:
        cat_dir = TARGET_DIR / cat
        if not cat_dir.exists():
            continue
        for img_path in sorted(cat_dir.glob("*.jpg")):
            file_hash = sha256_file(img_path)
            with Image.open(img_path) as img:
                w, h = img.size
                fmt = img.format

            is_mel = (cat == "melanoma")
            records.append({
                "image_id": f"mednode_{cat}_{img_path.stem}",
                "dataset_source": "MED-NODE",
                "original_filename": img_path.name,
                "file_path": str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "category": cat,
                "diagnostic_confirmation": "histopathology_biopsy",
                "subtype_target": "mel" if is_mel else "nev",
                "triage_target": 1 if is_mel else 0,  # 1 = Malignant, 0 = Benign
                "width": w,
                "height": h,
                "format": fmt,
                "sha256": file_hash,
                "patient_id": f"mednode_patient_{cat}_{img_path.stem}",
                "lesion_id": f"mednode_lesion_{cat}_{img_path.stem}",
                "license": "Research with attribution (Giotis et al., 2015)",
                "license_scope": "research_only",
            })

    df = pd.DataFrame(records)
    out_manifest = METADATA_DIR / "mednode_manifest.csv"
    df.to_csv(out_manifest, index=False)

    print(f"\n[+] Extracted and verified {len(df)} images:")
    print(f"    - Melanoma (biopsy-confirmed): {len(df[df['category'] == 'melanoma'])} images")
    print(f"    - Naevus (clinically/histopathologically confirmed): {len(df[df['category'] == 'naevus'])} images")
    print(f"[+] Manifest saved to {out_manifest}")
    print("=" * 80)


if __name__ == "__main__":
    main()
