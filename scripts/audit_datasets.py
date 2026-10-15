#!/usr/bin/env python3
"""Audit script for SkinAI datasets (ISIC Archive and PAD-UFES-20).

Queries API metadata and inspects license breakdowns and diagnostic categories.
"""

import json
import urllib.request
import urllib.parse


def audit_isic_archive():
    print("=" * 60)
    print(" 1. Auditing ISIC Archive (v2 API)")
    print("=" * 60)
    base_url = "https://api.isic-archive.com/api/v2"

    # Test basic connectivity
    try:
        req = urllib.request.Request(
            f"{base_url}/images/?limit=1",
            headers={"User-Agent": "SkinAI-Research-Agent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            total_count = data.get("count", 0)
            print(f"[+] Connected to ISIC API v2. Total archive images: {total_count}")
    except Exception as e:
        print(f"[-] Failed to query ISIC API: {e}")
        return

    # Check licenses via search/filter queries if supported
    # In ISIC v2, query parameter `query` supports search syntax e.g. copyright_license:CC-0
    licenses = ["CC-0", "CC-BY", "CC-BY-NC", "CC-BY-NC-ND"]
    license_counts = {}

    for lic in licenses:
        query_str = urllib.parse.quote(f"copyright_license:{lic}")
        url = f"{base_url}/images/?query={query_str}&limit=1"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SkinAI-Research-Agent/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                cnt = res.get("count", 0)
                license_counts[lic] = cnt
                print(f"  - License {lic:<15}: {cnt:>6} images")
        except Exception as e:
            print(f"  - License {lic:<15}: query error ({e})")

    # Audit common diagnoses for commercial subset (CC-0 OR CC-BY)
    diagnoses = [
        ("melanoma", "mel"),
        ("nevus", "nv"),
        ("basal cell carcinoma", "bcc"),
        ("actinic keratosis", "akiec"),
        ("seborrheic keratosis", "bkl"),
        ("dermatofibroma", "df"),
        ("vascular lesion", "vasc")
    ]
    print("\n[*] Auditing permissive diagnoses (CC-0 / CC-BY):")
    for diag_name, code in diagnoses:
        # Search query matching diagnosis and permissive license
        for lic in ["CC-0", "CC-BY"]:
            q = urllib.parse.quote(f"copyright_license:{lic} AND diagnosis:\"{diag_name}\"")
            url = f"{base_url}/images/?query={q}&limit=1"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SkinAI-Research-Agent/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    cnt = res.get("count", 0)
                    print(f"  - [{lic}] {diag_name:<25} ({code}): {cnt:>5} images")
            except Exception:
                pass


def audit_pad_ufes_20():
    print("\n" + "=" * 60)
    print(" 2. Auditing PAD-UFES-20 (Smartphone Clinical Profile)")
    print("=" * 60)
    # Mendeley data direct link or API
    # Mendeley Data ID: zr7vgbcyr2 version 1
    mendeley_api = "https://data.mendeley.com/api/datasets/zr7vgbcyr2?version=1"
    try:
        req = urllib.request.Request(mendeley_api, headers={"User-Agent": "SkinAI-Research-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[+] Dataset Title: {data.get('name')}")
            print(f"[+] License: {data.get('licence', {}).get('name')}")
            files = data.get("files", [])
            print(f"[+] Available Files ({len(files)} items):")
            for f in files:
                fname = f.get("filename", "")
                fsize_mb = f.get("size", 0) / (1024 * 1024)
                print(f"  - {fname:<30} ({fsize_mb:>8.2f} MB) -> Download: {f.get('download_url')}")
    except Exception as e:
        print(f"[-] Mendeley Data API query failed: {e}")


if __name__ == "__main__":
    audit_isic_archive()
    audit_pad_ufes_20()
