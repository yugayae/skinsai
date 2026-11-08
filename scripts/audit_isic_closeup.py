#!/usr/bin/env python3
"""Audit ISIC Archive API for clinical close-up images under CC-0 / CC-BY licenses."""

import json
import sys
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
headers = {"User-Agent": "SkinAI/1.0"}


def query_count(query_str: str) -> int:
    q_enc = urllib.parse.quote(query_str)
    url = f"https://api.isic-archive.com/api/v2/images/search/?query={q_enc}&limit=1"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("count", 0)
    except Exception as e:
        print(f"[-] Error querying '{query_str}': {e}")
        return -1


def main():
    print("=" * 70)
    print(" ISIC Archive Clinical Close-up Permissive License Audit")
    print(" Target Modality: image_type == 'clinical: close-up'")
    print(" Target Licenses: CC-0, CC-BY (commercial_ok)")
    print("=" * 70)

    # 1. Total overview by license
    print("\n[1] Overall Volume by License for 'clinical: close-up':")
    licenses = ["CC-0", "CC-BY", "CC-BY-NC"]
    for lic in licenses:
        q = f'image_type:"clinical: close-up" AND copyright_license:"{lic}"'
        c = query_count(q)
        print(f"    - {lic:<10}: {c} images")

    # 2. Total permissive clinical close-up
    permissive_q = 'image_type:"clinical: close-up" AND (copyright_license:"CC-0" OR copyright_license:"CC-BY")'
    total_perm = query_count(permissive_q)
    print(f"\n[+] Total Permissive (CC-0 + CC-BY) Close-up Images: {total_perm}")

    # 3. Breakdown by diagnostic categories for our 6 classes
    diagnoses = [
        ("MEL (Melanoma)", 'diagnosis:"melanoma"'),
        ("BCC (Basal cell carcinoma)", 'diagnosis:"basal cell carcinoma"'),
        ("SCC (Squamous cell carcinoma)", 'diagnosis:"squamous cell carcinoma"'),
        ("NEV (Nevus)", 'diagnosis:"nevus"'),
        ("ACK (Actinic keratosis)", 'diagnosis:"actinic keratosis"'),
        ("SEK (Seborrheic keratosis)", 'diagnosis:"seborrheic keratosis"'),
    ]

    print("\n[2] Permissive Close-up Volume for Target 6 Classes:")
    for label, diag_q in diagnoses:
        full_q = f'image_type:"clinical: close-up" AND (copyright_license:"CC-0" OR copyright_license:"CC-BY") AND {diag_q}'
        c = query_count(full_q)
        # Also check with histopathology confirmation
        histo_q = f'{full_q} AND diagnosis_confirm_type:"histopathology"'
        c_histo = query_count(histo_q)
        print(f"    * {label:<30}: Total = {c:<6} | Confirmed Biopsy = {c_histo}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
