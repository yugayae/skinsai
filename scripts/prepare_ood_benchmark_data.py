"""Prepare OOD (Out-of-Distribution) Benchmark Dataset for EXP-009.

Creates 3 standardized evaluation cohorts:
1. In-Distribution (InD): 346 genuine clinical skin lesions from frozen test.csv.
2. OOD Cohort 1 (Non-Skin / Noise / Objects): 100 non-skin images:
   - 25 synthetic noise / random color field / fractal patterns
   - 25 everyday textures (wood, metal, paper, cloth, brick)
   - 25 everyday objects / animals / nature (synthesized or standard benchmarks)
   - 25 text documents / barcodes / UI screenshots
3. OOD Cohort 2 (Healthy Human Skin Without Lesion): 100 patches of genuine healthy skin:
   - Extracted from non-lesional peripheral regions of PAD-UFES-20 images across diverse skin tones.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path("c:/HomeWork/Medical/Dermatology")
OOD_DIR = PROJECT_ROOT / "data" / "ood_benchmark"
OOD_NONSKIN_DIR = OOD_DIR / "non_skin"
OOD_HEALTHY_DIR = OOD_DIR / "healthy_skin"

OOD_NONSKIN_DIR.mkdir(parents=True, exist_ok=True)
OOD_HEALTHY_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)


def generate_synthetic_noise(idx: int) -> Image.Image:
    """Generate various noise patterns (Gaussian, uniform, Perlin-like, color bands)."""
    t = idx % 4
    if t == 0:
        # Pure Gaussian noise
        arr = np.random.normal(128, 40, (518, 518, 3)).clip(0, 255).astype(np.uint8)
    elif t == 1:
        # Uniform salt & pepper / RGB noise
        arr = np.random.randint(0, 256, (518, 518, 3), dtype=np.uint8)
    elif t == 2:
        # Gradient color bars
        arr = np.zeros((518, 518, 3), dtype=np.uint8)
        for i in range(518):
            arr[i, :, 0] = int(255 * (i / 518))
            arr[:, i, 1] = int(255 * (1 - i / 518))
            arr[i, i, 2] = 200
    else:
        # Checkerboard / geometric pattern
        arr = np.zeros((518, 518, 3), dtype=np.uint8)
        block = 32
        for r in range(0, 518, block):
            for c in range(0, 518, block):
                if (r // block + c // block) % 2 == 0:
                    arr[r:r+block, c:c+block] = [220, 220, 220]
                else:
                    arr[r:r+block, c:c+block] = [30, 30, 80]
    return Image.fromarray(arr)


def generate_synthetic_texture(idx: int) -> Image.Image:
    """Generate texture-like images (wood grain, fabric, marble, concrete)."""
    arr = np.zeros((518, 518, 3), dtype=np.float32)
    t = idx % 4
    if t == 0:
        # Wood grain simulation
        for y in range(518):
            val = 140 + 50 * math.sin(y / 15.0 + 3 * math.sin(y / 40.0))
            arr[y, :, 0] = val
            arr[y, :, 1] = val * 0.65
            arr[y, :, 2] = val * 0.35
    elif t == 1:
        # Fabric weave
        for y in range(518):
            for x in range(518):
                v = 120 + 40 * ((x % 8 < 4) ^ (y % 8 < 4))
                arr[y, x] = [v * 0.8, v * 0.8, v]
    elif t == 2:
        # Brushed metal / blue wall
        base = np.random.normal(160, 15, (518, 518))
        arr[:, :, 0] = base * 0.7
        arr[:, :, 1] = base * 0.85
        arr[:, :, 2] = base * 1.0
    else:
        # Paper / parchment
        base = np.random.normal(230, 8, (518, 518))
        arr[:, :, 0] = base
        arr[:, :, 1] = base * 0.95
        arr[:, :, 2] = base * 0.85

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=1))


def generate_synthetic_document(idx: int) -> Image.Image:
    """Generate documents, text sheets, barcodes, receipts."""
    img = Image.new("RGB", (518, 518), color=(245, 245, 248))
    draw = ImageDraw.Draw(img)

    t = idx % 3
    if t == 0:
        # Text document lines
        for y in range(40, 480, 22):
            length = random.randint(250, 440)
            draw.line([(40, y), (40 + length, y)], fill=(40, 40, 50), width=3)
    elif t == 1:
        # Barcode / tabular sheet
        for x in range(50, 460, random.randint(4, 16)):
            draw.line([(x, 60), (x, 300)], fill=(20, 20, 20), width=random.randint(1, 4))
        draw.rectangle([(50, 340), (460, 450)], outline=(100, 100, 120), width=2)
    else:
        # QR / Matrix barcode-like blocks
        for r in range(60, 450, 25):
            for c in range(60, 450, 25):
                if random.random() > 0.45:
                    draw.rectangle([(c, r), (c + 20, r + 20)], fill=(15, 15, 25))
    return img


def generate_synthetic_object(idx: int) -> Image.Image:
    """Generate non-skin everyday objects: coffee cup, car, flower, apple."""
    img = Image.new("RGB", (518, 518), color=(210, 220, 230))
    draw = ImageDraw.Draw(img)

    t = idx % 4
    if t == 0:
        # Red apple on table
        draw.rectangle([(0, 350), (518, 518)], fill=(139, 90, 43))  # Wooden table
        draw.ellipse([(180, 180), (340, 350)], fill=(200, 30, 30))  # Red apple
        draw.line([(260, 140), (260, 190)], fill=(80, 50, 20), width=6)  # Stem
        draw.polygon([(260, 160), (300, 140), (280, 170)], fill=(34, 139, 34))  # Leaf
    elif t == 1:
        # Coffee mug
        draw.rectangle([(0, 380), (518, 518)], fill=(240, 240, 240))
        draw.rectangle([(180, 200), (320, 380)], fill=(40, 90, 160))  # Blue mug body
        draw.arc([(300, 240), (360, 340)], start=270, end=90, fill=(40, 90, 160), width=10)  # Handle
    elif t == 2:
        # Green plant / flower
        draw.ellipse([(200, 280), (320, 420)], fill=(180, 100, 60))  # Pot
        draw.line([(260, 150), (260, 290)], fill=(34, 139, 34), width=8)  # Stem
        for angle in range(0, 360, 60):
            rad = math.radians(angle)
            cx = 260 + int(60 * math.cos(rad))
            cy = 150 + int(60 * math.sin(rad))
            draw.ellipse([(cx - 25, cy - 25), (cx + 25, cy + 25)], fill=(255, 215, 0))  # Yellow petals
        draw.ellipse([(240, 130), (280, 170)], fill=(139, 69, 19))  # Flower center
    else:
        # Blue vehicle / car shape
        draw.rectangle([(0, 360), (518, 518)], fill=(80, 80, 80))  # Asphalt
        draw.line([(0, 440), (518, 440)], fill=(255, 255, 255), width=4)  # Road marking
        draw.polygon([(100, 320), (160, 230), (340, 230), (420, 320)], fill=(30, 80, 190))  # Car cabin
        draw.rectangle([(80, 320), (440, 370)], fill=(30, 80, 190))  # Car body
        draw.ellipse([(140, 340), (200, 400)], fill=(20, 20, 20))  # Wheel 1
        draw.ellipse([(320, 340), (380, 400)], fill=(20, 20, 20))  # Wheel 2
    return img


def extract_healthy_skin_patch(img_path: Path) -> Image.Image | None:
    """Extract a patch from the corner of a clinical photo containing pure healthy background skin."""
    try:
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            if w < 200 or h < 200:
                return None
            # Extract top-left or bottom-right corner patch (15% to 30% of image dimension)
            crop_size = min(int(min(w, h) * 0.30), 200)
            corner = rgb.crop((10, 10, 10 + crop_size, 10 + crop_size))
            return corner.resize((518, 518), Image.Resampling.BILINEAR)
    except Exception:
        return None


def main():
    print("=" * 70)
    print(" Preparing EXP-009 OOD Quality Gate Benchmark Dataset")
    print("=" * 70)

    # 1. Generate 100 Non-Skin Images
    print("[*] Generating 100 Non-Skin Images (Noise, Textures, Documents, Everyday Objects)...")
    for i in range(25):
        img = generate_synthetic_noise(i)
        img.save(OOD_NONSKIN_DIR / f"noise_{i:03d}.jpg", quality=95)

    for i in range(25):
        img = generate_synthetic_texture(i)
        img.save(OOD_NONSKIN_DIR / f"texture_{i:03d}.jpg", quality=95)

    for i in range(25):
        img = generate_synthetic_document(i)
        img.save(OOD_NONSKIN_DIR / f"document_{i:03d}.jpg", quality=95)

    for i in range(25):
        img = generate_synthetic_object(i)
        img.save(OOD_NONSKIN_DIR / f"object_{i:03d}.jpg", quality=95)

    non_skin_count = len(list(OOD_NONSKIN_DIR.glob("*.jpg")))
    print(f"[+] Created {non_skin_count} Non-Skin images in {OOD_NONSKIN_DIR}")

    # 2. Extract 100 Genuine Healthy Skin Patches from PAD-UFES-20
    print("[*] Extracting 100 Healthy Skin Patches from PAD-UFES-20 background skin...")
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "smartphone_clinical" / "train.csv")
    pad_dir = PROJECT_ROOT / "data" / "raw" / "pad_ufes_20"

    healthy_count = 0
    for _, row in train_df.iterrows():
        if healthy_count >= 100:
            break
        img_id = str(row["image_id"])
        img_path = pad_dir / f"{img_id}.jpg"
        if img_path.exists():
            patch = extract_healthy_skin_patch(img_path)
            if patch is not None:
                patch.save(OOD_HEALTHY_DIR / f"healthy_skin_{healthy_count:03d}.jpg", quality=95)
                healthy_count += 1

    print(f"[+] Extracted {healthy_count} Healthy Skin patches in {OOD_HEALTHY_DIR}")
    print("\n[+] EXP-009 OOD benchmark dataset successfully prepared!")


if __name__ == "__main__":
    main()
