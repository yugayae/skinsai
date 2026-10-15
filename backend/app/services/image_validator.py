"""HTTP payload image security and format validator."""

from __future__ import annotations

import io
from typing import Optional, Tuple
from PIL import Image

from backend.app.core.config import settings

# Enforce strict decompression bomb protection
Image.MAX_IMAGE_PIXELS = 25_000_000

# Magic byte signatures
JPEG_MAGIC = b"\xFF\xD8\xFF"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
WEBP_RIFF = b"RIFF"
WEBP_FORMAT = b"WEBP"


def detect_image_format_from_bytes(data: bytes) -> Optional[str]:
    """Detect image format by inspecting magic header bytes."""
    if len(data) < 12:
        return None
    if data.startswith(JPEG_MAGIC):
        return "JPEG"
    if data.startswith(PNG_MAGIC):
        return "PNG"
    if data.startswith(WEBP_RIFF) and data[8:12] == WEBP_FORMAT:
        return "WEBP"
    return None


def validate_image_upload(raw_bytes: bytes) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """Validate image bytes for safety, size bounds, magic headers, and decodeability.
    
    Returns:
        (is_valid, error_message, pil_image_or_none)
    """
    # 1. Byte length check
    if len(raw_bytes) == 0:
        return False, "Empty file uploaded.", None
    if len(raw_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        mb_size = len(raw_bytes) / (1024 * 1024)
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        return False, f"File size ({mb_size:.1f} MB) exceeds maximum limit of {max_mb:.0f} MB.", None

    # 2. Magic byte validation
    fmt = detect_image_format_from_bytes(raw_bytes)
    if fmt is None:
        return False, "Unsupported or corrupt image format. Only JPEG, PNG, and WebP are allowed.", None

    # 3. Pillow decompression and sanity check
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.verify()  # verify integrity of byte stream
        
        # Re-open after verify (Pillow verify closes file pointer state)
        pil_img = Image.open(io.BytesIO(raw_bytes))
        width, height = pil_img.size

        if width < settings.MIN_IMAGE_DIMENSION or height < settings.MIN_IMAGE_DIMENSION:
            return False, f"Image resolution ({width}x{height}) is below minimum {settings.MIN_IMAGE_DIMENSION}x{settings.MIN_IMAGE_DIMENSION}.", None

        if width > settings.MAX_IMAGE_DIMENSION or height > settings.MAX_IMAGE_DIMENSION:
            return False, f"Image resolution ({width}x{height}) exceeds maximum {settings.MAX_IMAGE_DIMENSION}x{settings.MAX_IMAGE_DIMENSION}.", None

        # Convert to RGB mode
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        return True, None, pil_img

    except Image.DecompressionBombError:
        return False, "Decompression bomb detected: image dimensions exceed safe limits.", None
    except Exception as exc:
        return False, f"Failed to parse image: {type(exc).__name__}", None
