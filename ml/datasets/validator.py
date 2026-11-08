"""Image quality and integrity validation engine for SkinAI.

Validates:
1. File magic bytes (JPEG, PNG, WebP)
2. Pillow decode integrity and decompression bomb safeguards
3. Minimum spatial dimensions (width, height >= min_dimension)
4. Laplacian blur variance against threshold
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

# Supported magic bytes signatures
MAGIC_SIGNATURES = {
    "jpeg": b"\xff\xd8\xff",
    "png": b"\x89PNG\r\n\x1a\n",
    "webp": b"RIFF",  # WEBP follows at offset 8
}


class QualityMetrics(BaseModel):
    """Quality and validation assessment result for a single image."""

    is_valid: bool = Field(..., description="Whether the image passed all validation checks")
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Reason for rejection (e.g. 'below_min_dimension', 'excessive_blur', 'corrupt_file')"
    )
    width: int = Field(default=0, description="Image width in pixels")
    height: int = Field(default=0, description="Image height in pixels")
    channels: int = Field(default=0, description="Number of color channels")
    blur_score: float = Field(default=0.0, description="Laplacian variance blur score (higher = sharper)")
    magic_format: Optional[str] = Field(default=None, description="Detected image container format")


class ImageValidator:
    """Validates raw image bytes or file paths for inference and dataset ingestion."""

    def __init__(
        self,
        min_dimension: int = 128,
        min_blur_score: float = 60.0,
        max_bytes: int = 15 * 1024 * 1024,  # 15 MB
    ):
        self.min_dimension = min_dimension
        self.min_blur_score = min_blur_score
        self.max_bytes = max_bytes

    def check_magic_bytes(self, data: bytes) -> Optional[str]:
        """Verify image header bytes to protect against file extension spoofing."""
        if len(data) < 12:
            return None
        if data.startswith(MAGIC_SIGNATURES["jpeg"]):
            return "jpeg"
        if data.startswith(MAGIC_SIGNATURES["png"]):
            return "png"
        if data.startswith(MAGIC_SIGNATURES["webp"]) and data[8:12] == b"WEBP":
            return "webp"
        return None

    @staticmethod
    def compute_blur_score(pil_img: Image.Image) -> float:
        """Compute Laplacian variance on grayscale image using vectorized NumPy."""
        gray = pil_img.convert("L")
        arr = np.array(gray, dtype=np.float64)
        if arr.shape[0] < 3 or arr.shape[1] < 3:
            return 0.0
        # Discrete 2D Laplace operator: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        laplacian = (
            arr[:-2, 1:-1]
            + arr[2:, 1:-1]
            + arr[1:-1, :-2]
            + arr[1:-1, 2:]
            - 4.0 * arr[1:-1, 1:-1]
        )
        return float(laplacian.var())

    def validate_bytes(self, raw_bytes: bytes) -> Tuple[QualityMetrics, Optional[Image.Image]]:
        """Validate raw image payload in memory.

        Returns:
            Tuple of (QualityMetrics, loaded PIL Image or None)
        """
        # 1. Payload size check
        if len(raw_bytes) > self.max_bytes:
            return (
                QualityMetrics(
                    is_valid=False,
                    rejection_reason=f"payload_exceeds_max_size ({len(raw_bytes)} > {self.max_bytes})",
                ),
                None,
            )

        # 2. Magic bytes check
        fmt = self.check_magic_bytes(raw_bytes)
        if not fmt:
            return (
                QualityMetrics(
                    is_valid=False,
                    rejection_reason="invalid_magic_bytes",
                ),
                None,
            )

        # 3. Pillow decode check
        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                # Apply EXIF rotation if present
                img_oriented = ImageOps.exif_transpose(img)
                # Convert to RGB to ensure 3 channels
                rgb_img = img_oriented.convert("RGB")
                width, height = rgb_img.size
        except Exception as e:
            return (
                QualityMetrics(
                    is_valid=False,
                    rejection_reason=f"decode_error: {str(e)}",
                    magic_format=fmt,
                ),
                None,
            )

        # 4. Dimension checks
        if width < self.min_dimension or height < self.min_dimension:
            return (
                QualityMetrics(
                    is_valid=False,
                    rejection_reason=f"dimension_too_small ({width}x{height} < {self.min_dimension}x{self.min_dimension})",
                    width=width,
                    height=height,
                    channels=3,
                    magic_format=fmt,
                ),
                rgb_img,
            )

        # 5. Blur check
        blur_score = self.compute_blur_score(rgb_img)
        if blur_score < self.min_blur_score:
            return (
                QualityMetrics(
                    is_valid=False,
                    rejection_reason=f"excessive_blur (score {blur_score:.1f} < threshold {self.min_blur_score:.1f})",
                    width=width,
                    height=height,
                    channels=3,
                    blur_score=blur_score,
                    magic_format=fmt,
                ),
                rgb_img,
            )

        return (
            QualityMetrics(
                is_valid=True,
                width=width,
                height=height,
                channels=3,
                blur_score=blur_score,
                magic_format=fmt,
            ),
            rgb_img,
        )

    def validate_file(self, path: Union[str, Path]) -> Tuple[QualityMetrics, Optional[Image.Image]]:
        """Validate image file on disk."""
        p = Path(path)
        if not p.exists() or not p.is_file():
            return (
                QualityMetrics(
                    is_valid=False,
                    rejection_reason=f"file_not_found: {path}",
                ),
                None,
            )

        try:
            raw_bytes = p.read_bytes()
        except Exception as e:
            return (
                QualityMetrics(
                    is_valid=False,
                    rejection_reason=f"io_read_error: {str(e)}",
                ),
                None,
            )

        return self.validate_bytes(raw_bytes)
