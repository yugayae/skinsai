"""Deterministic Explainability and Case-Based Reasoning Engine for SkinAI.

Provides three deterministic capabilities for clinical interpretability:
1. Quantitative ABCDE Computer Vision Morphometry (Asymmetry, Border, Color CIE L*a*b*, Geometry)
2. Biopsy Atlas Visual k-NN Retrieval (Cosine similarity search over biopsy-verified clinical cases)
3. Pixel-Level DINOv2 Self-Attention Map Extraction & Lesion Concentration Verification
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw


@dataclass
class ABCDEMetrics:
    """Quantitative morphological biomarkers corresponding to clinical ABCDE criteria."""

    asymmetry_shape: float          # [0, 1] Bilateral shape IoU asymmetry along principal axis
    asymmetry_color: float          # [0, 1] Bilateral CIE LAB chromatic asymmetry
    border_irregularity: float      # Isoperimetric ratio P^2 / (4*pi*Area), normalized >= 1.0
    border_radial_std: float        # Normalized radial distance standard deviation of contour
    color_std_l: float              # CIE L* (lightness) standard deviation inside lesion
    color_std_a: float              # CIE a* (red-green erythema) standard deviation inside lesion
    color_std_b: float              # CIE b* (yellow-blue melanin) standard deviation inside lesion
    color_entropy: float            # Shannon entropy of chromatic distribution inside lesion
    distinct_color_clusters: int    # Count of active dermatological color zones (1 to 6)
    lesion_area_ratio: float        # Fraction of image area occupied by segmented lesion
    delta_l_contrast: float         # Luminance contrast between lesion core and surrounding skin
    delta_a_erythema: float         # Erythema (redness) shift relative to surrounding skin

    def to_vector(self) -> np.ndarray:
        """Return standardized 12-dimensional float32 feature vector for ML fusion."""
        return np.array(
            [
                self.asymmetry_shape,
                self.asymmetry_color,
                np.log1p(max(0.0, self.border_irregularity - 1.0)),
                self.border_radial_std,
                self.color_std_l / 25.0,
                self.color_std_a / 15.0,
                self.color_std_b / 15.0,
                self.color_entropy / 4.0,
                float(self.distinct_color_clusters) / 6.0,
                self.lesion_area_ratio,
                self.delta_l_contrast / 40.0,
                self.delta_a_erythema / 20.0,
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert biomarkers dataclass to dictionary with rounded floats.

        Returns:
            Dictionary mapping biomarker names to float/int values.
        """
        return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)) for k, v in asdict(self).items()}


@dataclass
class RetrievedBiopsyCase:
    """Metadata of a histologically verified twin case from the clinical reference atlas."""

    rank: int
    image_id: str
    patient_id: str
    cosine_similarity: float
    subtype_id: str
    malignancy_label: str
    age_approx: Optional[float]
    sex: str
    anatom_site: str
    dataset_source: str


class ABCDEMorphometryEngine:
    """Deterministic computer vision morphometry engine computing ABCDE lesion biomarkers."""

    @staticmethod
    def rgb_to_cielab(rgb_uint8: np.ndarray) -> np.ndarray:
        """Convert uint8 RGB image (H, W, 3) to perceptual CIE L*a*b* float32 array."""
        rgb = rgb_uint8.astype(np.float32) / 255.0
        # sRGB to linear RGB
        mask = rgb > 0.04045
        rgb_lin = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

        # Linear RGB to CIE XYZ (D65)
        r, g, b = rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]
        x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
        y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) / 1.00000
        z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883

        eps = 0.008856
        kappa = 903.3

        def f(t: np.ndarray) -> np.ndarray:
            return np.where(t > eps, np.cbrt(np.maximum(t, 1e-8)), (kappa * t + 16.0) / 116.0)

        fx, fy, fz = f(x), f(y), f(z)
        l_star = 116.0 * fy - 16.0
        a_star = 500.0 * (fx - fy)
        b_star = 200.0 * (fy - fz)
        return np.stack([l_star, a_star, b_star], axis=-1).astype(np.float32)

    @staticmethod
    def remove_hairs_blackhat(rgb_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove dark hair strands using morphological Black-Hat filtering and Telea inpainting (DullRazor-style).

        Returns:
            clean_rgb: Hair-inpainted RGB uint8 image (H, W, 3)
            hair_mask: Binary uint8 mask of detected hair pixels (H, W)
        """
        gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        # Dynamic threshold combining absolute contrast (>16) and 92nd percentile
        thresh_val = max(16.0, float(np.percentile(blackhat, 92)))
        _, hair_mask = cv2.threshold(blackhat, thresh_val, 255, cv2.THRESH_BINARY)

        # Dilate slightly to cover hair anti-aliasing fringe
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hair_mask = cv2.dilate(hair_mask, dil_kernel, iterations=1)

        # Guard against inpainting the entire lesion if it is very small & dark
        hair_frac = float(np.mean(hair_mask > 0))
        if hair_frac < 0.001 or hair_frac > 0.22:
            return rgb_uint8, np.zeros_like(gray, dtype=np.uint8)

        bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        clean_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        return clean_rgb, hair_mask

    @classmethod
    def segment_lesion_legacy_otsu(cls, lab: np.ndarray) -> np.ndarray:
        """Legacy center-prior Otsu segmentation (kept as baseline for DS-5 benchmark)."""
        h, w, _ = lab.shape
        margin_y = max(4, int(h * 0.12))
        margin_x = max(4, int(w * 0.12))
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[:margin_y, :] = True
        border_mask[-margin_y:, :] = True
        border_mask[:, :margin_x] = True
        border_mask[:, -margin_x:] = True

        bg_lab = np.median(lab[border_mask], axis=0)
        delta_l = np.maximum(0.0, bg_lab[0] - lab[..., 0])
        delta_a = np.abs(lab[..., 1] - bg_lab[1])
        delta_b = np.abs(lab[..., 2] - bg_lab[2])
        saliency = 1.2 * delta_l + 1.0 * delta_a + 0.8 * delta_b

        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h / 2.0, w / 2.0
        dist_sq = ((yy - cy) / (0.45 * h)) ** 2 + ((xx - cx) / (0.45 * w)) ** 2
        center_prior = np.exp(-0.5 * dist_sq)
        weighted_saliency = saliency * (0.35 + 0.65 * center_prior)

        s_min, s_max = float(np.percentile(weighted_saliency, 2)), float(np.percentile(weighted_saliency, 98))
        if s_max - s_min < 1e-3:
            return dist_sq <= 0.4

        hist, bin_edges = np.histogram(weighted_saliency.ravel(), bins=64, range=(s_min, s_max))
        hist = hist.astype(np.float64) / max(weighted_saliency.size, 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        w0 = np.cumsum(hist)
        w1 = 1.0 - w0
        mu = np.cumsum(hist * bin_centers)
        mu_t = mu[-1]

        valid = (w0 > 0.05) & (w1 > 0.05)
        sigma_b_sq = np.zeros_like(w0)
        sigma_b_sq[valid] = ((mu_t * w0[valid] - mu[valid]) ** 2) / (w0[valid] * w1[valid])
        best_idx = int(np.argmax(sigma_b_sq))
        thresh = float(bin_centers[best_idx])

        mask = weighted_saliency >= thresh
        area_frac = float(np.mean(mask))
        if area_frac < 0.06:
            mask = weighted_saliency >= np.percentile(weighted_saliency, 80)
        elif area_frac > 0.65:
            mask = weighted_saliency >= np.percentile(weighted_saliency, 55)
        return mask

    @classmethod
    def segment_lesion(
        cls,
        lab: np.ndarray,
        rgb_uint8: Optional[np.ndarray] = None,
        attn_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """DS-5 Attention-Seeded GrabCut segmentation with CIE LAB contrast & connected-component cleanup.

        Unlike center-locked Otsu or fixed bounding-box GrabCut, this method seeds foreground/background
        trimaps dynamically from the DINOv2 [CLS] self-attention map (when available) fused with
        perceptual CIE LAB skin contrast, enabling accurate segmentation of off-center lesions.
        """
        h, w, _ = lab.shape
        margin_y = max(4, int(h * 0.08))
        margin_x = max(4, int(w * 0.08))
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[:margin_y, :] = True
        border_mask[-margin_y:, :] = True
        border_mask[:, :margin_x] = True
        border_mask[:, -margin_x:] = True

        # If attention map is provided, estimate background skin from low-attention outer pixels
        if attn_map is not None and attn_map.shape == (h, w):
            attn_norm = (attn_map - float(np.min(attn_map))) / max(float(np.max(attn_map) - np.min(attn_map)), 1e-6)
            bg_candidates = border_mask | (attn_norm < np.percentile(attn_norm, 30))
            bg_lab = np.median(lab[bg_candidates], axis=0) if np.any(bg_candidates) else np.median(lab[border_mask], axis=0)
        else:
            attn_norm = None
            bg_lab = np.median(lab[border_mask], axis=0)

        delta_l = np.maximum(0.0, bg_lab[0] - lab[..., 0])
        delta_a = np.abs(lab[..., 1] - bg_lab[1])
        delta_b = np.abs(lab[..., 2] - bg_lab[2])
        saliency = 1.2 * delta_l + 1.0 * delta_a + 0.8 * delta_b

        # Fuse chromatic saliency with DINOv2 attention (replacing hardcoded center (h/2, w/2) Gaussian lock!)
        yy, xx = np.mgrid[0:h, 0:w]
        if attn_norm is not None:
            attn_smooth = cv2.GaussianBlur(attn_norm.astype(np.float32), (15, 15), 0)
            attn_smooth = (attn_smooth - float(np.min(attn_smooth))) / max(
                float(np.max(attn_smooth) - np.min(attn_smooth)), 1e-6
            )
            # Compute dynamic lesion centroid (cy_attn, cx_attn) from top 15% of DINOv2 self-attention
            top_attn_mask = attn_smooth >= np.percentile(attn_smooth, 85)
            if np.any(top_attn_mask):
                w_top = attn_smooth[top_attn_mask]
                cy_attn = float(np.sum(yy[top_attn_mask] * w_top) / np.sum(w_top))
                cx_attn = float(np.sum(xx[top_attn_mask] * w_top) / np.sum(w_top))
            else:
                cy_attn, cx_attn = h / 2.0, w / 2.0

            dist_sq = ((yy - cy_attn) / (0.45 * h)) ** 2 + ((xx - cx_attn) / (0.45 * w)) ** 2
            attn_gaussian_prior = np.exp(-0.5 * dist_sq)
            spatial_prior = 0.65 * attn_gaussian_prior + 0.35 * attn_smooth
            weighted_saliency = saliency * (0.35 + 0.65 * spatial_prior)
        else:
            cy, cx = h / 2.0, w / 2.0
            dist_sq = ((yy - cy) / (0.45 * h)) ** 2 + ((xx - cx) / (0.45 * w)) ** 2
            center_prior = np.exp(-0.5 * dist_sq)
            weighted_saliency = saliency * (0.35 + 0.65 * center_prior)

        # Compute adaptive Otsu threshold on Attention-Anchored Saliency
        s_min, s_max = float(np.percentile(weighted_saliency, 2)), float(np.percentile(weighted_saliency, 98))
        if s_max - s_min < 1e-3:
            return weighted_saliency >= np.median(weighted_saliency)

        hist, bin_edges = np.histogram(weighted_saliency.ravel(), bins=64, range=(s_min, s_max))
        hist = hist.astype(np.float64) / max(weighted_saliency.size, 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        w0 = np.cumsum(hist)
        w1 = 1.0 - w0
        mu = np.cumsum(hist * bin_centers)
        mu_t = mu[-1]

        valid = (w0 > 0.05) & (w1 > 0.05)
        sigma_b_sq = np.zeros_like(w0)
        sigma_b_sq[valid] = ((mu_t * w0[valid] - mu[valid]) ** 2) / (w0[valid] * w1[valid])
        best_idx = int(np.argmax(sigma_b_sq))
        thresh = float(bin_centers[best_idx])

        seed_mask = weighted_saliency >= thresh
        area_frac = float(np.mean(seed_mask))
        if area_frac < 0.06:
            seed_mask = weighted_saliency >= np.percentile(weighted_saliency, 80)
        elif area_frac > 0.65:
            seed_mask = weighted_saliency >= np.percentile(weighted_saliency, 55)

        # Deterministic GrabCut refinement initialized from Attention-Anchored Otsu Trimap
        seed_u8 = (seed_mask.astype(np.uint8)) * 255
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_core = cv2.erode(seed_u8, k_erode, iterations=1) > 0
        pr_band = cv2.dilate(seed_u8, k_dilate, iterations=1) > 0

        if rgb_uint8 is not None and np.sum(fg_core) >= 30 and np.sum(~pr_band) >= 100:
            gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
            gc_mask[pr_band] = cv2.GC_PR_BGD
            gc_mask[seed_mask] = cv2.GC_PR_FGD
            gc_mask[fg_core] = cv2.GC_FGD
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            try:
                cv2.setRNGSeed(42)
                cv2.grabCut(rgb_uint8, gc_mask, None, bgd_model, fgd_model, iterCount=1, mode=cv2.GC_INIT_WITH_MASK)
                gc_out = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
                # Blend GrabCut contour with Attention-Anchored Otsu seed (prevents GMM over-expansion on shifted borders)
                gc_iou = float(np.sum(gc_out & seed_mask) / max(np.sum(gc_out | seed_mask), 1))
                mask = (gc_out & pr_band) if gc_iou >= 0.70 else seed_mask
            except Exception:
                mask = seed_mask
        else:
            mask = seed_mask

        # Morphological closing to smooth hair/contour micro-notches
        mask_u8 = (mask.astype(np.uint8)) * 255
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_k) > 0

        return mask

    @classmethod
    def compute_metrics(
        cls,
        pil_img: Image.Image,
        target_size: int = 224,
        attn_map: Optional[np.ndarray] = None,
        method: str = "attention_grabcut",
    ) -> Tuple[ABCDEMetrics, np.ndarray]:
        """Compute full ABCDE morphometry biomarkers and return (ABCDEMetrics, binary_mask)."""
        resized = pil_img.convert("RGB").resize((target_size, target_size), Image.Resampling.BILINEAR)
        raw_rgb = np.array(resized, dtype=np.uint8)
        if method == "legacy_otsu":
            rgb = raw_rgb
            lab = cls.rgb_to_cielab(rgb)
            mask = cls.segment_lesion_legacy_otsu(lab)
        else:
            rgb, _ = cls.remove_hairs_blackhat(raw_rgb)
            lab = cls.rgb_to_cielab(rgb)
            mask = cls.segment_lesion(lab, rgb_uint8=rgb, attn_map=attn_map)

        h, w = mask.shape
        area = float(np.sum(mask))
        if area < 20:
            yy, xx = np.mgrid[0:h, 0:w]
            mask = ((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2) <= (0.25 * h) ** 2
            area = float(np.sum(mask))

        area_ratio = area / float(h * w)

        # 1. Principal Axis Alignment & Asymmetry (A)
        ys, xs = np.nonzero(mask)
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        dy = ys - cy
        dx = xs - cx

        # Rotate lesion coordinates into principal inertia frame (PCA)
        cov_xx = float(np.mean(dx * dx)) + 1e-6
        cov_yy = float(np.mean(dy * dy)) + 1e-6
        cov_xy = float(np.mean(dx * dy))
        theta = 0.5 * np.arctan2(2.0 * cov_xy, cov_xx - cov_yy)

        grid_y, grid_x = np.mgrid[0:h, 0:w]
        gy = grid_y - cy
        gx = grid_x - cx

        # Exact reflection across principal axis line normal n = (-sin(theta), cos(theta)):
        nx, ny = -np.sin(theta), np.cos(theta)
        dot_n = gx * nx + gy * ny
        refl_gx_major = gx - 2.0 * dot_n * nx
        refl_gy_major = gy - 2.0 * dot_n * ny
        ix_maj = np.clip(np.round(cx + refl_gx_major).astype(int), 0, w - 1)
        iy_maj = np.clip(np.round(cy + refl_gy_major).astype(int), 0, h - 1)
        mask_refl_maj = mask[iy_maj, ix_maj]

        # Reflection across minor axis line normal m = (cos(theta), sin(theta))
        mx, my = np.cos(theta), np.sin(theta)
        dot_m = gx * mx + gy * my
        refl_gx_minor = gx - 2.0 * dot_m * mx
        refl_gy_minor = gy - 2.0 * dot_m * my
        ix_min = np.clip(np.round(cx + refl_gx_minor).astype(int), 0, w - 1)
        iy_min = np.clip(np.round(cy + refl_gy_minor).astype(int), 0, h - 1)
        mask_refl_min = mask[iy_min, ix_min]

        asym_maj = float(np.sum(mask ^ mask_refl_maj)) / max(float(np.sum(mask | mask_refl_maj)), 1.0)
        asym_min = float(np.sum(mask ^ mask_refl_min)) / max(float(np.sum(mask | mask_refl_min)), 1.0)
        asymmetry_shape = 0.5 * (asym_maj + asym_min)

        # Chromatic Asymmetry across major & minor principal halves
        half1_lab = lab[mask & (dot_n >= 0)]
        half2_lab = lab[mask & (dot_n < 0)]
        half3_lab = lab[mask & (dot_m >= 0)]
        half4_lab = lab[mask & (dot_m < 0)]
        if len(half1_lab) > 5 and len(half2_lab) > 5 and len(half3_lab) > 5 and len(half4_lab) > 5:
            diff_maj = float(np.linalg.norm(np.mean(half1_lab, axis=0) - np.mean(half2_lab, axis=0)))
            diff_min = float(np.linalg.norm(np.mean(half3_lab, axis=0) - np.mean(half4_lab, axis=0)))
            asymmetry_color = min(1.0, (0.5 * (diff_maj + diff_min)) / 25.0)
        else:
            asymmetry_color = 0.0

        # 2. Border Irregularity & Radial Contour Variance (B)
        # Morphological boundary: mask pixels with at least one 4-neighbor false
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        boundary = mask & (~eroded)
        perimeter = float(np.sum(boundary))
        border_irregularity = max(1.0, (perimeter ** 2) / max(4.0 * np.pi * area, 1.0))

        bys, bxs = np.nonzero(boundary)
        if len(bys) > 8:
            radii = np.sqrt((bys - cy) ** 2 + (bxs - cx) ** 2)
            border_radial_std = float(np.std(radii) / max(np.mean(radii), 1.0))
        else:
            border_radial_std = 0.0

        # 3. Color Variegation & CIE LAB Chromatic Entropy (C)
        lesion_lab = lab[mask]
        bg_lab = lab[~mask] if np.sum(~mask) > 20 else lab.reshape(-1, 3)

        color_std_l = float(np.std(lesion_lab[:, 0]))
        color_std_a = float(np.std(lesion_lab[:, 1]))
        color_std_b = float(np.std(lesion_lab[:, 2]))

        delta_l_contrast = float(np.mean(bg_lab[:, 0]) - np.mean(lesion_lab[:, 0]))
        delta_a_erythema = float(np.mean(lesion_lab[:, 1]) - np.mean(bg_lab[:, 1]))

        # Joint chromatic histogram entropy (16 bins on L*, 16 bins on a*)
        h_la, _, _ = np.histogram2d(
            lesion_lab[:, 0],
            lesion_lab[:, 1],
            bins=12,
            range=[[0, 100], [-30, 60]],
        )
        p_la = h_la.ravel() / max(float(np.sum(h_la)), 1.0)
        p_la = p_la[p_la > 1e-8]
        color_entropy = float(-np.sum(p_la * np.log2(p_la)))

        # Count dermatoscopic color zones present (>= 3% of lesion pixels each):
        # 1. Black (L < 28), 2. Dark Brown (28 <= L < 48, b > 8), 3. Light Brown (48 <= L < 68, b > 10),
        # 4. Red/Pink Erythema (a > 18), 5. Blue-Gray Veil (L < 60, b < 2), 6. White/Regression (L >= 75, a < 12)
        L_v, a_v, b_v = lesion_lab[:, 0], lesion_lab[:, 1], lesion_lab[:, 2]
        n_pix = max(len(L_v), 1)
        zones = [
            np.sum(L_v < 28) / n_pix >= 0.03,
            np.sum((L_v >= 28) & (L_v < 48) & (b_v > 6)) / n_pix >= 0.03,
            np.sum((L_v >= 48) & (L_v < 68) & (b_v > 8)) / n_pix >= 0.03,
            np.sum(a_v > 18) / n_pix >= 0.03,
            np.sum((L_v < 60) & (b_v < 3)) / n_pix >= 0.03,
            np.sum((L_v >= 72) & (a_v < 12)) / n_pix >= 0.03,
        ]
        distinct_clusters = max(1, int(sum(zones)))

        return (
            ABCDEMetrics(
                asymmetry_shape=asymmetry_shape,
                asymmetry_color=asymmetry_color,
                border_irregularity=border_irregularity,
                border_radial_std=border_radial_std,
                color_std_l=color_std_l,
                color_std_a=color_std_a,
                color_std_b=color_std_b,
                color_entropy=color_entropy,
                distinct_color_clusters=distinct_clusters,
                lesion_area_ratio=area_ratio,
                delta_l_contrast=delta_l_contrast,
                delta_a_erythema=delta_a_erythema,
            ),
            mask,
        )


class BiopsyAtlasRetrievalEngine:
    """Visual Case-Based Reasoning (k-NN Retrieval) over histologically verified lesions."""

    def __init__(
        self,
        atlas_embeddings: torch.Tensor,
        atlas_df: pd.DataFrame,
        class_order: List[str],
        malignant_classes: set,
        temperature: float = 0.07,
    ):
        self.atlas_embeddings = F.normalize(atlas_embeddings.float(), dim=-1)
        self.atlas_df = atlas_df.reset_index(drop=True)
        self.class_order = class_order
        self.class_to_idx = {c: i for i, c in enumerate(class_order)}
        self.malignant_classes = malignant_classes
        self.temperature = temperature

        # Filter index to samples with verified 6-class subtype
        valid_mask = self.atlas_df["subtype_target"].isin(self.class_to_idx.keys()).values
        self.valid_indices = np.where(valid_mask)[0]
        self.valid_embeddings = self.atlas_embeddings[self.valid_indices]
        self.valid_df = self.atlas_df.iloc[self.valid_indices].reset_index(drop=True)
        self.valid_targets = np.array([self.class_to_idx[s] for s in self.valid_df["subtype_target"]])

    def retrieve_top_k(
        self,
        query_embedding: torch.Tensor,
        k: int = 5,
        exclude_patient_id: Optional[str] = None,
    ) -> Tuple[List[RetrievedBiopsyCase], np.ndarray]:
        """Retrieve Top-K biopsy-verified twins and compute similarity-weighted class probabilities."""
        q = F.normalize(query_embedding.float().view(1, -1), dim=-1)
        sims = torch.mm(q, self.valid_embeddings.t()).squeeze(0).cpu().numpy()

        if exclude_patient_id is not None:
            pat_mask = self.valid_df["patient_id"].astype(str).values == str(exclude_patient_id)
            sims[pat_mask] = -1.0

        top_idx = np.argsort(-sims)[:k]
        cases: List[RetrievedBiopsyCase] = []
        for rank, idx in enumerate(top_idx, start=1):
            row = self.valid_df.iloc[idx]
            sub = str(row["subtype_target"])
            mal_lbl = "Malignant" if sub in self.malignant_classes else "Benign"
            age_v = row.get("age_approx")
            try:
                age_f = float(age_v) if pd.notna(age_v) and age_v != "" else None
            except Exception:
                age_f = None
            cases.append(
                RetrievedBiopsyCase(
                    rank=rank,
                    image_id=str(row.get("image_id", f"case_{idx}")),
                    patient_id=str(row.get("patient_id", "unknown")),
                    cosine_similarity=round(float(sims[idx]), 4),
                    subtype_id=sub,
                    malignancy_label=mal_lbl,
                    age_approx=age_f,
                    sex=str(row.get("sex", "unknown")),
                    anatom_site=str(row.get("anatom_site_1", "unknown")),
                    dataset_source=str(row.get("dataset_source", "PAD-UFES-20")),
                )
            )

        # Compute soft k-NN probability distribution over all 6 classes using top-15 neighbors
        k_prob = min(15, len(sims))
        prob_idx = np.argsort(-sims)[:k_prob]
        top_sims = sims[prob_idx]
        weights = np.exp((top_sims - np.max(top_sims)) / max(self.temperature, 1e-3))
        weights /= np.sum(weights)

        knn_probs = np.zeros(len(self.class_order), dtype=np.float32)
        for w, idx in zip(weights, prob_idx):
            knn_probs[self.valid_targets[idx]] += float(w)

        return cases, knn_probs


class DINOv2AttentionExtractor:
    """Extracts pixel-level self-attention maps from the final ViT block of Meta DINOv2 Small."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._last_qkv: Optional[torch.Tensor] = None
        self._hook_handle = self.model.blocks[-1].attn.qkv.register_forward_hook(self._qkv_hook)

    def _qkv_hook(self, module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
        self._last_qkv = output.detach()

    def close(self) -> None:
        """Remove the forward hook on the attention module."""
        if self._hook_handle is not None:
            self._hook_handle.remove()

    @torch.no_grad()
    def extract_attention_map(self, img_tensor_518: torch.Tensor, target_size: int = 224) -> np.ndarray:
        """Compute [CLS]-to-patch self-attention map upsampled to (target_size, target_size) in [0, 1]."""
        if img_tensor_518.dim() == 3:
            img_tensor_518 = img_tensor_518.unsqueeze(0)
        _ = self.model(img_tensor_518.to(self.device))

        qkv = self._last_qkv  # shape (1, N_tokens, 3 * dim)
        if qkv is None:
            return np.ones((target_size, target_size), dtype=np.float32)

        b, n_tokens, three_d = qkv.shape
        num_heads = self.model.blocks[-1].attn.num_heads
        head_dim = (three_d // 3) // num_heads

        qkv_reshaped = qkv.reshape(b, n_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv_reshaped[0], qkv_reshaped[1]  # (1, num_heads, n_tokens, head_dim)

        scale = head_dim ** -0.5
        attn = torch.matmul(q[:, :, 0:1, :], k.transpose(-2, -1)) * scale  # [CLS] attention to all tokens
        attn = F.softmax(attn, dim=-1)  # (1, num_heads, 1, n_tokens)

        # Spatial patches are at the end (37 x 37 = 1369 tokens for 518x518 patch14)
        grid_size = 518 // 14  # 37
        num_patches = grid_size * grid_size
        patch_attn = attn[0, :, 0, -num_patches:]  # (num_heads, 1369)
        mean_attn = patch_attn.mean(dim=0).reshape(1, 1, grid_size, grid_size)

        upsampled = F.interpolate(mean_attn, size=(target_size, target_size), mode="bicubic", align_corners=False)
        amap = upsampled.squeeze().cpu().numpy()
        amap = (amap - amap.min()) / max(float(amap.max() - amap.min()), 1e-6)
        return amap.astype(np.float32)

    @staticmethod
    def render_explainability_card(
        pil_img: Image.Image,
        mask: np.ndarray,
        attn_map: np.ndarray,
        abcde: ABCDEMetrics,
        out_path: Path,
        title_text: str = "",
    ) -> None:
        """Render a 3-panel visual explainability card (Original+Contour | Attention Heatmap | Mask+Morphometry)."""
        sz = 224
        base_rgb = np.array(pil_img.convert("RGB").resize((sz, sz), Image.Resampling.BILINEAR), dtype=np.uint8)

        # Panel 1: Original + Green Lesion Contour
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        boundary = mask & (~eroded)
        panel1 = base_rgb.copy()
        panel1[boundary] = [0, 255, 128]

        # Panel 2: DINOv2 Self-Attention Heatmap Overlay (Jet-like colormap)
        v = np.clip(attn_map, 0.0, 1.0)
        heat_r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
        heat_g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
        heat_b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
        heatmap_rgb = (np.stack([heat_r, heat_g, heat_b], axis=-1) * 255.0).astype(np.uint8)
        panel2 = (0.45 * base_rgb + 0.55 * heatmap_rgb).astype(np.uint8)

        # Panel 3: Segmented Lesion Region + Principal Axis
        panel3 = (base_rgb * 0.25).astype(np.uint8)
        panel3[mask] = base_rgb[mask]
        panel3[boundary] = [255, 80, 80]

        canvas = Image.new("RGB", (sz * 3 + 20, sz + 52), color=(18, 22, 28))
        canvas.paste(Image.fromarray(panel1), (5, 28))
        canvas.paste(Image.fromarray(panel2), (sz + 10, 28))
        canvas.paste(Image.fromarray(panel3), (sz * 2 + 15, 28))

        draw = ImageDraw.Draw(canvas)
        header = (
            f"{title_text} | A(shape)={abcde.asymmetry_shape:.2f} | B(irreg)={abcde.border_irregularity:.2f} | "
            f"C(entropy)={abcde.color_entropy:.2f}, zones={abcde.distinct_color_clusters}"
        )
        draw.text((10, 6), header, fill=(235, 240, 245))
        draw.text((10, sz + 32), "[1] Contour & Margin", fill=(160, 220, 180))
        draw.text((sz + 15, sz + 32), "[2] DINOv2 Self-Attention Map", fill=(255, 200, 120))
        draw.text((sz * 2 + 20, sz + 32), "[3] ABCDE Segmented ROI", fill=(240, 160, 160))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)

