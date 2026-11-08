#!/usr/bin/env python3
"""Phase 2.8: Pre-Registered Locked External Evaluation on Stanford DDI.

Standards: CLAIM 2024 / STARD-AI / IEC 62304
Dataset: Diverse Dermatology Images (DDI), Stanford University
- 656 clinical macro/close-up images (570 patients)
- 100% biopsied/histopathologically confirmed
- Stratified by Fitzpatrick skin types: FST 12 (I-II), FST 34 (III-IV), FST 56 (V-VI)

Evaluation Constraints:
- Zero training/tuning: Model weights are frozen and strictly checked against SHA-256.
- Fixed pre-registered thresholds: tau_melanoma = 0.05, tau_triage = 0.20.
- Mandatory Wilson score 95% confidence intervals and exact fractions (n/N).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc,
)
import timm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.multi_head import CLASS_ORDER, MALIGNANT_CLASSES
from ml.models.multimodal import MultimodalLateFusionProbe
from scripts.train_development_pool_v2_model import extract_clinical_vector

# Pre-registered locked model checkpoint and hash
LOCKED_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "smartphone_clinical" / "development_champion_v2" / "best_model.pt"
EXPECTED_SHA256 = "4bc71ff1dbad7520f5c76469185bfe409672565683afe3a256134ce8bd74d58f"

DDI_DIR = PROJECT_ROOT / "data" / "raw" / "ddi"

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_ORDER)}
MALIGNANT_INDICES = [CLASS_TO_IDX[c] for c in MALIGNANT_CLASSES]

# Subtype name mapping for DDI disease strings
DISEASE_TO_SUBTYPE = {
    "melanoma": "mel",
    "basal cell carcinoma": "bcc",
    "squamous cell carcinoma": "scc",
    "melanocytic nevus": "nev",
    "seborrheic keratosis": "sek",
    "actinic keratosis": "ack",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate exact Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95%
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, float(centre - half_width)), min(1.0, float(centre + half_width))


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[Path], transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.paths[idx]) as img:
            rgb = img.convert("RGB")
        if self.transform:
            return self.transform(rgb)
        return transforms.ToTensor()(rgb)


def locate_ddi_dataset() -> Tuple[pd.DataFrame, Path]:
    """Locate DDI metadata and images directory."""
    candidates = [
        DDI_DIR,
        PROJECT_ROOT / "data" / "external" / "ddi",
        PROJECT_ROOT / "data" / "ddi",
        PROJECT_ROOT / "data" / "raw" / "DDI",
    ]

    for c in candidates:
        meta_p = c / "ddi_metadata.csv"
        if meta_p.exists():
            df = pd.read_csv(meta_p)
            return df, c

    print("[-] DDI Dataset not found in any standard path:")
    for c in candidates:
        print(f"    - {c}")
    print("\nPlease download the DDI dataset from Stanford AIMI:")
    print("https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965")
    print(f"and place `ddi_metadata.csv` and images into: {DDI_DIR}")
    sys.exit(1)


def find_image_file(root: Path, file_name: str) -> Path | None:
    """Find image file either in root or in images/ subdirectory."""
    for sub in ["", "images", "DDI", "data"]:
        p = root / sub / file_name if sub else root / file_name
        if p.exists():
            return p
    # Recursive search if not found directly
    found = list(root.rglob(file_name))
    return found[0] if found else None


def main():
    print("=" * 80)
    print(" SkinAI Phase 2.8: Pre-Registered Locked External Evaluation on Stanford DDI")
    print(" Standard: CLAIM 2024 / STARD-AI / IEC 62304")
    print("=" * 80)

    # 1. Model Verification
    if not LOCKED_MODEL_PATH.exists():
        print(f"[-] Error: Champion model not found at {LOCKED_MODEL_PATH}")
        sys.exit(1)

    actual_hash = sha256_file(LOCKED_MODEL_PATH)
    print(f"[*] Verifying Model Checkpoint SHA-256:")
    print(f"    Expected: {EXPECTED_SHA256}")
    print(f"    Actual  : {actual_hash}")

    if actual_hash != EXPECTED_SHA256:
        print("[-] INTEGRITY ERROR: Model weights have been modified or do not match pre-registered hash!")
        sys.exit(1)
    print("[+] Model integrity verified 100%. Locked for blind evaluation.\n")

    # 2. Dataset Ingestion
    df, ddi_root = locate_ddi_dataset()
    print(f"[+] Loaded DDI metadata from {ddi_root / 'ddi_metadata.csv'} ({len(df)} records).")

    # Normalize column names
    file_col = "DDI_file" if "DDI_file" in df.columns else "file_name"
    mal_col = "malignant" if "malignant" in df.columns else "malignancy(malig=1)"
    tone_col = "skin_tone" if "skin_tone" in df.columns else "FST"
    disease_col = "disease" if "disease" in df.columns else "diagnosis"

    image_paths = []
    valid_rows = []
    missing_count = 0

    for idx, row in df.iterrows():
        fname = str(row[file_col]).strip()
        p = find_image_file(ddi_root, fname)
        if p is None:
            missing_count += 1
            continue
        image_paths.append(p)
        valid_rows.append(row)

    if missing_count > 0:
        print(f"[!] Warning: {missing_count} images could not be located on disk.")
    df_eval = pd.DataFrame(valid_rows)
    print(f"[+] Evaluated image count: {len(df_eval)} out of {len(df)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 3. Vision Feature Extraction (Meta DINOv2 Small 384D)
    cache_path = PROJECT_ROOT / "artifacts" / "embeddings" / "ddi_dinov2.pt"
    if cache_path.exists():
        print(f"[*] Loading cached DINOv2 embeddings from {cache_path}...")
        vision_features = torch.load(cache_path, map_location=device, weights_only=True)
    else:
        print(f"[*] Extracting DINOv2 Small features for {len(df_eval)} DDI images...")
        dino_model = timm.create_model("vit_small_patch14_dinov2", pretrained=True, num_classes=0).to(device)
        dino_model.eval()

        transform = transforms.Compose([
            transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = ImagePathDataset(image_paths, transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        embeds = []
        t0 = time.time()
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                feats = dino_model(b)
                if isinstance(feats, (tuple, list)):
                    feats = feats[0]
                feats = feats / feats.norm(dim=-1, keepdim=True)
                embeds.append(feats.cpu())

        vision_features = torch.cat(embeds, dim=0).to(device)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vision_features, cache_path)
        print(f"[+] Extracted and cached embeddings in {time.time()-t0:.1f}s (shape: {vision_features.shape}).")

    # 4. Clinical Tabular Features (standardized unknown clinical vector as in training)
    clinical_features = torch.tensor(
        np.stack([extract_clinical_vector(r) for _, r in df_eval.iterrows()]),
        dtype=torch.float32,
    ).to(device)

    # 5. Model Inference
    probe = MultimodalLateFusionProbe(
        vis_dim=384,
        clin_dim=12,
        clin_embed_dim=64,
        hidden_dim=256,
        dropout=0.25,
        num_subtypes=6,
    ).to(device)

    probe.load_state_dict(torch.load(LOCKED_MODEL_PATH, map_location=device, weights_only=True))
    probe.eval()

    with torch.no_grad():
        t_logits, s_logits = probe(vision_features, clinical_features)
        t_probs = torch.softmax(t_logits, dim=1).cpu().numpy()
        s_probs = torch.softmax(s_logits, dim=1).cpu().numpy()

    # 6. Evaluation Metrics
    # Ground truth targets
    y_true_mal = df_eval[mal_col].apply(lambda x: 1 if (x is True or x == 1 or str(x).lower() in ["true", "1"]) else 0).values
    p_triage_mal = t_probs[:, 1]
    p_melanoma = s_probs[:, CLASS_TO_IDX["mel"]]
    s_derived_mal = np.sum(s_probs[:, MALIGNANT_INDICES], axis=1)

    # A. Global Triage Metrics
    n_total = len(y_true_mal)
    n_mal = int(np.sum(y_true_mal == 1))
    n_ben = int(np.sum(y_true_mal == 0))

    triage_auc = float(roc_auc_score(y_true_mal, p_triage_mal))

    h1_flag = p_triage_mal >= 0.20
    h2_flag = s_derived_mal >= 0.20
    consensus_flag = h1_flag | h2_flag

    triage_recall_020 = float(np.sum(h1_flag & (y_true_mal == 1)) / max(n_mal, 1))
    triage_spec_020 = float(np.sum((~h1_flag) & (y_true_mal == 0)) / max(n_ben, 1))
    mal_detected_h1 = int(np.sum(h1_flag & (y_true_mal == 1)))
    w_triage_low, w_triage_high = wilson_score_interval(mal_detected_h1, n_mal)

    consensus_recall = float(np.sum(consensus_flag & (y_true_mal == 1)) / max(n_mal, 1))
    consensus_spec = float(np.sum((~consensus_flag) & (y_true_mal == 0)) / max(n_ben, 1))
    mal_detected_cons = int(np.sum(consensus_flag & (y_true_mal == 1)))
    w_cons_low, w_cons_high = wilson_score_interval(mal_detected_cons, n_mal)

    # Full ROC Operating Points (STARD-AI Recommendation)
    fpr, tpr, roc_thresholds = roc_curve(y_true_mal, p_triage_mal)
    j_scores = tpr - fpr
    best_j_idx = int(np.argmax(j_scores))
    youden_tau = float(roc_thresholds[best_j_idx])
    youden_sens = float(tpr[best_j_idx])
    youden_spec = float(1.0 - fpr[best_j_idx])
    w_youden_sens_l, w_youden_sens_h = wilson_score_interval(int(round(youden_sens * n_mal)), n_mal)
    w_youden_spec_l, w_youden_spec_h = wilson_score_interval(int(round(youden_spec * n_ben)), n_ben)

    # Sensitivity at Fixed Specificities
    def sens_at_spec(target_spec: float) -> Tuple[float, float, int]:
        idx = int(np.argmin(np.abs((1.0 - fpr) - target_spec)))
        s = float(tpr[idx])
        t = float(roc_thresholds[idx])
        hits = int(round(s * n_mal))
        return s, t, hits

    sens_60, tau_60, hits_60 = sens_at_spec(0.60)
    sens_70, tau_70, hits_70 = sens_at_spec(0.70)
    sens_80, tau_80, hits_80 = sens_at_spec(0.80)

    # B. Melanoma Metrics
    is_mel = df_eval[disease_col].str.lower().str.contains("melanoma", na=False).values
    n_mel = int(np.sum(is_mel))
    mel_detected = int(np.sum((p_melanoma >= 0.05) & is_mel))
    mel_recall_005 = float(mel_detected / max(n_mel, 1))
    w_mel_low, w_mel_high = wilson_score_interval(mel_detected, n_mel)
    mel_spec_005 = float(np.sum((p_melanoma < 0.05) & (~is_mel)) / max(np.sum(~is_mel), 1))

    # C. Stratification by Fitzpatrick Skin Tone (Demographic Equity Audit)
    skin_tones = df_eval[tone_col].values
    unique_tones = sorted(list(set(skin_tones)))
    tone_results = {}

    for tone in unique_tones:
        mask_t = skin_tones == tone
        n_t = int(np.sum(mask_t))
        y_t = y_true_mal[mask_t]
        n_t_mal = int(np.sum(y_t == 1))
        n_t_ben = int(np.sum(y_t == 0))
        p_t = p_triage_mal[mask_t]

        t_auc_sub = float(roc_auc_score(y_t, p_t)) if len(set(y_t)) > 1 else 0.0
        c_flag_sub = consensus_flag[mask_t]
        rec_sub = float(np.sum(c_flag_sub & (y_t == 1)) / max(n_t_mal, 1))
        spec_sub = float(np.sum((~c_flag_sub) & (y_t == 0)) / max(n_t_ben, 1))
        det_sub = int(np.sum(c_flag_sub & (y_t == 1)))
        w_l, w_h = wilson_score_interval(det_sub, n_t_mal)

        tone_results[str(tone)] = {
            "total_images": n_t,
            "malignant_images": n_t_mal,
            "benign_images": n_t_ben,
            "triage_auc": t_auc_sub,
            "consensus_recall": rec_sub,
            "consensus_spec": spec_sub,
            "malignant_detected": det_sub,
            "wilson_ci_low": w_l,
            "wilson_ci_high": w_h,
        }

    # 7. Print Official CLAIM 2024 Locked Audit Report
    print("\n" + "=" * 80)
    print(" OFFICIAL STANFORD DDI LOCKED AUDIT REPORT (CLAIM 2024 / STARD-AI Standards)")
    print("=" * 80)
    print(f" Dataset: Stanford DDI (Diverse Dermatology Images)")
    print(f" Total Biopsied Images: {n_total} (Malignant: {n_mal}, Benign: {n_ben})")
    print(f" Model SHA-256: {actual_hash}\n")

    print(f" Primary Binary Triage ROC-AUC : {triage_auc:.4f}")
    print(f" [Pre-Registered Thresholds]")
    print(f"  - Triage Recall (tau=0.20)      : {mal_detected_h1}/{n_mal} ({triage_recall_020*100:.1f}%) | Wilson 95% CI: [{w_triage_low*100:.1f}%, {w_triage_high*100:.1f}%]")
    print(f"  - Triage Specificity (tau=0.20) : {triage_spec_020*100:.1f}%")
    print(f"  - Consensus Safety Recall (H1|H2): {mal_detected_cons}/{n_mal} ({consensus_recall*100:.1f}%) | Wilson 95% CI: [{w_cons_low*100:.1f}%, {w_cons_high*100:.1f}%]")
    print(f"  - Melanoma Recall (tau=0.05)    : {mel_detected}/{n_mel} ({mel_recall_005*100:.1f}%) | Wilson 95% CI: [{w_mel_low*100:.1f}%, {w_mel_high*100:.1f}%]\n")

    print(f" [STARD-AI Diagnostic ROC Operating Curve Analysis]")
    print(f"  - Optimal Youden Index (tau*={youden_tau:.4f}):")
    print(f"      Sensitivity: {int(round(youden_sens*n_mal))}/{n_mal} ({youden_sens*100:.1f}%) | Wilson 95% CI: [{w_youden_sens_l*100:.1f}%, {w_youden_sens_h*100:.1f}%]")
    print(f"      Specificity: {int(round(youden_spec*n_ben))}/{n_ben} ({youden_spec*100:.1f}%) | Wilson 95% CI: [{w_youden_spec_l*100:.1f}%, {w_youden_spec_h*100:.1f}%]")
    print(f"  - Sensitivity at 60% Specificity (tau={tau_60:.4f}): {hits_60}/{n_mal} ({sens_60*100:.1f}%)")
    print(f"  - Sensitivity at 70% Specificity (tau={tau_70:.4f}): {hits_70}/{n_mal} ({sens_70*100:.1f}%)")
    print(f"  - Sensitivity at 80% Specificity (tau={tau_80:.4f}): {hits_80}/{n_mal} ({sens_80*100:.1f}%)\n")

    print(" Demographic Equity Audit Across Fitzpatrick Skin Tones:")
    print(f" {'Skin Tone':<12} | {'Total':<6} | {'Malignant':<10} | {'AUROC':<8} | {'Safety Recall':<15} | {'Wilson 95% CI':<18}")
    print("-" * 80)
    for tone, r in tone_results.items():
        ci_str = f"[{r['wilson_ci_low']*100:.1f}%, {r['wilson_ci_high']*100:.1f}%]"
        det_str = f"{r['malignant_detected']}/{r['malignant_images']} ({r['consensus_recall']*100:.1f}%)"
        print(f" FST {tone:<8} | {r['total_images']:<6} | {r['malignant_images']:<10} | {r['triage_auc']:<8.4f} | {det_str:<15} | {ci_str:<18}")

    # 8. Save Official Locked Benchmark Artifact
    out_dir = PROJECT_ROOT / "artifacts" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "exp012_stanford_ddi_locked_evaluation.json"

    report_payload = {
        "benchmark": "EXP-012: Pre-Registered Locked External Evaluation on Stanford DDI",
        "standard": "CLAIM 2024 / STARD-AI / IEC 62304",
        "date": "2026-10-24",
        "model_path": str(LOCKED_MODEL_PATH),
        "model_sha256": actual_hash,
        "n_total": n_total,
        "n_malignant": n_mal,
        "n_benign": n_ben,
        "primary_triage_auc": triage_auc,
        "triage_recall_tau_020": triage_recall_020,
        "triage_recall_wilson_ci": [w_triage_low, w_triage_high],
        "triage_spec_tau_020": triage_spec_020,
        "consensus_safety_recall": consensus_recall,
        "consensus_safety_wilson_ci": [w_cons_low, w_cons_high],
        "consensus_spec": consensus_spec,
        "melanoma_recall_tau_005": mel_recall_005,
        "melanoma_recall_wilson_ci": [w_mel_low, w_mel_high],
        "melanoma_spec_tau_005": mel_spec_005,
        "youden_index": {
            "optimal_threshold": youden_tau,
            "sensitivity": youden_sens,
            "sensitivity_wilson_ci": [w_youden_sens_l, w_youden_sens_h],
            "specificity": youden_spec,
            "specificity_wilson_ci": [w_youden_spec_l, w_youden_spec_h],
        },
        "operating_points": {
            "spec_60": {"threshold": tau_60, "sensitivity": sens_60, "malignant_hits": hits_60},
            "spec_70": {"threshold": tau_70, "sensitivity": sens_70, "malignant_hits": hits_70},
            "spec_80": {"threshold": tau_80, "sensitivity": sens_80, "malignant_hits": hits_80},
        },
        "fitzpatrick_stratification": tone_results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Official locked audit report saved to: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
