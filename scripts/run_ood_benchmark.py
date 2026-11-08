"""EXP-009 Scientific Benchmark: OOD Quality Gate & Selective Prediction.

Evaluates 5 candidate OOD and Quality Gate algorithms on:
- Cohort 1: In-Distribution (InD) — 346 genuine clinical lesions (test.csv)
- Cohort 2: Out-of-Distribution (OOD Non-Skin) — 100 non-skin objects/textures/documents/noise
- Cohort 3: Out-of-Distribution (OOD Healthy Skin) — 100 genuine non-lesional healthy skin patches

Candidate Algorithms:
1. MSP (Maximum Softmax Probability)
2. Negative Predictive Entropy (-H)
3. Free Energy Score (Liu et al., 2020)
4. DINOv2 Skin Prototype Cosine Similarity (Sun et al., 2022)
5. DINOv2 Class Centroid Cosine Similarity
6. Dual Gate (DINOv2 Skin Cosine + Focal Lesion Contrast)

Metrics:
- AUROC (InD vs Non-Skin, InD vs Healthy Skin, InD vs All OOD)
- FPR@95%TPR (False Positive Rate at 95% True Positive Rate / Admission rate)
- Recommended Operating Thresholds
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import timm
import torch
import torch.nn as nn
from torchvision import transforms

PROJECT_ROOT = Path("c:/HomeWork/Medical/Dermatology")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.multi_head import CLASS_ORDER, MALIGNANT_CLASSES
from ml.models.multimodal import MultimodalLateFusionProbe
from scripts.run_multimodal_benchmark import extract_clinical_vector, train_multimodal

SEEDS = [0, 8, 27, 42, 2026]


def compute_fpr_at_95_tpr(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Compute False Positive Rate at 95% True Positive Rate, and the corresponding threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Find smallest index where TPR >= 0.95
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0, float(thresholds[-1])
    target_idx = idx[0]
    return float(fpr[target_idx]), float(thresholds[target_idx])


def compute_lesion_contrast_score(img_path: Path) -> float:
    """Compute local central contrast gradient: |L_center - L_surround|."""
    try:
        with Image.open(img_path) as img:
            gray = img.convert("L").resize((128, 128))
            arr = np.array(gray, dtype=np.float32)

            # Center 50%
            c_y1, c_y2 = 32, 96
            c_x1, c_x2 = 32, 96
            center = arr[c_y1:c_y2, c_x1:c_x2]

            # Surround (corners and margins)
            mask = np.ones((128, 128), dtype=bool)
            mask[c_y1:c_y2, c_x1:c_x2] = False
            surround = arr[mask]

            contrast = abs(float(np.mean(center)) - float(np.mean(surround)))
            # Standard deviation in center (texture sharpness)
            c_std = float(np.std(center))
            return contrast + 0.5 * c_std
    except Exception:
        return 0.0


def main():
    print("=" * 80)
    print(" EXP-009 Scientific Benchmark: OOD Quality Gate & Selective Prediction")
    print(" InD : 346 Clinical Skin Lesions (Frozen Test Set)")
    print(" OOD1: 100 Non-Skin Objects, Noise, Textures & Documents")
    print(" OOD2: 100 Healthy Human Skin Patches (No Lesion)")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Load Caches and Model
    cache_path = PROJECT_ROOT / "artifacts" / "cache" / "multi_head_dinov2_features.pt"
    cache = torch.load(cache_path, weights_only=False)
    train_vis, train_t, train_s = cache["train_x"], cache["train_t"], cache["train_s"]
    val_vis, val_t, val_s = cache["val_x"], cache["val_t"], cache["val_s"]
    test_vis, test_t, test_s = cache["test_x"], cache["test_t"], cache["test_s"]

    # Load DINOv2 model
    print("[*] Loading DINOv2 ViT-S/14 for OOD feature extraction...")
    dino = timm.create_model("vit_small_patch14_dinov2", pretrained=True, num_classes=0).to(device)
    dino.eval()

    dino_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Extract Features for OOD Cohorts
    non_skin_dir = PROJECT_ROOT / "data" / "ood_benchmark" / "non_skin"
    healthy_dir = PROJECT_ROOT / "data" / "ood_benchmark" / "healthy_skin"

    non_skin_paths = sorted(list(non_skin_dir.glob("*.jpg")))
    healthy_paths = sorted(list(healthy_dir.glob("*.jpg")))

    print(f"[*] Extracting embeddings for {len(non_skin_paths)} Non-Skin images...")
    non_skin_feats = []
    non_skin_contrasts = []
    with torch.no_grad():
        for p in non_skin_paths:
            with Image.open(p) as img:
                tensor = dino_transform(img.convert("RGB")).unsqueeze(0).to(device)
            f = dino(tensor)
            f = f / f.norm(dim=-1, keepdim=True)
            non_skin_feats.append(f.cpu())
            non_skin_contrasts.append(compute_lesion_contrast_score(p))
    non_skin_vis = torch.cat(non_skin_feats, dim=0)

    print(f"[*] Extracting embeddings for {len(healthy_paths)} Healthy Skin patches...")
    healthy_feats = []
    healthy_contrasts = []
    with torch.no_grad():
        for p in healthy_paths:
            with Image.open(p) as img:
                tensor = dino_transform(img.convert("RGB")).unsqueeze(0).to(device)
            f = dino(tensor)
            f = f / f.norm(dim=-1, keepdim=True)
            healthy_feats.append(f.cpu())
            healthy_contrasts.append(compute_lesion_contrast_score(p))
    healthy_vis = torch.cat(healthy_feats, dim=0)

    # InD Contrast scores
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "smartphone_clinical" / "test.csv")
    pad_dir = PROJECT_ROOT / "data" / "raw" / "pad_ufes_20"
    ind_contrasts = [compute_lesion_contrast_score(pad_dir / f"{r['image_id']}.jpg") for _, r in test_df.iterrows()]

    # 3. Compute DINOv2 Prototype & Class Centroids from Training Skin Lesions
    print("[*] Computing DINOv2 In-Distribution Skin Prototypes and Centroids...")
    train_vis_norm = train_vis / train_vis.norm(dim=-1, keepdim=True)
    skin_prototype = train_vis_norm.mean(dim=0, keepdim=True)
    skin_prototype = skin_prototype / skin_prototype.norm(dim=-1, keepdim=True)

    class_centroids = []
    for c in range(6):
        c_mask = (train_s == c)
        if c_mask.sum() > 0:
            c_mean = train_vis_norm[c_mask].mean(dim=0, keepdim=True)
            c_mean = c_mean / c_mean.norm(dim=-1, keepdim=True)
            class_centroids.append(c_mean)
        else:
            class_centroids.append(skin_prototype)
    class_centroids = torch.cat(class_centroids, dim=0)  # [6, 384]

    # Compute DINOv2 Skin Cosine Similarities
    ind_skin_sim = (test_vis @ skin_prototype.T).squeeze(-1).numpy()
    non_skin_sim = (non_skin_vis @ skin_prototype.T).squeeze(-1).numpy()
    healthy_skin_sim = (healthy_vis @ skin_prototype.T).squeeze(-1).numpy()

    # Compute Centroid Cosine Similarities
    ind_centroid_sim = (test_vis @ class_centroids.T).max(dim=-1).values.numpy()
    non_centroid_sim = (non_skin_vis @ class_centroids.T).max(dim=-1).values.numpy()
    healthy_centroid_sim = (healthy_vis @ class_centroids.T).max(dim=-1).values.numpy()

    # 4. Train Model and Compute Output-Based Scores (MSP, Entropy, Energy)
    with open(PROJECT_ROOT / "data" / "metadata" / "pad_ufes_20.json", "r") as f:
        pad_records = json.load(f)
    pad_meta = {item["isic_id"]: item.get("metadata", {}).get("clinical", {}) for item in pad_records}
    isic_manifest = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "isic_clinical_manifest.csv")
    new_isic_df = isic_manifest[~isic_manifest["is_existing_pad_ufes"]].copy()
    isic_meta = isic_manifest.set_index("image_id").to_dict(orient="index")

    train_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "smartphone_clinical" / "train.csv")
    val_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "smartphone_clinical" / "val.csv")

    train_ids = [str(r["image_id"]) for _, r in train_df.iterrows()]
    train_ids += [str(r["image_id"]) for _, r in new_isic_df.iterrows() if r["image_id"] in isic_meta]
    train_ids = train_ids[:len(train_vis)]
    val_ids = [str(r["image_id"]) for _, r in val_df.iterrows()]
    test_ids = [str(r["image_id"]) for _, r in test_df.iterrows()]

    train_clin = torch.tensor(
        np.array([extract_clinical_vector(iid, pad_meta, isic_meta, is_isic=(iid in isic_meta and iid not in pad_meta)) for iid in train_ids]),
        dtype=torch.float32,
    )
    val_clin = torch.tensor(
        np.array([extract_clinical_vector(iid, pad_meta, isic_meta) for iid in val_ids]),
        dtype=torch.float32,
    )
    test_clin = torch.tensor(
        np.array([extract_clinical_vector(iid, pad_meta, isic_meta) for iid in test_ids]),
        dtype=torch.float32,
    )
    # Default neutral clinical vector for OOD inputs
    neutral_clin = torch.zeros((1, 12), dtype=torch.float32)

    valid_s = train_s[train_s >= 0]
    counts_s = torch.bincount(valid_s, minlength=6).float()
    weights_s = 1.0 / torch.sqrt(torch.clamp(counts_s, min=1.0))
    weights_s = weights_s / weights_s.mean()

    valid_t = train_t[train_t >= 0]
    counts_t = torch.bincount(valid_t, minlength=2).float()
    weights_t = 1.0 / torch.sqrt(torch.clamp(counts_t, min=1.0))
    weights_t = weights_t / weights_t.mean()

    print("[*] Training champion model for scoring...")
    model = train_multimodal(
        train_vis, train_clin, train_t, train_s,
        val_vis, val_clin, val_t, val_s,
        weights_t, weights_s, device, seed=42, epochs=60
    )
    model.eval()

    def get_model_scores(vis_tensor: torch.Tensor, clin_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute MSP, -Entropy, and Energy score."""
        with torch.no_grad():
            _, s_probs_t = model.predict_proba(vis_tensor.to(device), clin_tensor.to(device))
            _, s_log = model(vis_tensor.to(device), clin_tensor.to(device))
            s_probs = s_probs_t.cpu().numpy()
            s_logits = s_log.cpu().numpy()

        msp = np.max(s_probs, axis=1)
        entropy = -np.sum(s_probs * np.log(np.clip(s_probs, 1e-12, 1.0)), axis=1)
        neg_entropy = -entropy
        # Energy score: E = logsumexp(s_logits)
        energy = np.log(np.sum(np.exp(s_logits), axis=1))
        return msp, neg_entropy, energy

    ind_msp, ind_neg_ent, ind_energy = get_model_scores(test_vis, test_clin)
    non_clin = neutral_clin.repeat(len(non_skin_vis), 1)
    non_msp, non_neg_ent, non_energy = get_model_scores(non_skin_vis, non_clin)
    hlth_clin = neutral_clin.repeat(len(healthy_vis), 1)
    hlth_msp, hlth_neg_ent, hlth_energy = get_model_scores(healthy_vis, hlth_clin)

    # 5. Evaluate All Candidate Detectors
    scores_dict = {
        "1. MSP (Max Softmax Prob)": {
            "ind": ind_msp,
            "non_skin": non_msp,
            "healthy": hlth_msp,
        },
        "2. Negative Entropy (-H)": {
            "ind": ind_neg_ent,
            "non_skin": non_neg_ent,
            "healthy": hlth_neg_ent,
        },
        "3. Free Energy Score": {
            "ind": ind_energy,
            "non_skin": non_energy,
            "healthy": hlth_energy,
        },
        "4. DINOv2 Skin Prototype Cosine": {
            "ind": ind_skin_sim,
            "non_skin": non_skin_sim,
            "healthy": healthy_skin_sim,
        },
        "5. DINOv2 Class Centroid Cosine": {
            "ind": ind_centroid_sim,
            "non_skin": non_centroid_sim,
            "healthy": healthy_centroid_sim,
        },
        "6. Focal Contrast (Lesion vs Healthy)": {
            "ind": np.array(ind_contrasts),
            "non_skin": np.array(non_skin_contrasts),
            "healthy": np.array(healthy_contrasts),
        },
        "7. SkinAI Dual Gate (DINOv2 Skin Cosine + Contrast)": {
            "ind": ind_skin_sim * 10.0 + np.clip(np.array(ind_contrasts) / 20.0, 0, 1.0),
            "non_skin": non_skin_sim * 10.0 + np.clip(np.array(non_skin_contrasts) / 20.0, 0, 1.0),
            "healthy": healthy_skin_sim * 10.0 + np.clip(np.array(healthy_contrasts) / 20.0, 0, 1.0),
        },
    }

    results = {}
    print("\n" + "=" * 95)
    print(f"{'OOD Candidate Method':<45} | {'Non-Skin AUC':<14} | {'Healthy AUC':<14} | {'All-OOD AUC':<14}")
    print("-" * 95)

    for method_name, sc in scores_dict.items():
        s_ind = sc["ind"]
        s_non = sc["non_skin"]
        s_hlth = sc["healthy"]
        s_all_ood = np.concatenate([s_non, s_hlth])

        # Task 1: InD vs Non-Skin (InD = 1, OOD = 0)
        y_t1 = np.concatenate([np.ones_like(s_ind), np.zeros_like(s_non)])
        sc_t1 = np.concatenate([s_ind, s_non])
        auc_t1 = float(roc_auc_score(y_t1, sc_t1))
        fpr95_t1, thresh_t1 = compute_fpr_at_95_tpr(y_t1, sc_t1)

        # Task 2: InD vs Healthy Skin
        y_t2 = np.concatenate([np.ones_like(s_ind), np.zeros_like(s_hlth)])
        sc_t2 = np.concatenate([s_ind, s_hlth])
        auc_t2 = float(roc_auc_score(y_t2, sc_t2))
        fpr95_t2, thresh_t2 = compute_fpr_at_95_tpr(y_t2, sc_t2)

        # Task 3: InD vs All OOD (Combined Gate)
        y_t3 = np.concatenate([np.ones_like(s_ind), np.zeros_like(s_all_ood)])
        sc_t3 = np.concatenate([s_ind, s_all_ood])
        auc_t3 = float(roc_auc_score(y_t3, sc_t3))
        fpr95_t3, thresh_t3 = compute_fpr_at_95_tpr(y_t3, sc_t3)

        results[method_name] = {
            "non_skin_auc": auc_t1,
            "non_skin_fpr95": fpr95_t1,
            "non_skin_thresh_95tpr": thresh_t1,
            "healthy_auc": auc_t2,
            "healthy_fpr95": fpr95_t2,
            "healthy_thresh_95tpr": thresh_t2,
            "all_ood_auc": auc_t3,
            "all_ood_fpr95": fpr95_t3,
            "all_ood_thresh_95tpr": thresh_t3,
        }

        print(f"{method_name:<45} | {auc_t1*100:6.2f}% (FPR {fpr95_t1*100:4.1f}%) | {auc_t2*100:6.2f}% (FPR {fpr95_t2*100:4.1f}%) | {auc_t3*100:6.2f}% (FPR {fpr95_t3*100:4.1f}%)")

    print("=" * 95)

    # Save to JSON
    out_file = PROJECT_ROOT / "artifacts" / "experiments" / "ood_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": "EXP-009",
                "name": "OOD Quality Gate & Selective Prediction Benchmark",
                "ind_samples": len(test_vis),
                "non_skin_samples": len(non_skin_vis),
                "healthy_skin_samples": len(healthy_vis),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[+] Full benchmark results saved to: {out_file}")


if __name__ == "__main__":
    main()
