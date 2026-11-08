"""EXP-008 Scientific Benchmark: ABCDE-Compliant Data Augmentations vs Naive Destructive Augmentations.

Dermatological Hypotheses:
1. Rule A & B (Asymmetry & Border Irregularity):
   Planar rotations (D4 dihedral group: 0, 90, 180, 270 deg and reflections) are strictly label-preserving
   and isometric. They model arbitrary smartphone orientation without distorting lesion contours.
2. Rule C (Color Constancy):
   Hue is a primary clinical diagnostic axis (melanocytic variegation vs erythematous pink BCC).
   Naive CV hue-jittering (e.g., hue=0.20) destroys diagnostic chromaticity and degrades model discriminability.
   ABCDE augmentation strictly enforces Delta Hue = 0.
3. Rule D (Diameter & Margin Preservation):
   Destructive crops (scale < 0.8) cut through lesion boundaries, removing border irregularity and falsifying relative size.
   ABCDE augmentation keeps scale in [0.90, 1.0] with border margin intact.
4. Camera Optics (Autofocus uncertainty):
   Subtle Gaussian blur (sigma in [0.1, 0.8]) models camera autofocus jitter without destroying microstructure.

Conditions evaluated across 5 random seeds [0, 8, 27, 42, 2026] on frozen test set (346 images, 205 patients):
- Condition A: EXP-007 Champion Baseline (Standard DINOv2 features, single-view test)
- Condition B: Test-Time Isometry (D4-TTA: 6-view dihedral group test averaging)
- Condition C: ABCDE-Compliant Training Augmentations (D4 rotation + Delta Hue=0 + border preservation + optics) + D4-TTA
- Condition D: Destructive / ABCDE-Violating Augmentations (Hue jitter=0.20 + aggressive crop scale=(0.4, 0.7))
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    auc,
)
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

PROJECT_ROOT = Path("c:/HomeWork/Medical/Dermatology")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.multi_head import (
    CLASS_ORDER,
    MALIGNANT_CLASSES,
    BENIGN_CLASSES,
    MultiHeadLoss,
    MaskedClassWeightedFocalLoss,
)
from ml.models.multimodal import MultimodalLateFusionProbe
from scripts.run_multimodal_benchmark import (
    extract_clinical_vector,
    evaluate_predictions,
    aggregate_metrics,
    train_multimodal,
)

SEEDS = [0, 8, 27, 42, 2026]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_ORDER)}
MALIGNANT_INDICES = [CLASS_TO_IDX[c] for c in MALIGNANT_CLASSES]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TransformImageDataset(Dataset):
    """Dataset applying custom PIL transformation pipeline."""

    def __init__(self, records: List[dict], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        with Image.open(rec["path"]) as img:
            rgb = img.convert("RGB")
        tensor = self.transform(rgb)
        return tensor, rec["triage_target"], rec["subtype_target"], rec["image_id"]


def build_abcde_transform():
    """ABCDE-Compliant Transform:
    - A & B: Random D4 dihedral symmetry (horizontal flip + vertical flip + 90 deg rotation)
    - C: Strict ZERO HUE SHIFT! Brightness/contrast jitter only
    - D: Scale [0.90, 1.0] preserving lesion borders
    - Optics: Subtle Gaussian blur simulating phone autofocus jitter
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(518, scale=(0.90, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.0),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_destructive_transform():
    """ABCDE-Violating Destructive Transform (Negative Control):
    - C VIOLATED: hue=0.20 alters diagnostic melanocytic pigmentation
    - A, B, D VIOLATED: aggressive crop scale=(0.40, 0.70) slices lesion borders
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(518, scale=(0.40, 0.70)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_features(dino, loader, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dino.eval()
    feats, t_targets, s_targets = [], [], []
    with torch.no_grad():
        for bx, bt, bs, _ in loader:
            bx = bx.to(device)
            f = dino(bx)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
            t_targets.append(bt)
            s_targets.append(bs)
    return torch.cat(feats, dim=0), torch.cat(t_targets, dim=0), torch.cat(s_targets, dim=0)


def main():
    print("=" * 80)
    print(" EXP-008 Scientific Benchmark: ABCDE-Compliant Data Augmentations")
    print(f" Seeds: {SEEDS}")
    print(" Frozen Test Set: 346 Clinical Images (205 Patients)")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device : {device} ({torch.cuda.get_device_name(0)})")

    split_dir = PROJECT_ROOT / "data" / "splits" / "smartphone_clinical"
    pad_dir = PROJECT_ROOT / "data" / "raw" / "pad_ufes_20"
    isic_dir = PROJECT_ROOT / "data" / "raw" / "isic_clinical"

    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    isic_manifest = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "isic_clinical_manifest.csv")
    new_isic_df = isic_manifest[~isic_manifest["is_existing_pad_ufes"]].copy()

    # Build Records
    def row_to_record(row, img_dir, is_isic=False):
        img_id = str(row["image_id"])
        c_id = str(row["class_id"])
        img_path = img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            return None
        s_target = CLASS_TO_IDX[c_id] if c_id in CLASS_TO_IDX else -1
        d1 = str(row.get("diagnosis_1", ""))
        if c_id in MALIGNANT_CLASSES or d1 == "Malignant":
            t_target = 1
        elif c_id in BENIGN_CLASSES or d1 == "Benign":
            t_target = 0
        else:
            t_target = -1
        return {
            "image_id": img_id,
            "path": img_path,
            "class_id": c_id,
            "triage_target": t_target,
            "subtype_target": s_target,
        }

    train_recs = [r for _, row in train_df.iterrows() if (r := row_to_record(row, pad_dir))]
    for _, row in new_isic_df.iterrows():
        rec = row_to_record(row, isic_dir, is_isic=True)
        if rec and (rec["triage_target"] != -1 or rec["subtype_target"] != -1):
            train_recs.append(rec)

    val_recs = [r for _, row in val_df.iterrows() if (r := row_to_record(row, pad_dir))]
    test_recs = [r for _, row in test_df.iterrows() if (r := row_to_record(row, pad_dir))]

    print(f"[*] Total Records -> Train: {len(train_recs)}, Val: {len(val_recs)}, Test: {len(test_recs)}")

    # 1. Load Clinical Questionnaire Metadata
    with open(PROJECT_ROOT / "data" / "metadata" / "pad_ufes_20.json", "r") as f:
        pad_records = json.load(f)
    pad_meta = {item["isic_id"]: item.get("metadata", {}).get("clinical", {}) for item in pad_records}
    isic_meta = isic_manifest.set_index("image_id").to_dict(orient="index")

    train_ids = [r["image_id"] for r in train_recs]
    val_ids = [r["image_id"] for r in val_recs]
    test_ids = [r["image_id"] for r in test_recs]

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

    # 2. Base Cache
    base_cache_path = PROJECT_ROOT / "artifacts" / "cache" / "multi_head_dinov2_features.pt"
    base_cache = torch.load(base_cache_path, weights_only=False)
    train_vis_base = base_cache["train_x"]
    val_vis = base_cache["val_x"]
    val_t = base_cache["val_t"]
    val_s = base_cache["val_s"]
    test_vis_base = base_cache["test_x"]
    test_t = base_cache["test_t"]
    test_s = base_cache["test_s"]
    train_t = base_cache["train_t"]
    train_s = base_cache["train_s"]

    # 3. TTA Cache
    tta_cache_path = PROJECT_ROOT / "artifacts" / "cache" / "test_d4_tta_dinov2.pt"
    test_vis_tta = torch.load(tta_cache_path, weights_only=False)

    # 4. ABCDE Training Feature Cache
    abcde_cache_path = PROJECT_ROOT / "artifacts" / "cache" / "train_abcde_dinov2_features.pt"
    destructive_cache_path = PROJECT_ROOT / "artifacts" / "cache" / "train_destructive_dinov2_features.pt"

    dino = None
    if not abcde_cache_path.exists() or not destructive_cache_path.exists():
        print("[*] Initializing DINOv2 ViT-S/14 for feature extraction...")
        dino = timm.create_model("vit_small_patch14_dinov2", pretrained=True, num_classes=0).to(device)
        dino.eval()

    if abcde_cache_path.exists():
        print(f"[*] Loading ABCDE training features from {abcde_cache_path}...")
        train_vis_abcde = torch.load(abcde_cache_path, weights_only=False)
    else:
        print("[*] Extracting ABCDE-Compliant augmented training features...")
        t0 = time.time()
        abcde_ds = TransformImageDataset(train_recs, transform=build_abcde_transform())
        abcde_loader = DataLoader(abcde_ds, batch_size=32, shuffle=False, num_workers=0)
        train_vis_abcde, _, _ = extract_features(dino, abcde_loader, device)
        torch.save(train_vis_abcde, abcde_cache_path)
        print(f"[+] Saved ABCDE features ({train_vis_abcde.shape}) in {time.time()-t0:.1f}s")

    if destructive_cache_path.exists():
        print(f"[*] Loading Destructive training features from {destructive_cache_path}...")
        train_vis_destr = torch.load(destructive_cache_path, weights_only=False)
    else:
        print("[*] Extracting Destructive augmented training features (Negative Control)...")
        t0 = time.time()
        destr_ds = TransformImageDataset(train_recs, transform=build_destructive_transform())
        destr_loader = DataLoader(destr_ds, batch_size=32, shuffle=False, num_workers=0)
        train_vis_destr, _, _ = extract_features(dino, destr_loader, device)
        torch.save(train_vis_destr, destructive_cache_path)
        print(f"[+] Saved Destructive features ({train_vis_destr.shape}) in {time.time()-t0:.1f}s")

    # Compute class weights
    valid_s = train_s[train_s >= 0]
    counts_s = torch.bincount(valid_s, minlength=6).float()
    weights_s = 1.0 / torch.sqrt(torch.clamp(counts_s, min=1.0))
    weights_s = weights_s / weights_s.mean()

    valid_t = train_t[train_t >= 0]
    counts_t = torch.bincount(valid_t, minlength=2).float()
    weights_t = 1.0 / torch.sqrt(torch.clamp(counts_t, min=1.0))
    weights_t = weights_t / weights_t.mean()

    print(f"[*] Weights -> Subtype: {weights_s.tolist()}, Triage: {weights_t.tolist()}")

    # Containers for results across seeds
    cond_a_runs = []  # Baseline Single-View
    cond_b_runs = []  # Test-Time D4-TTA Isometry
    cond_c_runs = []  # ABCDE-Augmented Training + D4-TTA
    cond_d_runs = []  # Destructive / ABCDE-Violating Augmentations

    print("\n" + "=" * 80)
    print(" Running 5-Seed Benchmark for All 4 Conditions...")
    print("=" * 80)

    for seed in SEEDS:
        print(f"\n--- [Random Seed {seed}] ---")

        # 1. Condition A & B: Model trained on standard features
        m_base = train_multimodal(
            train_vis_base, train_clin, train_t, train_s,
            val_vis, val_clin, val_t, val_s,
            weights_t, weights_s, device, seed=seed, epochs=60
        )
        m_base.eval()
        with torch.no_grad():
            # Cond A: Single-view evaluation
            at_probs, as_probs = m_base.predict_proba(test_vis_base.to(device), test_clin.to(device))
            a_res = evaluate_predictions(at_probs.cpu().numpy(), as_probs.cpu().numpy(), test_s.numpy())
            cond_a_runs.append(a_res)

            # Cond B: D4-TTA isometric evaluation
            bt_probs, bs_probs = m_base.predict_proba(test_vis_tta.to(device), test_clin.to(device))
            b_res = evaluate_predictions(bt_probs.cpu().numpy(), bs_probs.cpu().numpy(), test_s.numpy())
            cond_b_runs.append(b_res)

        print(f"  Cond A (Baseline Single) : Acc={a_res['accuracy']*100:.2f}%, AUC={a_res['macro_roc_auc']:.4f}, Mel_Rec={a_res['mel_recall_tau_005']*100:.1f}%, Tri_AUC={a_res['triage_roc_auc']:.4f}")
        print(f"  Cond B (ABCDE D4-TTA)    : Acc={b_res['accuracy']*100:.2f}%, AUC={b_res['macro_roc_auc']:.4f}, Mel_Rec={b_res['mel_recall_tau_005']*100:.1f}%, Tri_AUC={b_res['triage_roc_auc']:.4f}")

        # 2. Condition C: ABCDE Training Augmentation
        m_abcde = train_multimodal(
            train_vis_abcde, train_clin, train_t, train_s,
            val_vis, val_clin, val_t, val_s,
            weights_t, weights_s, device, seed=seed, epochs=60
        )
        m_abcde.eval()
        with torch.no_grad():
            ct_probs, cs_probs = m_abcde.predict_proba(test_vis_tta.to(device), test_clin.to(device))
            c_res = evaluate_predictions(ct_probs.cpu().numpy(), cs_probs.cpu().numpy(), test_s.numpy())
            cond_c_runs.append(c_res)

        print(f"  Cond C (ABCDE Train+TTA) : Acc={c_res['accuracy']*100:.2f}%, AUC={c_res['macro_roc_auc']:.4f}, Mel_Rec={c_res['mel_recall_tau_005']*100:.1f}%, Tri_AUC={c_res['triage_roc_auc']:.4f}")

        # 3. Condition D: Destructive Augmentation (Negative Control)
        m_destr = train_multimodal(
            train_vis_destr, train_clin, train_t, train_s,
            val_vis, val_clin, val_t, val_s,
            weights_t, weights_s, device, seed=seed, epochs=60
        )
        m_destr.eval()
        with torch.no_grad():
            dt_probs, ds_probs = m_destr.predict_proba(test_vis_base.to(device), test_clin.to(device))
            d_res = evaluate_predictions(dt_probs.cpu().numpy(), ds_probs.cpu().numpy(), test_s.numpy())
            cond_d_runs.append(d_res)

        print(f"  Cond D (Destructive Neg) : Acc={d_res['accuracy']*100:.2f}%, AUC={d_res['macro_roc_auc']:.4f}, Mel_Rec={d_res['mel_recall_tau_005']*100:.1f}%, Tri_AUC={d_res['triage_roc_auc']:.4f}")

    # Aggregate Metrics
    agg_a = aggregate_metrics(cond_a_runs)
    agg_b = aggregate_metrics(cond_b_runs)
    agg_c = aggregate_metrics(cond_c_runs)
    agg_d = aggregate_metrics(cond_d_runs)

    print("\n" + "=" * 90)
    print(" EXP-008 FINAL MULTISEED STATISTICAL SUMMARY (5 SEEDS: [0, 8, 27, 42, 2026])")
    print("=" * 90)
    header = f"{'Metric':<28} | {'Cond A (Baseline)':<18} | {'Cond B (D4-TTA)':<18} | {'Cond C (ABCDE-Aug)':<18} | {'Cond D (Destructive)':<18}"
    print(header)
    print("-" * len(header))

    key_metrics = [
        ("accuracy", "Overall Accuracy (%)", 100.0),
        ("balanced_accuracy", "Balanced Accuracy (%)", 100.0),
        ("macro_f1", "Macro F1-Score (%)", 100.0),
        ("macro_roc_auc", "Macro ROC-AUC (Subtypes)", 1.0),
        ("macro_pr_auc", "Macro PR-AUC", 1.0),
        ("mel_recall_tau_005", "Melanoma Recall (tau=0.05)", 100.0),
        ("mel_spec_tau_005", "Melanoma Spec (tau=0.05)", 100.0),
        ("triage_roc_auc", "Head 1 Triage ROC-AUC", 1.0),
        ("triage_recall_tau_020", "Head 1 Triage Recall (0.20)", 100.0),
        ("triage_spec_tau_020", "Head 1 Triage Spec (0.20)", 100.0),
        ("consensus_agreement_rate", "Head Agreement Rate (%)", 100.0),
        ("consensus_safety_recall", "Consensus Safety Recall", 100.0),
        ("consensus_safety_spec", "Consensus Safety Spec", 100.0),
    ]

    for metric_key, label, scale in key_metrics:
        va = f"{agg_a[metric_key]['mean']*scale:.2f} +/- {agg_a[metric_key]['std']*scale:.2f}"
        vb = f"{agg_b[metric_key]['mean']*scale:.2f} +/- {agg_b[metric_key]['std']*scale:.2f}"
        vc = f"{agg_c[metric_key]['mean']*scale:.2f} +/- {agg_c[metric_key]['std']*scale:.2f}"
        vd = f"{agg_d[metric_key]['mean']*scale:.2f} +/- {agg_d[metric_key]['std']*scale:.2f}"
        print(f"{label:<28} | {va:<18} | {vb:<18} | {vc:<18} | {vd:<18}")

    print("=" * 90)

    # Save to JSON artifact
    out_dir = PROJECT_ROOT / "artifacts" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "abcde_benchmark.json"

    benchmark_data = {
        "experiment_id": "EXP-008",
        "name": "ABCDE-Compliant Data Augmentations vs Naive Destructive Augmentations",
        "date": "2026-10-21",
        "seeds": SEEDS,
        "test_dataset": "data/splits/smartphone_clinical/test.csv",
        "test_images_count": len(test_recs),
        "conditions": {
            "condition_a_baseline": agg_a,
            "condition_b_d4_tta": agg_b,
            "condition_c_abcde_augmented": agg_c,
            "condition_d_destructive_control": agg_d,
        },
        "raw_runs": {
            "condition_a": cond_a_runs,
            "condition_b": cond_b_runs,
            "condition_c": cond_c_runs,
            "condition_d": cond_d_runs,
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Full scientific benchmark saved to: {out_file}")


if __name__ == "__main__":
    main()
