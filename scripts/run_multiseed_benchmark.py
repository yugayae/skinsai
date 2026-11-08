#!/usr/bin/env python3
"""Multi-Seed Scientific Benchmark for SkinAI Foundation Models.

Executes a 5-seed stability evaluation across 5 random seeds: [0, 8, 27, 42, 2026].
Evaluates both:
1. Series A: Clean PAD-UFES-20 baseline (no added images)
2. Series B: Expanded Dataset (PAD-UFES-20 + confirmed ISIC clinical additions)

All evaluations are performed on the exact untouched test split (346 images, 205 patients).
Reports Mean ± Standard Deviation (SD) for every metric.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import open_clip
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.losses import ClassWeightedFocalLoss, compute_class_weights  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

SEEDS = [0, 8, 27, 42, 2026]
CLASS_ORDER = ["bcc", "scc", "ack", "sek", "mel", "nev"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
MALIGNANT_CLASSES = {"bcc", "scc", "mel"}
MALIGNANT_INDICES = [CLASS_TO_IDX[c] for c in MALIGNANT_CLASSES]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImageListDataset(Dataset):
    """Dataset loading from dataframe containing image_id, class_id, and image_path."""

    def __init__(self, records: List[dict], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        img_path = Path(rec["path"])
        class_id = rec["class_id"]
        label = CLASS_TO_IDX[class_id]

        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
        if self.transform:
            tensor = self.transform(rgb)
        else:
            tensor = transforms.ToTensor()(rgb)
        return tensor, label, rec["image_id"]


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_biomedclip: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    model.eval()
    embeds, labels, ids = [], [], []

    for imgs, tgts, img_ids in loader:
        imgs = imgs.to(device)
        if is_biomedclip:
            feat = model.encode_image(imgs)
        else:
            feat = model(imgs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        embeds.append(feat.cpu())
        labels.append(tgts)
        ids.extend(img_ids)

    return torch.cat(embeds, dim=0), torch.cat(labels, dim=0), ids


class LinearProbeHead(nn.Module):
    """Linear probe head with LayerNorm and Dropout."""

    def __init__(self, in_features: int, num_classes: int = 6, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_single_run(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    seed: int,
    epochs: int = 60,
    lr: float = 1e-3,
) -> nn.Module:
    set_seed(seed)
    in_dim = train_x.shape[1]
    head = LinearProbeHead(in_dim, num_classes=6).to(device)
    criterion = ClassWeightedFocalLoss(alpha=weights.to(device), gamma=2.0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_val_loss = float("inf")
    best_weights = None

    for _ in range(epochs):
        head.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = head(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        head.eval()
        with torch.no_grad():
            vx, vy = val_x.to(device), val_y.to(device)
            v_logits = head(vx)
            val_loss = criterion(v_logits, vy).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    if best_weights:
        head.load_state_dict(best_weights)
    return head


def evaluate_single_run(head: nn.Module, test_x: torch.Tensor, test_y: torch.Tensor, device: torch.device) -> Dict[str, float]:
    head.eval()
    with torch.no_grad():
        logits = head(test_x.to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    y_true = test_y.numpy()
    y_pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    n_classes = probs.shape[1]
    y_true_oh = np.eye(n_classes)[y_true]
    macro_auc = float(roc_auc_score(y_true_oh, probs, average="macro", multi_class="ovr"))
    macro_pr_auc = float(average_precision_score(y_true_oh, probs, average="macro"))

    # Melanoma metrics
    mel_idx = CLASS_TO_IDX["mel"]
    mel_mask = (y_true == mel_idx)
    mel_support = int(np.sum(mel_mask))
    mel_recall_naive = float(np.sum((y_pred == mel_idx) & mel_mask) / max(mel_support, 1))

    mel_probs = probs[:, mel_idx]
    mel_recall_005 = float(np.sum((mel_probs >= 0.05) & mel_mask) / max(mel_support, 1))
    mel_spec_005 = float(np.sum((mel_probs < 0.05) & (~mel_mask)) / max(np.sum(~mel_mask), 1))
    mel_recall_020 = float(np.sum((mel_probs >= 0.20) & mel_mask) / max(mel_support, 1))

    # Malignancy group metrics
    mal_mask = np.isin(y_true, MALIGNANT_INDICES)
    mal_support = int(np.sum(mal_mask))
    mal_probs = np.sum(probs[:, MALIGNANT_INDICES], axis=1)
    mal_recall_020 = float(np.sum((mal_probs >= 0.20) & mal_mask) / max(mal_support, 1))
    mal_spec_020 = float(np.sum((mal_probs < 0.20) & (~mal_mask)) / max(np.sum(~mal_mask), 1))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "macro_roc_auc": macro_auc,
        "macro_pr_auc": macro_pr_auc,
        "mel_recall_naive": mel_recall_naive,
        "mel_recall_tau_005": mel_recall_005,
        "mel_spec_tau_005": mel_spec_005,
        "mel_recall_tau_020": mel_recall_020,
        "mal_recall_tau_020": mal_recall_020,
        "mal_spec_tau_020": mal_spec_020,
    }


def aggregate_seed_metrics(seed_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = seed_results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in seed_results]
        agg[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": vals,
        }
    return agg


def main():
    print("=" * 75)
    print(" SkinAI Multi-Seed Scientific Benchmark across 5 Random Seeds")
    print(f" Seeds: {SEEDS}")
    print(" Evaluating: [Series A: Clean PAD-UFES-20] vs [Series B: Expanded + ISIC]")
    print("=" * 75)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device : {device} ({torch.cuda.get_device_name(0)})")

    # Paths
    split_dir = PROJECT_ROOT / "data" / "splits" / "smartphone_clinical"
    pad_dir = PROJECT_ROOT / "data" / "raw" / "pad_ufes_20"
    isic_dir = PROJECT_ROOT / "data" / "raw" / "isic_clinical"

    train_df = pd.read_csv(split_dir / "train.csv", keep_default_na=False)
    val_df = pd.read_csv(split_dir / "val.csv", keep_default_na=False)
    test_df = pd.read_csv(split_dir / "test.csv", keep_default_na=False)

    # 1. Base records
    train_records = [{"image_id": r["image_id"], "class_id": r["class_id"], "path": pad_dir / f"{r['image_id']}.jpg"} for _, r in train_df.iterrows()]
    val_records = [{"image_id": r["image_id"], "class_id": r["class_id"], "path": pad_dir / f"{r['image_id']}.jpg"} for _, r in val_df.iterrows()]
    test_records = [{"image_id": r["image_id"], "class_id": r["class_id"], "path": pad_dir / f"{r['image_id']}.jpg"} for _, r in test_df.iterrows()]

    # 2. ISIC additional confirmed records (Series B)
    isic_manifest = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "isic_clinical_manifest.csv", keep_default_na=False)
    new_isic_df = isic_manifest[~isic_manifest["is_existing_pad_ufes"]].copy()
    labeled_isic = new_isic_df[new_isic_df["class_id"] != "unmapped"].copy()

    isic_records = []
    for _, r in labeled_isic.iterrows():
        img_id = r["image_id"]
        p = isic_dir / f"{img_id}.jpg"
        if p.exists():
            isic_records.append({"image_id": img_id, "class_id": r["class_id"], "path": p})

    train_expanded_records = train_records + isic_records

    print(f"[*] Base Datasets     : Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")
    print(f"[*] Added ISIC Records: {len(isic_records)} confirmed clinical images")
    print(f"[*] Expanded Train    : {len(train_expanded_records)} total images (MEL: 34 -> 54, +59%)")

    # Weights
    from collections import Counter
    weights_base = compute_class_weights(Counter([r["class_id"] for r in train_records]), CLASS_ORDER, power=0.5)
    weights_exp = compute_class_weights(Counter([r["class_id"] for r in train_expanded_records]), CLASS_ORDER, power=0.5)

    all_benchmark_results = {}

    # =========================================================================
    # MODEL 1: Meta DINOv2 Small
    # =========================================================================
    print("\n" + "=" * 75)
    print(" MODEL 1: Meta DINOv2 Small (vit_small_patch14_dinov2, 518x518)")
    print("=" * 75)

    t0 = time.time()
    dino_model = timm.create_model("vit_small_patch14_dinov2", pretrained=True, num_classes=0).to(device)
    dino_cfg = timm.data.resolve_model_data_config(dino_model)
    dino_trans = timm.data.create_transform(**dino_cfg, is_training=False)

    print("[*] Extracting DINOv2 embeddings...")
    dino_tr_x, dino_tr_y, _ = extract_features(dino_model, DataLoader(ImageListDataset(train_records, dino_trans), batch_size=16, shuffle=False), device)
    dino_val_x, dino_val_y, _ = extract_features(dino_model, DataLoader(ImageListDataset(val_records, dino_trans), batch_size=16, shuffle=False), device)
    dino_te_x, dino_te_y, _ = extract_features(dino_model, DataLoader(ImageListDataset(test_records, dino_trans), batch_size=16, shuffle=False), device)
    dino_isic_x, dino_isic_y, _ = extract_features(dino_model, DataLoader(ImageListDataset(isic_records, dino_trans), batch_size=16, shuffle=False), device)

    dino_exp_tr_x = torch.cat([dino_tr_x, dino_isic_x], dim=0)
    dino_exp_tr_y = torch.cat([dino_tr_y, dino_isic_y], dim=0)
    print(f"[+] DINOv2 feature extraction done in {time.time()-t0:.1f}s")

    # Clean backbone from GPU memory
    del dino_model
    torch.cuda.empty_cache()

    # Series A: DINOv2 on Base Train across 5 seeds
    print("\n--- [DINOv2] Series A: Clean PAD-UFES-20 (5 Seeds) ---")
    dino_base_runs = []
    for s in SEEDS:
        head = train_single_run(dino_tr_x, dino_tr_y, dino_val_x, dino_val_y, weights_base, device, seed=s)
        m = evaluate_single_run(head, dino_te_x, dino_te_y, device)
        dino_base_runs.append(m)
        print(f"    Seed {s:>4}: Accuracy = {m['accuracy']*100:.2f}%, Macro ROC-AUC = {m['macro_roc_auc']:.4f}, Mel Recall(0.05) = {m['mel_recall_tau_005']*100:.1f}%")

    dino_base_agg = aggregate_seed_metrics(dino_base_runs)
    all_benchmark_results["DINOv2_Base"] = dino_base_agg

    # Series B: DINOv2 on Expanded Train across 5 seeds
    print("\n--- [DINOv2] Series B: Expanded Dataset + ISIC (5 Seeds) ---")
    dino_exp_runs = []
    for s in SEEDS:
        head = train_single_run(dino_exp_tr_x, dino_exp_tr_y, dino_val_x, dino_val_y, weights_exp, device, seed=s)
        m = evaluate_single_run(head, dino_te_x, dino_te_y, device)
        dino_exp_runs.append(m)
        print(f"    Seed {s:>4}: Accuracy = {m['accuracy']*100:.2f}%, Macro ROC-AUC = {m['macro_roc_auc']:.4f}, Mel Recall(0.05) = {m['mel_recall_tau_005']*100:.1f}%")

    dino_exp_agg = aggregate_seed_metrics(dino_exp_runs)
    all_benchmark_results["DINOv2_Expanded"] = dino_exp_agg

    # =========================================================================
    # MODEL 2: Microsoft BiomedCLIP
    # =========================================================================
    print("\n" + "=" * 75)
    print(" MODEL 2: Microsoft BiomedCLIP (PubMedBERT + ViT-B/16)")
    print("=" * 75)

    t0 = time.time()
    biomed_model, biomed_preprocess = open_clip.create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    biomed_model = biomed_model.to(device)

    print("[*] Extracting BiomedCLIP embeddings...")
    biomed_tr_x, biomed_tr_y, _ = extract_features(biomed_model, DataLoader(ImageListDataset(train_records, biomed_preprocess), batch_size=32, shuffle=False), device, is_biomedclip=True)
    biomed_val_x, biomed_val_y, _ = extract_features(biomed_model, DataLoader(ImageListDataset(val_records, biomed_preprocess), batch_size=32, shuffle=False), device, is_biomedclip=True)
    biomed_te_x, biomed_te_y, _ = extract_features(biomed_model, DataLoader(ImageListDataset(test_records, biomed_preprocess), batch_size=32, shuffle=False), device, is_biomedclip=True)
    biomed_isic_x, biomed_isic_y, _ = extract_features(biomed_model, DataLoader(ImageListDataset(isic_records, biomed_preprocess), batch_size=32, shuffle=False), device, is_biomedclip=True)

    biomed_exp_tr_x = torch.cat([biomed_tr_x, biomed_isic_x], dim=0)
    biomed_exp_tr_y = torch.cat([biomed_tr_y, biomed_isic_y], dim=0)
    print(f"[+] BiomedCLIP feature extraction done in {time.time()-t0:.1f}s")

    del biomed_model
    torch.cuda.empty_cache()

    # Series A: BiomedCLIP Base
    print("\n--- [BiomedCLIP] Series A: Clean PAD-UFES-20 (5 Seeds) ---")
    biomed_base_runs = []
    for s in SEEDS:
        head = train_single_run(biomed_tr_x, biomed_tr_y, biomed_val_x, biomed_val_y, weights_base, device, seed=s)
        m = evaluate_single_run(head, biomed_te_x, biomed_te_y, device)
        biomed_base_runs.append(m)
        print(f"    Seed {s:>4}: Accuracy = {m['accuracy']*100:.2f}%, Macro ROC-AUC = {m['macro_roc_auc']:.4f}, Mel Recall(0.05) = {m['mel_recall_tau_005']*100:.1f}%")

    biomed_base_agg = aggregate_seed_metrics(biomed_base_runs)
    all_benchmark_results["BiomedCLIP_Base"] = biomed_base_agg

    # Series B: BiomedCLIP Expanded
    print("\n--- [BiomedCLIP] Series B: Expanded Dataset + ISIC (5 Seeds) ---")
    biomed_exp_runs = []
    for s in SEEDS:
        head = train_single_run(biomed_exp_tr_x, biomed_exp_tr_y, biomed_val_x, biomed_val_y, weights_exp, device, seed=s)
        m = evaluate_single_run(head, biomed_te_x, biomed_te_y, device)
        biomed_exp_runs.append(m)
        print(f"    Seed {s:>4}: Accuracy = {m['accuracy']*100:.2f}%, Macro ROC-AUC = {m['macro_roc_auc']:.4f}, Mel Recall(0.05) = {m['mel_recall_tau_005']*100:.1f}%")

    biomed_exp_agg = aggregate_seed_metrics(biomed_exp_runs)
    all_benchmark_results["BiomedCLIP_Expanded"] = biomed_exp_agg

    # =========================================================================
    # FINAL STATISTICAL EVIDENCE TABLE (Mean ± SD)
    # =========================================================================
    print("\n" + "=" * 115)
    print(f"{'Metric':<30} | {'DINOv2 Base (5 Seeds)':<24} | {'DINOv2 Expanded (5 Seeds)':<26} | {'BiomedCLIP Base':<15} | {'BiomedCLIP Exp':<15}")
    print("-" * 115)

    metrics_display = [
        ("Macro ROC-AUC", "macro_roc_auc", "{:.4f} ± {:.4f}"),
        ("Macro PR-AUC", "macro_pr_auc", "{:.4f} ± {:.4f}"),
        ("Overall Accuracy", "accuracy", "{:.2%} ± {:.2%}"),
        ("Balanced Accuracy", "balanced_accuracy", "{:.2%} ± {:.2%}"),
        ("Macro F1-Score", "macro_f1", "{:.4f} ± {:.4f}"),
        ("Melanoma Recall (tau=0.05)", "mel_recall_tau_005", "{:.1%} ± {:.1%}"),
        ("Melanoma Spec. (tau=0.05)", "mel_spec_tau_005", "{:.1%}"),
        ("Melanoma Recall (Argmax)", "mel_recall_naive", "{:.1%} ± {:.1%}"),
        ("Malignancy Recall (tau=0.20)", "mal_recall_tau_020", "{:.1%} ± {:.1%}"),
        ("Malignancy Spec. (tau=0.20)", "mal_spec_tau_020", "{:.1%}"),
    ]

    for label, k, fmt in metrics_display:
        d_b = dino_base_agg[k]
        d_e = dino_exp_agg[k]
        b_b = biomed_base_agg[k]
        b_e = biomed_exp_agg[k]

        s_db = f"{d_b['mean']:.4f} ± {d_b['std']:.4f}" if "ROC" in label or "PR" in label or "F1" in label else f"{d_b['mean']*100:.1f}% ± {d_b['std']*100:.1f}%"
        s_de = f"{d_e['mean']:.4f} ± {d_e['std']:.4f}" if "ROC" in label or "PR" in label or "F1" in label else f"{d_e['mean']*100:.1f}% ± {d_e['std']*100:.1f}%"
        s_bb = f"{b_b['mean']:.4f} ± {b_b['std']:.4f}" if "ROC" in label or "PR" in label or "F1" in label else f"{b_b['mean']*100:.1f}% ± {b_b['std']*100:.1f}%"
        s_be = f"{b_e['mean']:.4f} ± {b_e['std']:.4f}" if "ROC" in label or "PR" in label or "F1" in label else f"{b_e['mean']*100:.1f}% ± {b_e['std']*100:.1f}%"

        print(f"{label:<30} | {s_db:<24} | {s_de:<26} | {s_bb:<15} | {s_be:<15}")

    print("=" * 115)

    # Save artifact
    out_file = PROJECT_ROOT / "artifacts" / "experiments" / "multiseed_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_benchmark_results, f, indent=2)
    print(f"\n[+] Full multi-seed benchmark artifact saved to: {out_file}")


if __name__ == "__main__":
    main()
