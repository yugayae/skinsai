#!/usr/bin/env python3
"""Foundation Models Evaluation Experiment for SkinAI.

Compares Dermatology & Vision Foundation Models against EfficientNet-B0 baseline:
1. Baseline: EfficientNet-B0 (from production checkpoint)
2. Meta DINOv2 Small (vit_small_patch14_dinov2, 384-dim self-supervised representation)
3. Microsoft BiomedCLIP (PubMedBERT + ViT-B/16, 512-dim biomedical representation)

Evaluates on the exact untouched test split (PAD-UFES-20: 346 images, 205 patients).
"""

from __future__ import annotations

import json
from pathlib import Path
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
    classification_report,
    f1_score,
    roc_auc_score,
)
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.losses import ClassWeightedFocalLoss, compute_class_weights  # noqa: E402

CLASS_ORDER = ["bcc", "scc", "ack", "sek", "mel", "nev"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
MALIGNANT_CLASSES = {"bcc", "scc", "mel"}
MALIGNANT_INDICES = [CLASS_TO_IDX[c] for c in MALIGNANT_CLASSES]


class ImageDataset(Dataset):
    """Simple image loader dataset returning RGB PIL images and targets."""

    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_id = str(row["image_id"])
        label = CLASS_TO_IDX[str(row["class_id"])]
        img_path = self.img_dir / f"{img_id}.jpg"
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
        if self.transform:
            tensor = self.transform(rgb)
        else:
            tensor = transforms.ToTensor()(rgb)
        return tensor, label, img_id


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_biomedclip: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Extract representation vectors from frozen foundation backbone."""
    model.eval()
    embeds = []
    labels = []
    ids = []

    for imgs, tgts, img_ids in loader:
        imgs = imgs.to(device)
        if is_biomedclip:
            feat = model.encode_image(imgs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        else:
            feat = model(imgs)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        embeds.append(feat.cpu())
        labels.append(tgts)
        ids.extend(img_ids)

    return torch.cat(embeds, dim=0), torch.cat(labels, dim=0), ids


class LinearProbeClassifier(nn.Module):
    """Lightweight 2-layer classifier head with LayerNorm and Dropout."""

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


def train_classifier_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    weights: torch.Tensor,
    num_classes: int = 6,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> nn.Module:
    """Train linear probe head with focal loss and validation early stopping."""
    in_dim = train_x.shape[1]
    head = LinearProbeClassifier(in_dim, num_classes=num_classes).to(device)
    criterion = ClassWeightedFocalLoss(alpha=weights.to(device), gamma=2.0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        head.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = head(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
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


def evaluate_head(
    head: nn.Module,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Compute scientific benchmark metrics on untouched test set."""
    head.eval()
    with torch.no_grad():
        logits = head(test_x.to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    y_true = test_y.numpy()
    y_pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # One-hot labels for AUCs
    n_classes = probs.shape[1]
    y_true_oh = np.eye(n_classes)[y_true]
    macro_auc = float(roc_auc_score(y_true_oh, probs, average="macro", multi_class="ovr"))
    macro_pr_auc = float(average_precision_score(y_true_oh, probs, average="macro"))

    # Melanoma (class 4) recall & PR-AUC
    mel_idx = CLASS_TO_IDX["mel"]
    mel_mask = (y_true == mel_idx)
    mel_support = int(np.sum(mel_mask))
    mel_recall_naive = float(np.sum((y_pred == mel_idx) & mel_mask) / max(mel_support, 1))

    # Recall at safety thresholds tau=0.05 and tau=0.20
    mel_probs = probs[:, mel_idx]
    mel_recall_005 = float(np.sum((mel_probs >= 0.05) & mel_mask) / max(mel_support, 1))
    mel_spec_005 = float(np.sum((mel_probs < 0.05) & (~mel_mask)) / max(np.sum(~mel_mask), 1))
    mel_recall_020 = float(np.sum((mel_probs >= 0.20) & mel_mask) / max(mel_support, 1))
    mel_spec_020 = float(np.sum((mel_probs < 0.20) & (~mel_mask)) / max(np.sum(~mel_mask), 1))
    mel_pr_auc = float(average_precision_score(y_true_oh[:, mel_idx], mel_probs))

    # Malignancy group (BCC + SCC + MEL)
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
        "mel_pr_auc": mel_pr_auc,
        "mel_recall_naive": mel_recall_naive,
        "mel_recall_tau_005": mel_recall_005,
        "mel_spec_tau_005": mel_spec_005,
        "mel_recall_tau_020": mel_recall_020,
        "mel_spec_tau_020": mel_spec_020,
        "mal_recall_tau_020": mal_recall_020,
        "mal_spec_tau_020": mal_spec_020,
    }


def main():
    print("=" * 75)
    print(" SkinAI Foundation Models Express Benchmark")
    print(" Comparing: Baseline EfficientNet-B0 vs Meta DINOv2 vs Microsoft BiomedCLIP")
    print("=" * 75)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device : {device}")
    if device.type == "cuda":
        print(f"[*] GPU Name       : {torch.cuda.get_device_name(0)}")

    split_dir = PROJECT_ROOT / "data" / "splits" / "smartphone_clinical"
    train_df = pd.read_csv(split_dir / "train.csv", keep_default_na=False)
    val_df = pd.read_csv(split_dir / "val.csv", keep_default_na=False)
    test_df = pd.read_csv(split_dir / "test.csv", keep_default_na=False)
    img_dir = PROJECT_ROOT / "data" / "raw" / "pad_ufes_20"

    print(f"[*] Data Splits    : Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Class weights for focal loss
    from collections import Counter
    train_counts = Counter(train_df["class_id"])
    weights = compute_class_weights(train_counts, CLASS_ORDER, power=0.5)

    results = {}

    # -------------------------------------------------------------
    # 0. Baseline: EfficientNet-B0 (from SCIENTIFIC_VALIDATION_REPORT.md)
    # -------------------------------------------------------------
    results["EfficientNet-B0 (Baseline)"] = {
        "accuracy": 0.6474,
        "balanced_accuracy": 0.5141,
        "macro_f1": 0.5089,
        "macro_roc_auc": 0.8552,
        "macro_pr_auc": 0.5772,
        "mel_pr_auc": 0.3196,
        "mel_recall_naive": 0.2500,
        "mel_recall_tau_005": 0.7500,
        "mel_spec_tau_005": 0.8728,
        "mel_recall_tau_020": 0.5000,
        "mel_spec_tau_020": 0.9763,
        "mal_recall_tau_020": 0.9568,
        "mal_spec_tau_020": 0.5489,
    }

    # -------------------------------------------------------------
    # 1. Meta DINOv2 Small (vit_small_patch14_dinov2)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("[1/2] Evaluating Meta DINOv2 Small (vit_small_patch14_dinov2)...")
    t0 = time.time()
    dino_model = timm.create_model("vit_small_patch14_dinov2", pretrained=True, num_classes=0).to(device)
    dino_data_config = timm.data.resolve_model_data_config(dino_model)
    dino_transform = timm.data.create_transform(**dino_data_config, is_training=False)

    train_ds_dino = ImageDataset(train_df, img_dir, transform=dino_transform)
    val_ds_dino = ImageDataset(val_df, img_dir, transform=dino_transform)
    test_ds_dino = ImageDataset(test_df, img_dir, transform=dino_transform)

    train_loader_dino = DataLoader(train_ds_dino, batch_size=16, shuffle=False)
    val_loader_dino = DataLoader(val_ds_dino, batch_size=16, shuffle=False)
    test_loader_dino = DataLoader(test_ds_dino, batch_size=16, shuffle=False)

    print("    Extracting DINOv2 embeddings...")
    dino_train_x, dino_train_y, _ = extract_embeddings(dino_model, train_loader_dino, device)
    dino_val_x, dino_val_y, _ = extract_embeddings(dino_model, val_loader_dino, device)
    dino_test_x, dino_test_y, _ = extract_embeddings(dino_model, test_loader_dino, device)
    print(f"    Embeddings shape: {dino_train_x.shape} (Time: {time.time()-t0:.1f}s)")

    print("    Training classification probe head...")
    dino_head = train_classifier_head(dino_train_x, dino_train_y, dino_val_x, dino_val_y, device, weights)
    dino_metrics = evaluate_head(dino_head, dino_test_x, dino_test_y, device)
    results["Meta DINOv2 Small"] = dino_metrics
    print(f"    [+] DINOv2 Macro ROC-AUC: {dino_metrics['macro_roc_auc']:.4f}, Accuracy: {dino_metrics['accuracy']*100:.2f}%, Mel Recall (tau=0.05): {dino_metrics['mel_recall_tau_005']*100:.1f}%")

    # Clean GPU memory
    del dino_model, dino_head
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 2. Microsoft BiomedCLIP
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("[2/2] Evaluating Microsoft BiomedCLIP (PubMedBERT + ViT-B/16)...")
    t0 = time.time()
    biomed_model, biomed_preprocess = open_clip.create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    biomed_model = biomed_model.to(device)

    train_ds_biomed = ImageDataset(train_df, img_dir, transform=biomed_preprocess)
    val_ds_biomed = ImageDataset(val_df, img_dir, transform=biomed_preprocess)
    test_ds_biomed = ImageDataset(test_df, img_dir, transform=biomed_preprocess)

    train_loader_biomed = DataLoader(train_ds_biomed, batch_size=32, shuffle=False)
    val_loader_biomed = DataLoader(val_ds_biomed, batch_size=32, shuffle=False)
    test_loader_biomed = DataLoader(test_ds_biomed, batch_size=32, shuffle=False)

    print("    Extracting BiomedCLIP embeddings...")
    biomed_train_x, biomed_train_y, _ = extract_embeddings(biomed_model, train_loader_biomed, device, is_biomedclip=True)
    biomed_val_x, biomed_val_y, _ = extract_embeddings(biomed_model, val_loader_biomed, device, is_biomedclip=True)
    biomed_test_x, biomed_test_y, _ = extract_embeddings(biomed_model, test_loader_biomed, device, is_biomedclip=True)
    print(f"    Embeddings shape: {biomed_train_x.shape} (Time: {time.time()-t0:.1f}s)")

    print("    Training classification probe head...")
    biomed_head = train_classifier_head(biomed_train_x, biomed_train_y, biomed_val_x, biomed_val_y, device, weights)
    biomed_metrics = evaluate_head(biomed_head, biomed_test_x, biomed_test_y, device)
    results["Microsoft BiomedCLIP"] = biomed_metrics
    print(f"    [+] BiomedCLIP Macro ROC-AUC: {biomed_metrics['macro_roc_auc']:.4f}, Accuracy: {biomed_metrics['accuracy']*100:.2f}%, Mel Recall (tau=0.05): {biomed_metrics['mel_recall_tau_005']*100:.1f}%")

    # Clean GPU memory
    del biomed_model, biomed_head
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Output Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"{'Metric / Indicator':<32} | {'EfficientNet-B0 (Base)':<22} | {'Meta DINOv2 Small':<18} | {'Microsoft BiomedCLIP':<20}")
    print("-" * 90)

    rows = [
        ("Macro ROC-AUC", "macro_roc_auc", "{:.4f}"),
        ("Macro PR-AUC", "macro_pr_auc", "{:.4f}"),
        ("Overall Accuracy", "accuracy", "{:.2%}"),
        ("Balanced Accuracy", "balanced_accuracy", "{:.2%}"),
        ("Macro F1-Score", "macro_f1", "{:.4f}"),
        ("Melanoma PR-AUC", "mel_pr_auc", "{:.4f}"),
        ("Melanoma Recall (Argmax)", "mel_recall_naive", "{:.1%}"),
        ("Melanoma Recall (tau=0.05)", "mel_recall_tau_005", "{:.1%}"),
        ("Melanoma Spec. (tau=0.05)", "mel_spec_tau_005", "{:.1%}"),
        ("Melanoma Recall (tau=0.20)", "mel_recall_tau_020", "{:.1%}"),
        ("Malignancy Recall (tau=0.20)", "mal_recall_tau_020", "{:.1%}"),
        ("Malignancy Spec. (tau=0.20)", "mal_spec_tau_020", "{:.1%}"),
    ]

    for label, key, fmt in rows:
        v_base = fmt.format(results["EfficientNet-B0 (Baseline)"][key])
        v_dino = fmt.format(results["Meta DINOv2 Small"][key])
        v_biomed = fmt.format(results["Microsoft BiomedCLIP"][key])
        print(f"{label:<32} | {v_base:<22} | {v_dino:<18} | {v_biomed:<20}")

    print("=" * 90)

    # Save artifact
    out_dir = PROJECT_ROOT / "artifacts" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "foundation_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved to {out_file}")


if __name__ == "__main__":
    main()
