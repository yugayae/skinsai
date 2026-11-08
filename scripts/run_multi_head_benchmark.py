"""Multi-Head Architecture Benchmark: Decoupled Binary Oncology Triage + 6-Class Subtyping.

Compares:
1. Baseline Single-Head DINOv2 (EXP-005)
2. Multi-Head DINOv2 (Dedicated Head 1 Triage + Head 2 Subtypes)

Evaluated across 5 seeds: [0, 8, 27, 42, 2026] on frozen test set (346 images).
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path("c:/HomeWork/Medical/Dermatology")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.multi_head import (
    CLASS_ORDER,
    MALIGNANT_CLASSES,
    BENIGN_CLASSES,
    MultiHeadProbe,
    MultiHeadLoss,
    MaskedClassWeightedFocalLoss,
)

PROJECT_ROOT = Path("c:/HomeWork/Medical/Dermatology")
SEEDS = [0, 8, 27, 42, 2026]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_ORDER)}
MALIGNANT_INDICES = [CLASS_TO_IDX[c] for c in MALIGNANT_CLASSES]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiHeadImageDataset(Dataset):
    """Dataset providing (image, triage_target, subtype_target, image_id)."""

    def __init__(self, records: List[dict], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        rec = self.records[idx]
        img_path = Path(rec["path"])
        triage_target = rec["triage_target"]
        subtype_target = rec["subtype_target"]

        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
        if self.transform:
            tensor = self.transform(rgb)
        else:
            tensor = transforms.ToTensor()(rgb)
        return tensor, triage_target, subtype_target, rec["image_id"]


@torch.no_grad()
def extract_multi_head_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    model.eval()
    embeds, t_targets, s_targets, ids = [], [], [], []

    for imgs, t_tgts, s_tgts, img_ids in loader:
        imgs = imgs.to(device)
        feat = model(imgs)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        embeds.append(feat.cpu())
        t_targets.append(t_tgts)
        s_targets.append(s_tgts)
        ids.extend(img_ids)

    return (
        torch.cat(embeds, dim=0),
        torch.cat(t_targets, dim=0),
        torch.cat(s_targets, dim=0),
        ids,
    )


class SingleHeadBaselineProbe(nn.Module):
    """Single head baseline matching EXP-005."""

    def __init__(self, in_features: int, num_classes: int = 6, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_single_head(
    train_x: torch.Tensor,
    train_s: torch.Tensor,
    val_x: torch.Tensor,
    val_s: torch.Tensor,
    weights_s: torch.Tensor,
    device: torch.device,
    seed: int,
    epochs: int = 60,
    lr: float = 1e-3,
) -> nn.Module:
    set_seed(seed)
    valid_train = train_s >= 0
    tx, ty = train_x[valid_train], train_s[valid_train]
    valid_val = val_s >= 0
    vx, vy = val_x[valid_val], val_s[valid_val]

    head = SingleHeadBaselineProbe(tx.shape[1], num_classes=6).to(device)
    criterion = MaskedClassWeightedFocalLoss(alpha=weights_s.to(device), gamma=2.0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_ds = torch.utils.data.TensorDataset(tx, ty)
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
            v_logits = head(vx.to(device))
            val_loss = criterion(v_logits, vy.to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    if best_weights:
        head.load_state_dict(best_weights)
    return head


def train_multi_head(
    train_x: torch.Tensor,
    train_t: torch.Tensor,
    train_s: torch.Tensor,
    val_x: torch.Tensor,
    val_t: torch.Tensor,
    val_s: torch.Tensor,
    weights_t: torch.Tensor,
    weights_s: torch.Tensor,
    device: torch.device,
    seed: int,
    epochs: int = 60,
    lr: float = 1e-3,
    triage_weight: float = 1.0,
    subtype_weight: float = 1.0,
) -> MultiHeadProbe:
    set_seed(seed)
    in_dim = train_x.shape[1]
    probe = MultiHeadProbe(in_features=in_dim, num_subtypes=6).to(device)

    loss_fn = MultiHeadLoss(
        triage_weight=triage_weight,
        subtype_weight=subtype_weight,
        triage_criterion=MaskedClassWeightedFocalLoss(alpha=weights_t.to(device), gamma=2.0),
        subtype_criterion=MaskedClassWeightedFocalLoss(alpha=weights_s.to(device), gamma=2.0),
    )

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_ds = torch.utils.data.TensorDataset(train_x, train_t, train_s)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_val_loss = float("inf")
    best_weights = None

    for _ in range(epochs):
        probe.train()
        for bx, bt, bs in train_loader:
            bx, bt, bs = bx.to(device), bt.to(device), bs.to(device)
            optimizer.zero_grad()
            t_logits, s_logits = probe(bx)
            loss, _, _ = loss_fn(t_logits, s_logits, bt, bs)
            loss.backward()
            optimizer.step()
        scheduler.step()

        probe.eval()
        with torch.no_grad():
            vt_logits, vs_logits = probe(val_x.to(device))
            val_loss, _, _ = loss_fn(vt_logits, vs_logits, val_t.to(device), val_s.to(device))
            val_loss_val = val_loss.item()

        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            best_weights = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    if best_weights:
        probe.load_state_dict(best_weights)

    # Calibrate temperatures
    probe.calibrate_temperatures(
        val_features=val_x.to(device),
        val_triage_targets=val_t.to(device),
        val_subtype_targets=val_s.to(device),
    )
    return probe


def evaluate_single_head_baseline(
    head: nn.Module,
    test_x: torch.Tensor,
    test_s: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    head.eval()
    with torch.no_grad():
        logits = head(test_x.to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    y_true = test_s.numpy()
    y_pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        y_true_onehot = np.eye(6)[y_true]
        macro_auc = float(roc_auc_score(y_true_onehot, probs, average="macro", multi_class="ovr"))
    except Exception:
        macro_auc = 0.0

    pr_aucs = []
    for c in range(6):
        try:
            p_c, r_c, _ = precision_recall_curve((y_true == c).astype(int), probs[:, c])
            pr_aucs.append(float(auc(r_c, p_c)))
        except Exception:
            pass
    macro_pr_auc = float(np.mean(pr_aucs)) if pr_aucs else 0.0

    # Melanoma metrics
    mel_idx = CLASS_TO_IDX["mel"]
    mel_mask = (y_true == mel_idx)
    mel_support = int(np.sum(mel_mask))
    mel_probs = probs[:, mel_idx]
    mel_recall_005 = float(np.sum((mel_probs >= 0.05) & mel_mask) / max(mel_support, 1))
    mel_spec_005 = float(np.sum((mel_probs < 0.05) & (~mel_mask)) / max(np.sum(~mel_mask), 1))

    # Malignancy group metrics (sum of bcc + scc + mel)
    mal_mask = np.isin(y_true, MALIGNANT_INDICES)
    mal_support = int(np.sum(mal_mask))
    mal_probs = np.sum(probs[:, MALIGNANT_INDICES], axis=1)
    mal_recall_020 = float(np.sum((mal_probs >= 0.20) & mal_mask) / max(mal_support, 1))
    mal_spec_020 = float(np.sum((mal_probs < 0.20) & (~mal_mask)) / max(np.sum(~mal_mask), 1))

    # Binary triage ROC-AUC from summed probabilities
    try:
        triage_auc_from_sum = float(roc_auc_score(mal_mask.astype(int), mal_probs))
    except Exception:
        triage_auc_from_sum = 0.0

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "macro_roc_auc": macro_auc,
        "macro_pr_auc": macro_pr_auc,
        "mel_recall_tau_005": mel_recall_005,
        "mel_spec_tau_005": mel_spec_005,
        "mal_recall_tau_020": mal_recall_020,
        "mal_spec_tau_020": mal_spec_020,
        "triage_auc": triage_auc_from_sum,
    }


def evaluate_multi_head(
    probe: MultiHeadProbe,
    test_x: torch.Tensor,
    test_t: torch.Tensor,
    test_s: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    probe.eval()
    with torch.no_grad():
        t_probs, s_probs = probe.predict_proba(test_x.to(device))
        t_probs = t_probs.cpu().numpy()
        s_probs = s_probs.cpu().numpy()

    y_true_s = test_s.numpy()
    y_pred_s = np.argmax(s_probs, axis=1)

    # 1. Head 2 (Subtyping) metrics
    acc = float(accuracy_score(y_true_s, y_pred_s))
    bal_acc = float(balanced_accuracy_score(y_true_s, y_pred_s))
    macro_f1 = float(f1_score(y_true_s, y_pred_s, average="macro", zero_division=0))

    try:
        y_true_onehot = np.eye(6)[y_true_s]
        macro_auc = float(roc_auc_score(y_true_onehot, s_probs, average="macro", multi_class="ovr"))
    except Exception:
        macro_auc = 0.0

    pr_aucs = []
    for c in range(6):
        try:
            p_c, r_c, _ = precision_recall_curve((y_true_s == c).astype(int), s_probs[:, c])
            pr_aucs.append(float(auc(r_c, p_c)))
        except Exception:
            pass
    macro_pr_auc = float(np.mean(pr_aucs)) if pr_aucs else 0.0

    mel_idx = CLASS_TO_IDX["mel"]
    mel_mask = (y_true_s == mel_idx)
    mel_support = int(np.sum(mel_mask))
    mel_probs = s_probs[:, mel_idx]
    mel_recall_005 = float(np.sum((mel_probs >= 0.05) & mel_mask) / max(mel_support, 1))
    mel_spec_005 = float(np.sum((mel_probs < 0.05) & (~mel_mask)) / max(np.sum(~mel_mask), 1))

    # Head 2 derived malignancy
    mal_mask = np.isin(y_true_s, MALIGNANT_INDICES)
    mal_support = int(np.sum(mal_mask))
    mal_derived_probs = np.sum(s_probs[:, MALIGNANT_INDICES], axis=1)

    # 2. Head 1 (Dedicated Triage) metrics
    p_triage_mal = t_probs[:, 1]  # P(malignant) from Head 1
    try:
        triage_roc_auc = float(roc_auc_score(mal_mask.astype(int), p_triage_mal))
    except Exception:
        triage_roc_auc = 0.0

    try:
        p_prec, p_rec, _ = precision_recall_curve(mal_mask.astype(int), p_triage_mal)
        triage_pr_auc = float(auc(p_rec, p_prec))
    except Exception:
        triage_pr_auc = 0.0

    # Dedicated Head 1 sensitivity and specificity at tau=0.20 and tau=0.50
    triage_recall_020 = float(np.sum((p_triage_mal >= 0.20) & mal_mask) / max(mal_support, 1))
    triage_spec_020 = float(np.sum((p_triage_mal < 0.20) & (~mal_mask)) / max(np.sum(~mal_mask), 1))
    triage_recall_050 = float(np.sum((p_triage_mal >= 0.50) & mal_mask) / max(mal_support, 1))
    triage_spec_050 = float(np.sum((p_triage_mal < 0.50) & (~mal_mask)) / max(np.sum(~mal_mask), 1))

    # 3. Clinical Agreement & Consensus
    # Agreement at tau=0.20
    head1_referral = (p_triage_mal >= 0.20)
    head2_referral = (mal_derived_probs >= 0.20)
    agreement_mask = (head1_referral == head2_referral)
    agreement_rate = float(np.mean(agreement_mask))

    # Conservative Consensus Trigger: Referral if EITHER head flags malignancy
    consensus_referral = head1_referral | head2_referral
    consensus_recall = float(np.sum(consensus_referral & mal_mask) / max(mal_support, 1))
    consensus_spec = float(np.sum((~consensus_referral) & (~mal_mask)) / max(np.sum(~mal_mask), 1))

    return {
        # Head 2 (Subtypes)
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "macro_roc_auc": macro_auc,
        "macro_pr_auc": macro_pr_auc,
        "mel_recall_tau_005": mel_recall_005,
        "mel_spec_tau_005": mel_spec_005,
        # Head 1 (Dedicated Triage)
        "triage_roc_auc": triage_roc_auc,
        "triage_pr_auc": triage_pr_auc,
        "triage_recall_tau_020": triage_recall_020,
        "triage_spec_tau_020": triage_spec_020,
        "triage_recall_tau_050": triage_recall_050,
        "triage_spec_tau_050": triage_spec_050,
        # Joint Consensus
        "consensus_agreement_rate": agreement_rate,
        "consensus_safety_recall": consensus_recall,
        "consensus_safety_spec": consensus_spec,
    }


def aggregate_metrics(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in results]
        agg[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": vals,
        }
    return agg


def main():
    print("=" * 80)
    print(" SkinAI Multi-Head Architecture Scientific Benchmark across 5 Random Seeds")
    print(f" Seeds: {SEEDS}")
    print(" Models: [Single-Head Baseline DINOv2] vs [Multi-Head Decoupled DINOv2]")
    print(" Datasets: PAD-UFES-20 + ISIC Close-up Permissive (CC-0 / CC-BY)")
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

    # Build Records with triage_target and subtype_target
    def row_to_record(row, img_dir, is_isic=False):
        img_id = str(row["image_id"])
        c_id = str(row["class_id"])
        img_path = img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            return None

        # Subtype target
        if c_id in CLASS_TO_IDX:
            s_target = CLASS_TO_IDX[c_id]
        else:
            s_target = -1  # Masked for 6-class loss

        # Triage target: 1 = Malignant (bcc, scc, mel), 0 = Benign (nev, sek, ISIC Benign)
        # ack = -1 (masked from binary malignancy triage)
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
    val_recs = [r for _, row in val_df.iterrows() if (r := row_to_record(row, pad_dir))]
    test_recs = [r for _, row in test_df.iterrows() if (r := row_to_record(row, pad_dir))]

    # Add all valid new ISIC images to training pool
    for _, row in new_isic_df.iterrows():
        rec = row_to_record(row, isic_dir, is_isic=True)
        if rec and (rec["triage_target"] != -1 or rec["subtype_target"] != -1):
            train_recs.append(rec)

    print(f"[*] Total Records -> Train: {len(train_recs)}, Val: {len(val_recs)}, Test: {len(test_recs)}")

    # Check targets breakdown in train
    t_targets = [r["triage_target"] for r in train_recs]
    s_targets = [r["subtype_target"] for r in train_recs]
    print(f"[*] Training Triage targets breakdown: 0 (Benign)={t_targets.count(0)}, 1 (Malignant)={t_targets.count(1)}, -1 (Masked ACK)={t_targets.count(-1)}")
    print(f"[*] Training Subtype targets breakdown: Valid 6-class={sum(1 for x in s_targets if x >= 0)}, Masked general={s_targets.count(-1)}")

    # Extract Features with DINOv2 Small (518x518)
    cache_path = PROJECT_ROOT / "artifacts" / "cache" / "multi_head_dinov2_features.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"[*] Loading pre-extracted DINOv2 features from {cache_path}...")
        cache = torch.load(cache_path)
        train_x, train_t, train_s = cache["train_x"], cache["train_t"], cache["train_s"]
        val_x, val_t, val_s = cache["val_x"], cache["val_t"], cache["val_s"]
        test_x, test_t, test_s = cache["test_x"], cache["test_t"], cache["test_s"]
    else:
        print("[*] Extracting DINOv2 embeddings on RTX 3060...")
        dino = timm.create_model("vit_small_patch14_dinov2", pretrained=True, num_classes=0).to(device)
        dino_cfg = timm.data.resolve_model_data_config(dino)
        dino_trans = timm.data.create_transform(**dino_cfg, is_training=False)

        train_ds = MultiHeadImageDataset(train_recs, transform=dino_trans)
        val_ds = MultiHeadImageDataset(val_recs, transform=dino_trans)
        test_ds = MultiHeadImageDataset(test_recs, transform=dino_trans)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

        train_x, train_t, train_s, _ = extract_multi_head_features(dino, train_loader, device)
        val_x, val_t, val_s, _ = extract_multi_head_features(dino, val_loader, device)
        test_x, test_t, test_s, _ = extract_multi_head_features(dino, test_loader, device)

        torch.save({
            "train_x": train_x, "train_t": train_t, "train_s": train_s,
            "val_x": val_x, "val_t": val_t, "val_s": val_s,
            "test_x": test_x, "test_t": test_t, "test_s": test_s,
        }, cache_path)
        print(f"[+] Features cached successfully to {cache_path}")

    # Compute class weights
    # Subtype weights (inverse square root)
    valid_s = train_s[train_s >= 0]
    counts_s = torch.bincount(valid_s, minlength=6).float()
    weights_s = 1.0 / torch.sqrt(torch.clamp(counts_s, min=1.0))
    weights_s = weights_s / weights_s.mean()

    # Triage weights (0 = Benign, 1 = Malignant)
    valid_t = train_t[train_t >= 0]
    counts_t = torch.bincount(valid_t, minlength=2).float()
    weights_t = 1.0 / torch.sqrt(torch.clamp(counts_t, min=1.0))
    weights_t = weights_t / weights_t.mean()

    print(f"[*] Subtype weights: {weights_s.tolist()}")
    print(f"[*] Triage weights: {weights_t.tolist()}")

    # Training & Evaluation across 5 seeds
    baseline_runs = []
    multi_head_runs = []

    print("\n" + "=" * 80)
    print(" Running Multi-Seed Benchmarks across 5 Seeds...")
    print("=" * 80)

    for seed in SEEDS:
        print(f"\n--- [Seed {seed}] ---")
        # 1. Single Head Baseline
        sh_model = train_single_head(
            train_x, train_s, val_x, val_s, weights_s, device, seed=seed, epochs=60
        )
        sh_res = evaluate_single_head_baseline(sh_model, test_x, test_s, device)
        baseline_runs.append(sh_res)
        print(f"  Single-Head Baseline : Acc={sh_res['accuracy']:.4f}, AUC={sh_res['macro_roc_auc']:.4f}, Mal_Recall={sh_res['mal_recall_tau_020']:.4f}")

        # 2. Multi-Head Model
        mh_model = train_multi_head(
            train_x, train_t, train_s, val_x, val_t, val_s,
            weights_t, weights_s, device, seed=seed, epochs=60,
            triage_weight=1.0, subtype_weight=1.0,
        )
        mh_res = evaluate_multi_head(mh_model, test_x, test_t, test_s, device)
        multi_head_runs.append(mh_res)
        print(f"  Multi-Head Decoupled : Acc={mh_res['accuracy']:.4f}, Subtype_AUC={mh_res['macro_roc_auc']:.4f}, Triage_AUC={mh_res['triage_roc_auc']:.4f}")
        print(f"                         Head 1 Triage Recall={mh_res['triage_recall_tau_020']:.4f}, Spec={mh_res['triage_spec_tau_020']:.4f}")
        print(f"                         Consensus Safety Recall={mh_res['consensus_safety_recall']:.4f}, Agreement={mh_res['consensus_agreement_rate']:.4f}")

    # Aggregating results
    sh_agg = aggregate_metrics(baseline_runs)
    mh_agg = aggregate_metrics(multi_head_runs)

    results_bundle = {
        "SingleHead_Baseline": sh_agg,
        "MultiHead_Decoupled": mh_agg,
        "seeds": SEEDS,
    }

    out_file = PROJECT_ROOT / "artifacts" / "experiments" / "multi_head_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_bundle, f, indent=2)

    print("\n" + "=" * 80)
    print(" FINAL MULTI-HEAD EVALUATION REPORT (Mean ± SD on 346 Test Images)")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Single-Head Baseline':<20} | {'Multi-Head Model':<20} | {'Delta':<12}")
    print("-" * 80)

    comparison_keys = [
        ("accuracy", "Overall Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("macro_f1", "Macro F1-Score"),
        ("macro_roc_auc", "Macro ROC-AUC (Subtypes)"),
        ("mel_recall_tau_005", "Melanoma Recall (tau=0.05)"),
        ("mel_spec_tau_005", "Melanoma Spec (tau=0.05)"),
    ]

    for k, name in comparison_keys:
        sh_m, sh_s = sh_agg[k]["mean"], sh_agg[k]["std"]
        mh_m, mh_s = mh_agg[k]["mean"], mh_agg[k]["std"]
        delta = mh_m - sh_m
        sign = "+" if delta >= 0 else ""
        print(f"{name:<35} | {sh_m*100:6.2f}% ± {sh_s*100:4.2f}%    | {mh_m*100:6.2f}% ± {mh_s*100:4.2f}%    | {sign}{delta*100:5.2f}%")

    print("-" * 80)
    print(" Dedicated Head 1 (Binary Oncology Triage) Metrics:")
    print(f"  Triage ROC-AUC                   : {mh_agg['triage_roc_auc']['mean']:.4f} ± {mh_agg['triage_roc_auc']['std']:.4f}")
    print(f"  Triage PR-AUC                    : {mh_agg['triage_pr_auc']['mean']:.4f} ± {mh_agg['triage_pr_auc']['std']:.4f}")
    print(f"  Triage Sensitivity (tau=0.20)    : {mh_agg['triage_recall_tau_020']['mean']*100:.2f}% ± {mh_agg['triage_recall_tau_020']['std']*100:.2f}%")
    print(f"  Triage Specificity (tau=0.20)    : {mh_agg['triage_spec_tau_020']['mean']*100:.2f}% ± {mh_agg['triage_spec_tau_020']['std']*100:.2f}%")
    print(f"  Triage Sensitivity (tau=0.50)    : {mh_agg['triage_recall_tau_050']['mean']*100:.2f}% ± {mh_agg['triage_recall_tau_050']['std']*100:.2f}%")
    print(f"  Triage Specificity (tau=0.50)    : {mh_agg['triage_spec_tau_050']['mean']*100:.2f}% ± {mh_agg['triage_spec_tau_050']['std']*100:.2f}%")
    print("-" * 80)
    print(" Joint Clinical Consensus & Safety:")
    print(f"  Head 1 & Head 2 Agreement Rate   : {mh_agg['consensus_agreement_rate']['mean']*100:.2f}% ± {mh_agg['consensus_agreement_rate']['std']*100:.2f}%")
    print(f"  Consensus Safety Recall          : {mh_agg['consensus_safety_recall']['mean']*100:.2f}% ± {mh_agg['consensus_safety_recall']['std']*100:.2f}%")
    print(f"  Consensus Safety Specificity     : {mh_agg['consensus_safety_spec']['mean']*100:.2f}% ± {mh_agg['consensus_safety_spec']['std']*100:.2f}%")
    print("=" * 80)
    print(f"[+] Full JSON results saved to: {out_file}")


if __name__ == "__main__":
    main()
