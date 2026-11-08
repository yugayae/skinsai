#!/usr/bin/env python3
"""EXP-010: Train and Validate Multi-Head Multimodal Model on Unified Development Pool.

Dataset: Unified Development Pool (3,597 images, 1,504 patients)
- Train: 2,894 images (1,203 patients, 80.5%)
- Validation: 703 images (301 patients, 19.5%)
- Zero patient and lesion leakage verified.

Architecture:
- Vision Encoder: Meta DINOv2 Small (384-dim, vit_small_patch14_dinov2)
- Clinical Encoder: TabularClinicalEncoder (12-dim -> 64-dim, masked if absent)
- Head 1 (Binary Oncology Triage): Malignant vs Benign (2 logits)
- Head 2 (Differential 6-Class Subtyping): [ack, bcc, mel, nev, scc, sek] (6 logits)
- Joint Clinical Consensus: Defense-in-depth safety arbitration
"""

from __future__ import annotations

from collections import Counter
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

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_ORDER)}
MALIGNANT_INDICES = [CLASS_TO_IDX[c] for c in MALIGNANT_CLASSES]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_clinical_vector(row: pd.Series) -> np.ndarray:
    """Extract standardized 12-dimensional clinical vector from manifest row."""
    age = row.get("age_approx")
    sex = str(row.get("sex", "")).lower()
    site = str(row.get("anatom_site_1", "")).lower()

    vec = []
    # 1. Age normalized
    try:
        age_val = float(age) if age != "" and not pd.isna(age) else None
    except Exception:
        age_val = None

    if age_val is not None:
        vec.append((age_val - 50.0) / 25.0)
    else:
        vec.append(0.0)

    # 2. Site one-hot: (head_neck, trunk, upper, lower, other)
    vec.append(1.0 if ("head" in site or "neck" in site) else 0.0)
    vec.append(1.0 if ("trunk" in site or "torso" in site) else 0.0)
    vec.append(1.0 if ("upper" in site or "arm" in site or "hand" in site) else 0.0)
    vec.append(1.0 if ("lower" in site or "leg" in site or "foot" in site) else 0.0)
    has_known_site = any(vec[-4:])
    vec.append(0.0 if has_known_site else 1.0)

    # 3. Sex one-hot: (female, male, unknown)
    vec.append(1.0 if "female" in sex else 0.0)
    vec.append(1.0 if ("male" in sex and "female" not in sex) else 0.0)
    has_known_sex = vec[-2] == 1.0 or vec[-1] == 1.0
    vec.append(0.0 if has_known_sex else 1.0)

    # 4. Diameter features: (diameter_ge_7mm, diameter_unknown, diameter_norm)
    # Default unknown if not available
    vec.append(0.0)
    vec.append(1.0)  # unknown flag
    vec.append(0.0)

    return np.array(vec, dtype=np.float32)


class ImageOnlyDataset(Dataset):
    def __init__(self, paths: List[Path], transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        with Image.open(p) as img:
            rgb = img.convert("RGB")
        if self.transform:
            return self.transform(rgb)
        return transforms.ToTensor()(rgb)


@torch.no_grad()
def precompute_dinov2_features(
    df: pd.DataFrame,
    cache_path: Path,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    if cache_path.exists():
        print(f"[*] Loading cached DINOv2 embeddings from {cache_path}...")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"[*] Extracting DINOv2 Small features for {len(df)} images on {device}...")
    model = timm.create_model("vit_small_patch14_dinov2", pretrained=True, num_classes=0).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    paths = [PROJECT_ROOT / row["file_path"] for _, row in df.iterrows()]
    dataset = ImageOnlyDataset(paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    embeds = []
    t0 = time.time()
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        feats = model(batch)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeds.append(feats.cpu())
        if (i + 1) % 20 == 0 or (i + 1) == len(loader):
            print(f"    Processed {min((i+1)*batch_size, len(df))}/{len(df)} images ({time.time() - t0:.1f}s)...")

    all_embeds = torch.cat(embeds, dim=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_embeds, cache_path)
    print(f"[+] Saved embeddings to {cache_path} (shape: {all_embeds.shape})")
    return all_embeds


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


def evaluate_multimodal_predictions(
    triage_probs: np.ndarray,
    subtype_probs: np.ndarray,
    y_true_s: np.ndarray,
) -> Dict[str, float]:
    """Evaluate predictions strictly against ground truth 6-class targets."""
    # Filter only samples with valid subtype targets (exclude unmapped ISIC Benign)
    valid_mask = y_true_s >= 0
    s_probs_v = subtype_probs[valid_mask]
    y_true_v = y_true_s[valid_mask]
    t_probs_v = triage_probs[valid_mask]

    y_pred_s = np.argmax(s_probs_v, axis=1)

    acc = float(accuracy_score(y_true_v, y_pred_s))
    bal_acc = float(balanced_accuracy_score(y_true_v, y_pred_s))
    macro_f1 = float(f1_score(y_true_v, y_pred_s, average="macro", zero_division=0))

    try:
        y_onehot = np.eye(6)[y_true_v]
        macro_auc = float(roc_auc_score(y_onehot, s_probs_v, average="macro", multi_class="ovr"))
    except Exception:
        macro_auc = 0.0

    pr_aucs = []
    for c in range(6):
        try:
            p_c, r_c, _ = precision_recall_curve((y_true_v == c).astype(int), s_probs_v[:, c])
            pr_aucs.append(float(auc(r_c, p_c)))
        except Exception:
            pass
    macro_pr_auc = float(np.mean(pr_aucs)) if pr_aucs else 0.0

    # Melanoma metrics at tau=0.05
    mel_idx = CLASS_TO_IDX["mel"]
    mel_mask = (y_true_v == mel_idx)
    mel_support = int(np.sum(mel_mask))
    mel_probs = s_probs_v[:, mel_idx]
    mel_hits = int(np.sum((mel_probs >= 0.05) & mel_mask))
    mel_recall_005 = float(mel_hits / max(mel_support, 1))
    mel_spec_005 = float(np.sum((mel_probs < 0.05) & (~mel_mask)) / max(np.sum(~mel_mask), 1))
    mel_ci_low, mel_ci_high = wilson_score_interval(mel_hits, mel_support)

    # Head 1 Triage Metrics (Malignant vs Benign)
    mal_mask = np.isin(y_true_v, MALIGNANT_INDICES)
    mal_support = int(np.sum(mal_mask))
    p_triage_mal = t_probs_v[:, 1]

    try:
        triage_auc = float(roc_auc_score(mal_mask.astype(int), p_triage_mal))
    except Exception:
        triage_auc = 0.0

    triage_recall_020 = float(np.sum((p_triage_mal >= 0.20) & mal_mask) / max(mal_support, 1))
    triage_spec_020 = float(np.sum((p_triage_mal < 0.20) & (~mal_mask)) / max(np.sum(~mal_mask), 1))

    # Dual-Head Clinical Consensus
    mal_derived = np.sum(s_probs_v[:, MALIGNANT_INDICES], axis=1)
    h1_flag = p_triage_mal >= 0.20
    h2_flag = mal_derived >= 0.20
    consensus_flag = h1_flag | h2_flag
    consensus_recall = float(np.sum(consensus_flag & mal_mask) / max(mal_support, 1))
    consensus_spec = float(np.sum((~consensus_flag) & (~mal_mask)) / max(np.sum(~mal_mask), 1))
    agreement_rate = float(np.mean(h1_flag == h2_flag))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "macro_roc_auc": macro_auc,
        "macro_pr_auc": macro_pr_auc,
        "mel_support": mel_support,
        "mel_hits": mel_hits,
        "mel_recall_tau_005": mel_recall_005,
        "mel_recall_ci_low": mel_ci_low,
        "mel_recall_ci_high": mel_ci_high,
        "mel_spec_tau_005": mel_spec_005,
        "triage_roc_auc": triage_auc,
        "triage_recall_tau_020": triage_recall_020,
        "triage_spec_tau_020": triage_spec_020,
        "consensus_safety_recall": consensus_recall,
        "consensus_safety_spec": consensus_spec,
        "consensus_agreement_rate": agreement_rate,
    }


def main():
    print("=" * 80)
    print(" SkinAI EXP-010: Training Multi-Head Multimodal Model on Unified Development Pool")
    print(" Standard: CLAIM 2024 / STARD-AI Rigor")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    splits_dir = PROJECT_ROOT / "data" / "splits" / "development_pool"
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")

    print(f"[*] Loaded Train records: {len(train_df)}, Val records: {len(val_df)}")

    # 1. Feature Extraction (Vision DINOv2 Small 384D)
    cache_dir = PROJECT_ROOT / "artifacts" / "embeddings"
    train_vision = precompute_dinov2_features(train_df, cache_dir / "dev_pool_train_dinov2.pt", device)
    val_vision = precompute_dinov2_features(val_df, cache_dir / "dev_pool_val_dinov2.pt", device)

    # 2. Feature Extraction (Clinical 12D)
    train_clin = torch.tensor(np.stack([extract_clinical_vector(r) for _, r in train_df.iterrows()]), dtype=torch.float32)
    val_clin = torch.tensor(np.stack([extract_clinical_vector(r) for _, r in val_df.iterrows()]), dtype=torch.float32)

    # 3. Targets
    train_triage = torch.tensor(train_df["triage_target"].values, dtype=torch.long)
    val_triage = torch.tensor(val_df["triage_target"].values, dtype=torch.long)

    def subtype_to_idx(s: str) -> int:
        return CLASS_TO_IDX.get(s, -1)

    train_subtype = torch.tensor([subtype_to_idx(s) for s in train_df["subtype_target"]], dtype=torch.long)
    val_subtype = torch.tensor([subtype_to_idx(s) for s in val_df["subtype_target"]], dtype=torch.long)

    # Compute balanced class weights for 6 classes
    valid_subtypes = train_subtype[train_subtype >= 0].numpy()
    counts = Counter(valid_subtypes)
    num_samples = len(valid_subtypes)
    weights_s = [num_samples / (6.0 * max(counts[i], 1)) for i in range(6)]
    weights_s_tensor = torch.tensor(weights_s, dtype=torch.float32)

    # Binary triage weights
    valid_triage = train_triage[train_triage >= 0].numpy()
    counts_t = Counter(valid_triage)
    weights_t = [len(valid_triage) / (2.0 * max(counts_t[i], 1)) for i in [0, 1]]
    weights_t_tensor = torch.tensor(weights_t, dtype=torch.float32)

    print(f"[*] Subtype class weights (inverse frequency): {[f'{w:.2f}' for w in weights_s]}")
    print(f"[*] Triage class weights [Benign, Malignant]: {[f'{w:.2f}' for w in weights_t]}")

    # 4. Multi-seed Training (5 seeds)
    seeds = [0, 8, 27, 42, 2026]
    seed_results = []
    best_model_probe = None
    best_val_f1 = -1.0

    train_ds = TensorDataset(train_vision, train_clin, train_triage, train_subtype)
    val_ds = TensorDataset(val_vision, val_clin, val_triage, val_subtype)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    print(f"\n[*] Starting training across {len(seeds)} random seeds: {seeds}...")

    for seed in seeds:
        set_seed(seed)
        probe = MultimodalLateFusionProbe(
            vis_dim=384,
            clin_dim=12,
            clin_embed_dim=64,
            hidden_dim=256,
            dropout=0.25,
            num_subtypes=6,
        ).to(device)

        loss_fn = MultiHeadLoss(
            triage_weight=1.0,
            subtype_weight=1.0,
            triage_criterion=MaskedClassWeightedFocalLoss(alpha=weights_t_tensor.to(device), gamma=2.0),
            subtype_criterion=MaskedClassWeightedFocalLoss(alpha=weights_s_tensor.to(device), gamma=2.0),
        )

        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

        epochs = 60
        for ep in range(epochs):
            probe.train()
            for vx, cx, t_tgt, s_tgt in train_loader:
                vx, cx, t_tgt, s_tgt = vx.to(device), cx.to(device), t_tgt.to(device), s_tgt.to(device)
                optimizer.zero_grad()
                t_logits, s_logits = probe(vx, cx)
                loss, _, _ = loss_fn(t_logits, s_logits, t_tgt, s_tgt)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluate on Validation Set
        probe.eval()
        t_probs_list, s_probs_list = [], []
        with torch.no_grad():
            for vx, cx, _, _ in val_loader:
                vx, cx = vx.to(device), cx.to(device)
                t_logits, s_logits = probe(vx, cx)
                t_probs_list.append(torch.softmax(t_logits, dim=1).cpu())
                s_probs_list.append(torch.softmax(s_logits, dim=1).cpu())

        t_probs_all = torch.cat(t_probs_list, dim=0).numpy()
        s_probs_all = torch.cat(s_probs_list, dim=0).numpy()

        res = evaluate_multimodal_predictions(t_probs_all, s_probs_all, val_subtype.numpy())
        res["seed"] = seed
        seed_results.append(res)

        print(
            f"    Seed {seed:>4}: Macro F1={res['macro_f1']:.4f} | Balanced Acc={res['balanced_accuracy']*100:.2f}% | "
            f"ROC-AUC={res['macro_roc_auc']:.4f} | Mel Recall={res['mel_hits']}/{res['mel_support']} ({res['mel_recall_tau_005']*100:.1f}%) | "
            f"Consensus Safety Recall={res['consensus_safety_recall']*100:.2f}% (Spec={res['consensus_safety_spec']*100:.2f}%)"
        )

        if res["macro_f1"] > best_val_f1:
            best_val_f1 = res["macro_f1"]
            best_model_probe = probe

    # 5. Aggregate Results Across Seeds (Aviation & Medical Reporting)
    print("\n" + "=" * 80)
    print(" EXP-010 VALIDATION BENCHMARK RESULTS (Aviation & Medical Standards)")
    print("=" * 80)

    keys = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_roc_auc",
        "macro_pr_auc",
        "mel_recall_tau_005",
        "mel_spec_tau_005",
        "triage_roc_auc",
        "triage_recall_tau_020",
        "triage_spec_tau_020",
        "consensus_safety_recall",
        "consensus_safety_spec",
        "consensus_agreement_rate",
    ]

    summary = {}
    for k in keys:
        vals = [r[k] for r in seed_results]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        min_v = float(np.min(vals))
        max_v = float(np.max(vals))
        summary[k] = {"mean": mean_v, "std": std_v, "min": min_v, "max": max_v}
        if "auc" in k or "f1" in k:
            print(f" {k:<25}: {mean_v:.4f} ± {std_v:.4f} (range: [{min_v:.4f}, {max_v:.4f}])")
        else:
            print(f" {k:<25}: {mean_v*100:.2f}% ± {std_v*100:.2f}% (range: [{min_v*100:.2f}%, {max_v*100:.2f}%])")

    mel_hits_mean = float(np.mean([r["mel_hits"] for r in seed_results]))
    mel_total = seed_results[0]["mel_support"]
    print(f"\n Melanoma Recall at tau=0.05: {mel_hits_mean:.1f}/{mel_total} ({summary['mel_recall_tau_005']['mean']*100:.1f}%)")
    w_low, w_high = wilson_score_interval(int(round(mel_hits_mean)), mel_total)
    print(f" Wilson 95% Confidence Interval for Melanoma: [{w_low*100:.1f}%, {w_high*100:.1f}%]")

    # 6. Save Champion Checkpoint and Benchmark Logs
    out_model_dir = PROJECT_ROOT / "artifacts" / "models" / "smartphone_clinical" / "development_champion"
    out_model_dir.mkdir(parents=True, exist_ok=True)
    if best_model_probe:
        torch.save(best_model_probe.state_dict(), out_model_dir / "best_model.pt")
        print(f"\n[+] Champion model weights saved to: {out_model_dir / 'best_model.pt'}")

    out_artifact = PROJECT_ROOT / "artifacts" / "experiments" / "exp010_development_pool_benchmark.json"
    out_artifact.parent.mkdir(parents=True, exist_ok=True)
    with open(out_artifact, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "seeds": seed_results}, f, indent=2, ensure_ascii=False)
    print(f"[+] Detailed experimental report saved to: {out_artifact}")
    print("=" * 80)


if __name__ == "__main__":
    main()
