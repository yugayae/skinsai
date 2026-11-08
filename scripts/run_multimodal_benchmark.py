"""EXP-007 Scientific Benchmark: Multimodal Late Fusion (Vision + Clinical Metadata).

Compares:
1. Condition A: Vision-Only DINOv2 Small (Baseline)
2. Condition B: Multimodal DINOv2 Small (Vision + Clinical Questionnaire Late Fusion)

Evaluated across 5 random seeds: [0, 8, 27, 42, 2026] on frozen test split (346 images).
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    auc,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
from ml.models.multimodal import MultimodalLateFusionProbe

SEEDS = [0, 8, 27, 42, 2026]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_ORDER)}
MALIGNANT_INDICES = [CLASS_TO_IDX[c] for c in MALIGNANT_CLASSES]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_clinical_vector(
    img_id: str,
    pad_meta: dict,
    isic_meta: dict,
    is_isic: bool = False,
) -> np.ndarray:
    """Extract standardized 12-dimensional clinical questionnaire vector.

    Features:
    [0] age_normalized (continuous)
    [1..5] anatom_site one-hot: (head_neck, trunk, upper, lower, other)
    [6..8] sex one-hot: (female, male, unknown)
    [9..11] diameter features: (diameter_ge_7mm, diameter_unknown, diameter_norm)
    """
    vec = []
    if is_isic and img_id in isic_meta:
        row = isic_meta[img_id]
        age = row.get("age_approx")
        sex = str(row.get("sex", "")).lower()
        site = str(row.get("anatom_site_1", "")).lower()
        diam = np.nan
    elif img_id in pad_meta:
        clin = pad_meta[img_id]
        age = clin.get("age_approx")
        sex = str(clin.get("sex", "")).lower()
        site = str(clin.get("anatom_site_1", "")).lower()
        diam = clin.get("clin_size_long_diam_mm")
        try:
            diam = float(diam) if diam is not None else np.nan
        except Exception:
            diam = np.nan
    else:
        age, sex, site, diam = None, "", "", np.nan

    # 1. Age normalized
    if age is not None and not (isinstance(age, float) and np.isnan(age)):
        vec.append((float(age) - 50.0) / 25.0)
    else:
        vec.append(0.0)

    # 2. Site one-hot
    vec.append(1.0 if "head" in site or "neck" in site else 0.0)
    vec.append(1.0 if "trunk" in site or "torso" in site else 0.0)
    vec.append(1.0 if "upper" in site or "arm" in site or "hand" in site else 0.0)
    vec.append(1.0 if "lower" in site or "leg" in site or "foot" in site else 0.0)
    has_known_site = any(vec[-4:])
    vec.append(0.0 if has_known_site else 1.0)

    # 3. Sex one-hot
    vec.append(1.0 if "female" in sex else 0.0)
    vec.append(1.0 if "male" in sex and "female" not in sex else 0.0)
    has_known_sex = vec[-2] == 1.0 or vec[-1] == 1.0
    vec.append(0.0 if has_known_sex else 1.0)

    # 4. Diameter
    if not np.isnan(diam):
        vec.append(1.0 if diam >= 7.0 else 0.0)
        vec.append(0.0)  # known
        vec.append(min((diam - 10.0) / 10.0, 3.0))
    else:
        vec.append(0.0)
        vec.append(1.0)  # unknown
        vec.append(0.0)

    return np.array(vec, dtype=np.float32)


def evaluate_predictions(
    t_probs: np.ndarray,
    s_probs: np.ndarray,
    y_true_s: np.ndarray,
) -> Dict[str, float]:
    y_pred_s = np.argmax(s_probs, axis=1)

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

    # Triage metrics
    mal_mask = np.isin(y_true_s, MALIGNANT_INDICES)
    mal_support = int(np.sum(mal_mask))
    mal_derived_probs = np.sum(s_probs[:, MALIGNANT_INDICES], axis=1)

    p_triage_mal = t_probs[:, 1]
    try:
        triage_roc_auc = float(roc_auc_score(mal_mask.astype(int), p_triage_mal))
    except Exception:
        triage_roc_auc = 0.0

    triage_recall_020 = float(np.sum((p_triage_mal >= 0.20) & mal_mask) / max(mal_support, 1))
    triage_spec_020 = float(np.sum((p_triage_mal < 0.20) & (~mal_mask)) / max(np.sum(~mal_mask), 1))

    # Clinical consensus
    head1_referral = (p_triage_mal >= 0.20)
    head2_referral = (mal_derived_probs >= 0.20)
    agreement_rate = float(np.mean(head1_referral == head2_referral))
    consensus_referral = head1_referral | head2_referral
    consensus_recall = float(np.sum(consensus_referral & mal_mask) / max(mal_support, 1))
    consensus_spec = float(np.sum((~consensus_referral) & (~mal_mask)) / max(np.sum(~mal_mask), 1))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "macro_roc_auc": macro_auc,
        "macro_pr_auc": macro_pr_auc,
        "mel_recall_tau_005": mel_recall_005,
        "mel_spec_tau_005": mel_spec_005,
        "triage_roc_auc": triage_roc_auc,
        "triage_recall_tau_020": triage_recall_020,
        "triage_spec_tau_020": triage_spec_020,
        "consensus_agreement_rate": agreement_rate,
        "consensus_safety_recall": consensus_recall,
        "consensus_safety_spec": consensus_spec,
    }


def train_vision_only(
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
) -> MultiHeadProbe:
    set_seed(seed)
    probe = MultiHeadProbe(in_features=train_x.shape[1], num_subtypes=6).to(device)

    loss_fn = MultiHeadLoss(
        triage_weight=1.0,
        subtype_weight=1.0,
        triage_criterion=MaskedClassWeightedFocalLoss(alpha=weights_t.to(device), gamma=2.0),
        subtype_criterion=MaskedClassWeightedFocalLoss(alpha=weights_s.to(device), gamma=2.0),
    )

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_ds = TensorDataset(train_x, train_t, train_s)
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
            v_val = val_loss.item()

        if v_val < best_val_loss:
            best_val_loss = v_val
            best_weights = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    if best_weights:
        probe.load_state_dict(best_weights)

    probe.calibrate_temperatures(
        val_features=val_x.to(device),
        val_triage_targets=val_t.to(device),
        val_subtype_targets=val_s.to(device),
    )
    return probe


def train_multimodal(
    train_vis: torch.Tensor,
    train_clin: torch.Tensor,
    train_t: torch.Tensor,
    train_s: torch.Tensor,
    val_vis: torch.Tensor,
    val_clin: torch.Tensor,
    val_t: torch.Tensor,
    val_s: torch.Tensor,
    weights_t: torch.Tensor,
    weights_s: torch.Tensor,
    device: torch.device,
    seed: int,
    epochs: int = 60,
    lr: float = 1e-3,
) -> MultimodalLateFusionProbe:
    set_seed(seed)
    probe = MultimodalLateFusionProbe(
        vis_dim=train_vis.shape[1],
        clin_dim=train_clin.shape[1],
        clin_embed_dim=64,
        hidden_dim=256,
        num_subtypes=6,
    ).to(device)

    loss_fn = MultiHeadLoss(
        triage_weight=1.0,
        subtype_weight=1.0,
        triage_criterion=MaskedClassWeightedFocalLoss(alpha=weights_t.to(device), gamma=2.0),
        subtype_criterion=MaskedClassWeightedFocalLoss(alpha=weights_s.to(device), gamma=2.0),
    )

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_ds = TensorDataset(train_vis, train_clin, train_t, train_s)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_val_loss = float("inf")
    best_weights = None

    for _ in range(epochs):
        probe.train()
        for b_vis, b_clin, bt, bs in train_loader:
            b_vis, b_clin, bt, bs = b_vis.to(device), b_clin.to(device), bt.to(device), bs.to(device)
            optimizer.zero_grad()
            t_logits, s_logits = probe(b_vis, b_clin)
            loss, _, _ = loss_fn(t_logits, s_logits, bt, bs)
            loss.backward()
            optimizer.step()
        scheduler.step()

        probe.eval()
        with torch.no_grad():
            vt_logits, vs_logits = probe(val_vis.to(device), val_clin.to(device))
            val_loss, _, _ = loss_fn(vt_logits, vs_logits, val_t.to(device), val_s.to(device))
            v_val = val_loss.item()

        if v_val < best_val_loss:
            best_val_loss = v_val
            best_weights = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    if best_weights:
        probe.load_state_dict(best_weights)

    probe.calibrate_temperatures(
        val_vis=val_vis.to(device),
        val_clin=val_clin.to(device),
        val_triage_targets=val_t.to(device),
        val_subtype_targets=val_s.to(device),
    )
    return probe


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
    print(" EXP-007 Scientific Benchmark: Multimodal Late Fusion across 5 Random Seeds")
    print(f" Seeds: {SEEDS}")
    print(" Comparing: [Condition A: Vision-Only DINOv2] vs [Condition B: Multimodal DINOv2]")
    print(" Test Set : Frozen 346 Clinical Images (205 Patients)")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device : {device} ({torch.cuda.get_device_name(0)})")

    split_dir = PROJECT_ROOT / "data" / "splits" / "smartphone_clinical"
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    # Load PAD-UFES clinical metadata
    with open(PROJECT_ROOT / "data" / "metadata" / "pad_ufes_20.json", "r") as f:
        pad_records = json.load(f)
    pad_meta = {item["isic_id"]: item.get("metadata", {}).get("clinical", {}) for item in pad_records}

    # Load ISIC manifest
    isic_manifest = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "isic_clinical_manifest.csv")
    new_isic_df = isic_manifest[~isic_manifest["is_existing_pad_ufes"]].copy()
    isic_meta = isic_manifest.set_index("image_id").to_dict(orient="index")

    # Load cached DINOv2 visual features
    cache_path = PROJECT_ROOT / "artifacts" / "cache" / "multi_head_dinov2_features.pt"
    if not cache_path.exists():
        print(f"[!] Error: Visual feature cache not found at {cache_path}")
        return

    print(f"[*] Loading visual features from {cache_path}...")
    cache = torch.load(cache_path, weights_only=False)
    train_vis, train_t, train_s = cache["train_x"], cache["train_t"], cache["train_s"]
    val_vis, val_t, val_s = cache["val_x"], cache["val_t"], cache["val_s"]
    test_vis, test_t, test_s = cache["test_x"], cache["test_t"], cache["test_s"]

    # Build aligned clinical vectors
    train_ids = [str(r["image_id"]) for _, r in train_df.iterrows()]
    train_ids += [str(r["image_id"]) for _, r in new_isic_df.iterrows() if r["image_id"] in isic_meta]
    # Filter to match train_vis length (2897)
    train_ids = train_ids[:len(train_vis)]

    val_ids = [str(r["image_id"]) for _, r in val_df.iterrows()]
    test_ids = [str(r["image_id"]) for _, r in test_df.iterrows()]

    print("[*] Extracting 12-dimensional clinical vectors...")
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

    print(f"[*] Train Tensors -> Vision: {train_vis.shape}, Clinical: {train_clin.shape}")
    print(f"[*] Val Tensors   -> Vision: {val_vis.shape}, Clinical: {val_clin.shape}")
    print(f"[*] Test Tensors  -> Vision: {test_vis.shape}, Clinical: {test_clin.shape}")

    # Compute class weights
    valid_s = train_s[train_s >= 0]
    counts_s = torch.bincount(valid_s, minlength=6).float()
    weights_s = 1.0 / torch.sqrt(torch.clamp(counts_s, min=1.0))
    weights_s = weights_s / weights_s.mean()

    valid_t = train_t[train_t >= 0]
    counts_t = torch.bincount(valid_t, minlength=2).float()
    weights_t = 1.0 / torch.sqrt(torch.clamp(counts_t, min=1.0))
    weights_t = weights_t / weights_t.mean()

    vision_runs = []
    multimodal_runs = []

    print("\n" + "=" * 80)
    print(" Running Benchmarks across 5 Seeds...")
    print("=" * 80)

    for seed in SEEDS:
        print(f"\n--- [Seed {seed}] ---")
        # 1. Vision-Only Model
        vis_model = train_vision_only(
            train_vis, train_t, train_s, val_vis, val_t, val_s,
            weights_t, weights_s, device, seed=seed, epochs=60
        )
        vis_model.eval()
        with torch.no_grad():
            v_t_probs, v_s_probs = vis_model.predict_proba(test_vis.to(device))
            v_res = evaluate_predictions(v_t_probs.cpu().numpy(), v_s_probs.cpu().numpy(), test_s.numpy())
        vision_runs.append(v_res)
        print(f"  Condition A (Vision-Only) : Acc={v_res['accuracy']*100:.2f}%, AUC={v_res['macro_roc_auc']:.4f}, Mel_Recall={v_res['mel_recall_tau_005']*100:.1f}%")

        # 2. Multimodal Late Fusion Model
        mm_model = train_multimodal(
            train_vis, train_clin, train_t, train_s, val_vis, val_clin, val_t, val_s,
            weights_t, weights_s, device, seed=seed, epochs=60
        )
        mm_model.eval()
        with torch.no_grad():
            m_t_probs, m_s_probs = mm_model.predict_proba(test_vis.to(device), test_clin.to(device))
            m_res = evaluate_predictions(m_t_probs.cpu().numpy(), m_s_probs.cpu().numpy(), test_s.numpy())
        multimodal_runs.append(m_res)
        print(f"  Condition B (Multimodal)  : Acc={m_res['accuracy']*100:.2f}%, AUC={m_res['macro_roc_auc']:.4f}, Mel_Recall={m_res['mel_recall_tau_005']*100:.1f}%")
        print(f"                              Triage_AUC={m_res['triage_roc_auc']:.4f}, Consensus_Recall={m_res['consensus_safety_recall']*100:.2f}%")

    vis_agg = aggregate_metrics(vision_runs)
    mm_agg = aggregate_metrics(multimodal_runs)

    results_bundle = {
        "Condition_A_VisionOnly": vis_agg,
        "Condition_B_Multimodal": mm_agg,
        "seeds": SEEDS,
    }

    out_file = PROJECT_ROOT / "artifacts" / "experiments" / "multimodal_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_bundle, f, indent=2)

    print("\n" + "=" * 80)
    print(" FINAL MULTIMODAL EVALUATION REPORT (Mean ± SD on 346 Test Images)")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Vision-Only':<20} | {'Multimodal Fusion':<20} | {'Delta':<12}")
    print("-" * 80)

    comparison_keys = [
        ("accuracy", "Overall Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("macro_f1", "Macro F1-Score"),
        ("macro_roc_auc", "Macro ROC-AUC (Subtypes)"),
        ("macro_pr_auc", "Macro PR-AUC"),
        ("mel_recall_tau_005", "Melanoma Recall (tau=0.05)"),
        ("mel_spec_tau_005", "Melanoma Spec (tau=0.05)"),
        ("triage_roc_auc", "Head 1 Triage ROC-AUC"),
        ("triage_recall_tau_020", "Head 1 Triage Recall (0.20)"),
        ("triage_spec_tau_020", "Head 1 Triage Spec (0.20)"),
        ("consensus_safety_recall", "Consensus Safety Recall"),
        ("consensus_safety_spec", "Consensus Safety Spec"),
    ]

    for k, name in comparison_keys:
        v_m, v_s = vis_agg[k]["mean"], vis_agg[k]["std"]
        m_m, m_s = mm_agg[k]["mean"], mm_agg[k]["std"]
        delta = m_m - v_m
        sign = "+" if delta >= 0 else ""
        print(f"{name:<35} | {v_m*100:6.2f}% ± {v_s*100:4.2f}%    | {m_m*100:6.2f}% ± {m_s*100:4.2f}%    | {sign}{delta*100:5.2f}%")

    print("=" * 80)
    print(f"[+] Full JSON results saved to: {out_file}")


if __name__ == "__main__":
    main()
