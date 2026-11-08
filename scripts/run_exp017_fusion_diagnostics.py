#!/usr/bin/env python3
"""EXP-017: Technical Review P0 Diagnostics, De-Confounding & External DDI Audit.

Executes the complete P0 verification protocol formulated by the external Technical Audit
(Independent Technical Audit) on SkinAI:

1. P0-01 & P0-02 (Data Integrity & Exact Per-Source Breakdown):
   - Exact image, patient, lesion, and class counts per source (`pad_ufes_20`, `med_node`, `isic_clinical`).
   - Patient/lesion zero-overlap verification + cross-dataset embedding cosine duplicate audit.
2. P0-01 & P0-03 (Source-Probe & Metadata-Only Shortcut Audit):
   - Linear Source-Probe on Clinical 12D, DINOv2 384D, and BiomedCLIP 512D embeddings.
   - Metadata-Only Classifier (zero visual features) to quantify exact shortcut ceiling.
3. P0-01 & P0-03 (Image-Only vs. De-Confounded Multimodal vs. Original Multimodal):
   - Evaluates across 5 seeds with 95% Wilson CIs and Patient-Clustered Bootstrap CIs:
     * Globally on `val.csv` (757 images, 591 patients)
     * Within `PAD-UFES-20 Only` (pure smartphone subset)
     * Within `MED-NODE Only` (macroscopic mel vs nev subset)
     * Within `ISIC Close-up Only`
4. Experiment E (Leave-One-Source-Out & Pure-PAD Training):
   - Train on `PAD-UFES-20 + ISIC` (0% MED-NODE) -> zero-shot test on all 170 `MED-NODE` images.
   - Train on `PAD-UFES-20 Only` -> test on `PAD-UFES-20 val`, `MED-NODE`, and `Stanford DDI`.
5. P0-04 (5-Fold StratifiedGroupKFold Out-of-Fold Evaluation across all 3,767 images / 142 melanomas):
   - Full 5-fold patient-grouped cross-validation for both Image-Only Fusion and Multimodal Fusion.
6. P0-05 (Stanford DDI 656-Image External Evaluation of EXP-015 Fusion):
   - Tests the primary root-cause hypothesis (`unknown` metadata flag shift vs. Image-Only Fusion).
   - Evaluates label-free Saerens EM Prior Shift Adaptation, Three-Way Decision Triage (Green/Grey/Red),
     and Split-Calibration Conformal Triage without Oracle test snooping.
   - Separates In-Taxonomy (301 images) vs. Open-Set (355 images) and FST I-II / III-IV / V-VI.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import open_clip
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.multi_head import (
    CLASS_ORDER,
    MALIGNANT_CLASSES,
    MaskedClassWeightedFocalLoss,
    MultiHeadLoss,
)
from scripts.run_exp013_architecture_tournament import (
    CLASS_TO_IDX,
    MALIGNANT_INDICES,
    SEEDS,
    extract_clinical_vector,
    set_seed,
    wilson_score_interval,
)
from scripts.run_exp014_optuna_and_hyperparam_proofs import (
    ConfigurableMultimodalProbe,
    compute_powered_weights,
    get_activation,
)


class PureImageProbe(nn.Module):
    """Pure Image-Only Dual-Head Classifier (Zero Clinical Metadata / Zero Shortcut Risk)."""

    def __init__(
        self,
        vis_dim: int = 384,
        hidden_dim: int = 128,
        num_subtypes: int = 6,
        dropout: float = 0.15,
        activation: str = "silu",
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Dropout(dropout),
            nn.Linear(vis_dim, hidden_dim),
            get_activation(activation),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.triage_head = nn.Linear(hidden_dim, 2)
        self.subtype_head = nn.Linear(hidden_dim, num_subtypes)

    def forward(self, vis_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(vis_feat)
        return self.triage_head(h), self.subtype_head(h)


def extract_deconfounded_clinical_vector(row: pd.Series) -> np.ndarray:
    """Extract 7D clinical vector WITHOUT `site_unknown`, `sex_unknown`, or dead diameter constants."""
    age = row.get("age_approx")
    sex = str(row.get("sex", "")).lower()
    site = str(row.get("anatom_site_1", "")).lower()

    try:
        age_val = float(age) if age != "" and not pd.isna(age) else None
    except Exception:
        age_val = None

    age_norm = (age_val - 50.0) / 25.0 if age_val is not None else 0.0
    s_head = 1.0 if ("head" in site or "neck" in site) else 0.0
    s_trunk = 1.0 if ("trunk" in site or "torso" in site) else 0.0
    s_upper = 1.0 if ("upper" in site or "arm" in site or "hand" in site) else 0.0
    s_lower = 1.0 if ("lower" in site or "leg" in site or "foot" in site) else 0.0
    sex_f = 1.0 if "female" in sex else 0.0
    sex_m = 1.0 if ("male" in sex and "female" not in sex) else 0.0

    return np.array([age_norm, s_head, s_trunk, s_upper, s_lower, sex_f, sex_m], dtype=np.float32)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) on binary probabilities."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if np.any(mask):
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def patient_clustered_bootstrap_ci(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int = 400,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute 95% confidence interval via patient-clustered bootstrap resampling."""
    rng = np.random.default_rng(seed)
    unique_pids, inverse_idx = np.unique(patient_ids, return_inverse=True)
    n_patients = len(unique_pids)
    if n_patients < 5:
        return 0.0, 0.0

    # Group indices by patient
    pid_to_indices = [[] for _ in range(n_patients)]
    for idx, p_idx in enumerate(inverse_idx):
        pid_to_indices[p_idx].append(idx)
    pid_to_indices = [np.array(ix, dtype=np.int64) for ix in pid_to_indices]

    stats = []
    for _ in range(n_boot):
        sampled_pids = rng.integers(0, n_patients, size=n_patients)
        sampled_indices = np.concatenate([pid_to_indices[p] for p in sampled_pids])
        yt = y_true[sampled_indices]
        ys = y_score[sampled_indices]
        try:
            val = metric_fn(yt, ys)
            if not np.isnan(val):
                stats.append(float(val))
        except Exception:
            continue

    if not stats:
        return 0.0, 0.0
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def compute_detailed_subset_metrics(
    t_probs: np.ndarray,
    s_probs: np.ndarray,
    t_targets: np.ndarray,
    s_targets: np.ndarray,
    patient_ids: np.ndarray | None = None,
    tau_mal: float = 0.20,
    tau_mel: float = 0.05,
    compute_bootstrap: bool = False,
) -> Dict[str, Any]:
    """Compute comprehensive metrics (6-class, binary triage, per-class recall, TP/FN/N, Wilson CI)."""
    y_bin = (t_targets == 1).astype(np.int64)
    n_total = len(y_bin)
    n_mal = int(np.sum(y_bin == 1))
    n_ben = int(np.sum(y_bin == 0))

    p_triage = t_probs[:, 1]
    s_derived_mal = np.sum(s_probs[:, MALIGNANT_INDICES], axis=1)
    p_consensus = 0.5 * p_triage + 0.5 * s_derived_mal

    if len(set(y_bin)) > 1:
        triage_auc = float(roc_auc_score(y_bin, p_triage))
        consensus_auc = float(roc_auc_score(y_bin, p_consensus))
        brier = float(brier_score_loss(y_bin, p_triage))
        ece = expected_calibration_error(y_bin, p_triage)
    else:
        triage_auc = 0.0
        consensus_auc = 0.0
        brier = 0.0
        ece = 0.0

    # Also compute strict Head-1 Triage ROC-AUC on rows where triage_target >= 0 (excluding masked ACK=-1)
    strict_t_mask = t_targets >= 0
    if np.sum(strict_t_mask) > 0 and len(set(t_targets[strict_t_mask])) > 1:
        triage_auc_strict = float(roc_auc_score(t_targets[strict_t_mask], p_triage[strict_t_mask]))
    else:
        triage_auc_strict = triage_auc

    h1_flag = p_triage >= tau_mal
    h2_flag = s_derived_mal >= tau_mal
    cons_flag = h1_flag | h2_flag

    tp_h1 = int(np.sum(h1_flag & (y_bin == 1)))
    tn_h1 = int(np.sum((~h1_flag) & (y_bin == 0)))
    tp_cons = int(np.sum(cons_flag & (y_bin == 1)))
    tn_cons = int(np.sum((~cons_flag) & (y_bin == 0)))

    triage_sens = float(tp_h1 / max(n_mal, 1))
    triage_spec = float(tn_h1 / max(n_ben, 1))
    cons_sens = float(tp_cons / max(n_mal, 1))
    cons_spec = float(tn_cons / max(n_ben, 1))

    # Subtype metrics on valid 6-class rows
    valid_mask = s_targets >= 0
    n_valid_sub = int(np.sum(valid_mask))
    res: Dict[str, Any] = {
        "n_images": n_total,
        "n_patients": int(len(set(patient_ids))) if patient_ids is not None else n_total,
        "n_malignant": n_mal,
        "n_benign": n_ben,
        "n_labeled_subtypes": n_valid_sub,
        "triage_roc_auc": triage_auc,
        "triage_roc_auc_no_ack": triage_auc_strict,
        "consensus_roc_auc": consensus_auc,
        "brier_score": brier,
        "ece": ece,
        "triage_sens": triage_sens,
        "triage_sens_wilson_95": list(wilson_score_interval(tp_h1, n_mal)),
        "triage_spec": triage_spec,
        "triage_spec_wilson_95": list(wilson_score_interval(tn_h1, n_ben)),
        "consensus_sens": cons_sens,
        "consensus_spec": cons_spec,
    }

    if n_valid_sub > 0:
        st = s_targets[valid_mask]
        sp = s_probs[valid_mask]
        sp_pred = np.argmax(sp, axis=1)

        present_classes = sorted(list(set(st)))
        macro_f1 = float(f1_score(st, sp_pred, labels=present_classes, average="macro", zero_division=0))
        bal_acc = float(balanced_accuracy_score(st, sp_pred))

        per_class_f1 = {}
        per_class_recall = {}
        per_class_counts = {}
        for idx, cname in enumerate(CLASS_ORDER):
            c_mask = st == idx
            n_c = int(np.sum(c_mask))
            per_class_counts[cname] = n_c
            if n_c > 0:
                tp_c = int(np.sum((sp_pred == idx) & c_mask))
                per_class_recall[cname] = {
                    "recall": float(tp_c / n_c),
                    "tp": tp_c,
                    "fn": n_c - tp_c,
                    "n": n_c,
                    "wilson_95": list(wilson_score_interval(tp_c, n_c)),
                }
                c_f1 = float(f1_score(st == idx, sp_pred == idx, pos_label=True, zero_division=0))
                per_class_f1[cname] = c_f1

        # Also check thresholded melanoma recall (p_mel >= tau_mel)
        mel_idx = CLASS_TO_IDX["mel"]
        mel_mask = st == mel_idx
        n_mel = int(np.sum(mel_mask))
        if n_mel > 0:
            tp_mel_tau = int(np.sum((sp[:, mel_idx] >= tau_mel) & mel_mask))
            mel_tau_recall = float(tp_mel_tau / n_mel)
            mel_tau_ci = list(wilson_score_interval(tp_mel_tau, n_mel))
        else:
            tp_mel_tau = 0
            mel_tau_recall = 0.0
            mel_tau_ci = [0.0, 0.0]

        res.update(
            {
                "present_classes": [CLASS_ORDER[i] for i in present_classes],
                "macro_f1_present": macro_f1,
                "balanced_acc": bal_acc,
                "per_class_f1": per_class_f1,
                "per_class_recall_argmax": per_class_recall,
                "mel_recall_tau005": {
                    "recall": mel_tau_recall,
                    "tp": tp_mel_tau,
                    "fn": n_mel - tp_mel_tau,
                    "n": n_mel,
                    "wilson_95": mel_tau_ci,
                },
            }
        )

        if compute_bootstrap and patient_ids is not None:
            pids_sub = patient_ids[valid_mask]
            f1_ci = patient_clustered_bootstrap_ci(
                pids_sub,
                st,
                sp_pred,
                lambda yt, yp: f1_score(yt, yp, labels=present_classes, average="macro", zero_division=0),
            )
            res["macro_f1_bootstrap_95_ci"] = list(f1_ci)
            if len(set(y_bin)) > 1:
                auc_ci = patient_clustered_bootstrap_ci(
                    patient_ids,
                    y_bin,
                    p_triage,
                    roc_auc_score,
                )
                res["triage_auc_bootstrap_95_ci"] = list(auc_ci)

    return res


def train_and_predict_image_only(
    tr_vis: torch.Tensor,
    tr_t: torch.Tensor,
    tr_s: torch.Tensor,
    ev_vis_list: List[torch.Tensor],
    params: Dict[str, Any],
    seed: int,
    device: torch.device,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Train a pure Image-Only specialist and return (t_probs, s_probs) for each evaluation tensor in ev_vis_list."""
    set_seed(seed)
    vis_dim = tr_vis.shape[1]
    weights_s = compute_powered_weights(tr_s.numpy(), 6, params["class_weight_power"], device)
    weights_t = compute_powered_weights(tr_t.numpy(), 2, params["class_weight_power"], device)

    probe = PureImageProbe(
        vis_dim=vis_dim,
        hidden_dim=params["hidden_dim"],
        num_subtypes=6,
        dropout=params["dropout"],
        activation=params["activation"],
    ).to(device)

    loss_fn = MultiHeadLoss(
        triage_weight=1.0,
        subtype_weight=1.0,
        triage_criterion=MaskedClassWeightedFocalLoss(alpha=weights_t, gamma=params["gamma"]),
        subtype_criterion=MaskedClassWeightedFocalLoss(alpha=weights_s, gamma=params["gamma"]),
    )
    optimizer = torch.optim.AdamW(probe.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

    tr_vis_d = tr_vis.to(device)
    tr_t_d = tr_t.to(device)
    tr_s_d = tr_s.to(device)
    n_samples = tr_vis_d.shape[0]
    bs = params["batch_size"]

    for _ in range(60):
        probe.train()
        perm = torch.randperm(n_samples, device=device)
        for i in range(0, n_samples, bs):
            idx = perm[i : i + bs]
            if idx.numel() < 2:
                continue
            vx, tt, st = tr_vis_d[idx], tr_t_d[idx], tr_s_d[idx]
            optimizer.zero_grad()
            tl, sl = probe(vx)
            loss, _, _ = loss_fn(tl, sl, tt, st)
            loss.backward()
            optimizer.step()
        scheduler.step()

    probe.eval()
    outputs = []
    with torch.no_grad():
        for ev_vis in ev_vis_list:
            tl, sl = probe(ev_vis.to(device))
            tp = torch.softmax(tl, dim=1).cpu().numpy()
            sp = torch.softmax(sl, dim=1).cpu().numpy()
            outputs.append((tp, sp))
    return outputs


def train_and_predict_multimodal(
    tr_vis: torch.Tensor,
    tr_clin: torch.Tensor,
    tr_t: torch.Tensor,
    tr_s: torch.Tensor,
    ev_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    params: Dict[str, Any],
    seed: int,
    device: torch.device,
    metadata_dropout_prob: float = 0.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Train Multimodal specialist (with optional metadata dropout) and return predictions for ev_pairs."""
    set_seed(seed)
    vis_dim = tr_vis.shape[1]
    clin_dim = tr_clin.shape[1]
    weights_s = compute_powered_weights(tr_s.numpy(), 6, params["class_weight_power"], device)
    weights_t = compute_powered_weights(tr_t.numpy(), 2, params["class_weight_power"], device)

    probe = ConfigurableMultimodalProbe(
        vis_dim=vis_dim,
        clin_dim=clin_dim,
        clin_embed_dim=params["clin_embed_dim"],
        hidden_dim=params["hidden_dim"],
        num_subtypes=6,
        dropout=params["dropout"],
        activation=params["activation"],
    ).to(device)

    loss_fn = MultiHeadLoss(
        triage_weight=1.0,
        subtype_weight=1.0,
        triage_criterion=MaskedClassWeightedFocalLoss(alpha=weights_t, gamma=params["gamma"]),
        subtype_criterion=MaskedClassWeightedFocalLoss(alpha=weights_s, gamma=params["gamma"]),
    )
    optimizer = torch.optim.AdamW(probe.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

    tr_vis_d = tr_vis.to(device)
    tr_clin_d = tr_clin.to(device)
    tr_t_d = tr_t.to(device)
    tr_s_d = tr_s.to(device)
    n_samples = tr_vis_d.shape[0]
    bs = params["batch_size"]

    for _ in range(60):
        probe.train()
        perm = torch.randperm(n_samples, device=device)
        for i in range(0, n_samples, bs):
            idx = perm[i : i + bs]
            if idx.numel() < 2:
                continue
            vx, cx, tt, st = tr_vis_d[idx], tr_clin_d[idx], tr_t_d[idx], tr_s_d[idx]
            if metadata_dropout_prob > 0.0:
                keep_mask = (torch.rand((cx.shape[0], 1), device=device) >= metadata_dropout_prob).float()
                cx = cx * keep_mask
            optimizer.zero_grad()
            tl, sl = probe(vx, cx)
            loss, _, _ = loss_fn(tl, sl, tt, st)
            loss.backward()
            optimizer.step()
        scheduler.step()

    probe.eval()
    outputs = []
    with torch.no_grad():
        for ev_vis, ev_clin in ev_pairs:
            tl, sl = probe(ev_vis.to(device), ev_clin.to(device))
            tp = torch.softmax(tl, dim=1).cpu().numpy()
            sp = torch.softmax(sl, dim=1).cpu().numpy()
            outputs.append((tp, sp))
    return outputs


def saerens_em_prior_shift(
    p_source: np.ndarray,
    pi_source: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Saerens et al. (2002) EM algorithm for Unsupervised Prior Probability Shift Adaptation.

    Estimates target malignancy prevalence `pi_target` and adjusted posterior probabilities
     strictly from unlabelled target predictions `p_source` (zero target labels used!).
    """
    pi_s = float(np.clip(pi_source, 1e-4, 1.0 - 1e-4))
    pi_t = pi_s
    p_s = np.clip(p_source, 1e-6, 1.0 - 1e-6)

    for _ in range(max_iter):
        w1 = (pi_t / pi_s) * p_s
        w0 = ((1.0 - pi_t) / (1.0 - pi_s)) * (1.0 - p_s)
        denom = np.clip(w1 + w0, 1e-12, None)
        p_adj = w1 / denom
        new_pi_t = float(np.mean(p_adj))
        if abs(new_pi_t - pi_t) < tol:
            pi_t = new_pi_t
            break
        pi_t = new_pi_t

    return p_adj, pi_t


class SimplePilDataset(Dataset):
    def __init__(self, paths: List[Path], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.paths[idx]) as img:
            rgb = img.convert("RGB")
        return self.preprocess(rgb)


def get_or_extract_ddi_biomedclip(ddi_paths: List[Path], device: torch.device) -> torch.Tensor:
    """Load or extract BiomedCLIP 512D embeddings for the 656 Stanford DDI images."""
    cache_path = PROJECT_ROOT / "artifacts" / "embeddings" / "ddi_biomedclip.pt"
    if cache_path.exists():
        print(f"[*] Loading cached DDI BiomedCLIP embeddings from {cache_path}...")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"[*] Extracting BiomedCLIP (ViT-B/16) embeddings for {len(ddi_paths)} Stanford DDI images...")
    model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, _, preprocess = open_clip.create_model_and_transforms(model_id)
    model = model.to(device)
    model.eval()

    ds = SimplePilDataset(ddi_paths, preprocess)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    embeds = []
    t0 = time.time()
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            feats = model.encode_image(b)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeds.append(feats.cpu())

    tensor_out = torch.cat(embeds, dim=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_out, cache_path)
    print(f"[+] Extracted DDI BiomedCLIP embeddings in {time.time() - t0:.1f}s (shape: {tensor_out.shape})")
    return tensor_out


def main():
    t_start = time.time()
    print("=" * 104)
    print(" SkinAI EXP-017: Technical Review P0 Diagnostics, De-Confounding & External DDI Audit")
    print("=" * 104)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    splits_dir = PROJECT_ROOT / "data" / "splits" / "development_pool_v2"
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")

    cache_dir = PROJECT_ROOT / "artifacts" / "embeddings"
    tr_dino = torch.load(cache_dir / "dev_pool_v2_train_dinov2.pt", map_location="cpu", weights_only=True)
    va_dino = torch.load(cache_dir / "dev_pool_v2_val_dinov2.pt", map_location="cpu", weights_only=True)
    tr_biomed = torch.load(cache_dir / "dev_pool_v2_train_biomedclip.pt", map_location="cpu", weights_only=True)
    va_biomed = torch.load(cache_dir / "dev_pool_v2_val_biomedclip.pt", map_location="cpu", weights_only=True)

    # =========================================================================
    # SECTION 1: DATA INTEGRITY, EXACT SOURCE BREAKDOWN & LEAKAGE AUDIT (P0-01/02)
    # =========================================================================
    print("\n[SECTION 1] Exact Dataset Breakdown & Cross-Dataset Leakage Audit (P0-01 / P0-02)...")
    train_pids = set(train_df["patient_id"].astype(str))
    val_pids = set(val_df["patient_id"].astype(str))
    pid_overlap = len(train_pids.intersection(val_pids))

    train_lids = set(train_df["lesion_id"].astype(str))
    val_lids = set(val_df["lesion_id"].astype(str))
    lid_overlap = len(train_lids.intersection(val_lids))

    # Embedding near-duplicate check between train and val (cosine similarity > 0.995 across different sources)
    sim_matrix = torch.mm(va_dino, tr_dino.T)
    max_sims, max_tr_idx = torch.max(sim_matrix, dim=1)
    high_sim_count = int(torch.sum(max_sims > 0.995).item())
    cross_source_near_dups = 0
    for v_i in range(len(val_df)):
        if float(max_sims[v_i]) > 0.990:
            t_i = int(max_tr_idx[v_i])
            if str(val_df.iloc[v_i]["source_dataset"]) != str(train_df.iloc[t_i]["source_dataset"]):
                cross_source_near_dups += 1

    source_breakdown = {}
    for split_name, df_s in [("train", train_df), ("val", val_df)]:
        source_breakdown[split_name] = {}
        for src in ["pad_ufes_20", "mednode", "isic_clinical"]:
            sub = df_s[df_s["source_dataset"] == src]
            cls_counts = Counter(sub["subtype_target"].fillna("unmapped").astype(str))
            has_age = int(sub["age_approx"].notna().sum())
            has_sex = int((sub["sex"].fillna("").astype(str) != "").sum())
            has_site = int((sub["anatom_site_1"].fillna("").astype(str) != "").sum())
            source_breakdown[split_name][src] = {
                "n_images": int(len(sub)),
                "n_patients": int(sub["patient_id"].nunique()),
                "n_lesions": int(sub["lesion_id"].nunique()),
                "n_malignant": int((sub["triage_target"] == 1).sum()),
                "n_benign": int((sub["triage_target"] == 0).sum()),
                "class_counts": dict(cls_counts),
                "metadata_completeness": {
                    "age_present": has_age,
                    "sex_present": has_sex,
                    "site_present": has_site,
                },
            }
            print(
                f"  [{split_name.upper():5s} | {src:13s}] Images={len(sub):4d}, Patients={sub['patient_id'].nunique():4d}, "
                f"Mal={int((sub['triage_target']==1).sum()):3d}, Classes={dict(cls_counts)}"
            )

    print(
        f"  -> Patient Overlap={pid_overlap}, Lesion Overlap={lid_overlap}, "
        f"Exact/Near Dup (>0.995)={high_sim_count}, Cross-Source Near Dup (>0.990)={cross_source_near_dups}"
    )

    # =========================================================================
    # SECTION 2: SOURCE-PROBE & METADATA-ONLY SHORTCUT DIAGNOSTICS (P0-01/03)
    # =========================================================================
    print("\n[SECTION 2] Source-Probe & Metadata-Only Classifier Audit (P0-01 / P0-03)...")
    tr_clin_12d = np.stack([extract_clinical_vector(r) for _, r in train_df.iterrows()])
    va_clin_12d = np.stack([extract_clinical_vector(r) for _, r in val_df.iterrows()])

    tr_clin_7d = np.stack([extract_deconfounded_clinical_vector(r) for _, r in train_df.iterrows()])
    va_clin_7d = np.stack([extract_deconfounded_clinical_vector(r) for _, r in val_df.iterrows()])

    src_map = {"pad_ufes_20": 0, "mednode": 1, "isic_clinical": 2}
    y_src_tr = np.array([src_map[s] for s in train_df["source_dataset"]])
    y_src_va = np.array([src_map[s] for s in val_df["source_dataset"]])

    source_probes = {}
    for name, x_tr, x_va in [
        ("clinical_12d_raw", tr_clin_12d, va_clin_12d),
        ("clinical_7d_deconfounded", tr_clin_7d, va_clin_7d),
        ("dinov2_384d", tr_dino.numpy(), va_dino.numpy()),
        ("biomedclip_512d", tr_biomed.numpy(), va_biomed.numpy()),
    ]:
        clf_src = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf_src.fit(x_tr, y_src_tr)
        pred_src = clf_src.predict(x_va)
        acc_src = float(accuracy_score(y_src_va, pred_src))
        bal_src = float(balanced_accuracy_score(y_src_va, pred_src))
        source_probes[name] = {"accuracy": acc_src, "balanced_accuracy": bal_src}
        print(f"  [Source-Probe: {name:25s}] Source Acc = {acc_src*100:.2f}%, Balanced Acc = {bal_src*100:.2f}%")

    # Metadata-Only Classifier for Triage and 6-Class Subtyping
    y_triage_tr = train_df["triage_target"].values
    y_triage_va = val_df["triage_target"].values
    y_sub_tr = np.array([CLASS_TO_IDX.get(str(s), -1) for s in train_df["subtype_target"]])
    y_sub_va = np.array([CLASS_TO_IDX.get(str(s), -1) for s in val_df["subtype_target"]])

    clf_meta_triage = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf_meta_triage.fit(tr_clin_12d, (y_triage_tr == 1).astype(int))
    p_meta_triage = clf_meta_triage.predict_proba(va_clin_12d)

    valid_tr = y_sub_tr >= 0
    clf_meta_sub = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf_meta_sub.fit(tr_clin_12d[valid_tr], y_sub_tr[valid_tr])
    p_meta_sub = clf_meta_sub.predict_proba(va_clin_12d)

    meta_only_global = compute_detailed_subset_metrics(
        p_meta_triage, p_meta_sub, y_triage_va, y_sub_va, val_df["patient_id"].values
    )
    mask_pad_va = (val_df["source_dataset"] == "pad_ufes_20").values
    mask_med_va = (val_df["source_dataset"] == "mednode").values
    mask_isic_va = (val_df["source_dataset"] == "isic_clinical").values

    meta_only_pad = compute_detailed_subset_metrics(
        p_meta_triage[mask_pad_va],
        p_meta_sub[mask_pad_va],
        y_triage_va[mask_pad_va],
        y_sub_va[mask_pad_va],
        val_df["patient_id"].values[mask_pad_va],
    )
    meta_only_med = compute_detailed_subset_metrics(
        p_meta_triage[mask_med_va],
        p_meta_sub[mask_med_va],
        y_triage_va[mask_med_va],
        y_sub_va[mask_med_va],
        val_df["patient_id"].values[mask_med_va],
    )
    print(
        f"  [Metadata-Only Classifier] Global Macro F1 = {meta_only_global['macro_f1_present']:.4f}, "
        f"Triage AUC = {meta_only_global['triage_roc_auc']:.4f} | "
        f"Within PAD-UFES-20 Macro F1 = {meta_only_pad['macro_f1_present']:.4f}, Triage AUC = {meta_only_pad['triage_roc_auc']:.4f} | "
        f"Within MED-NODE Triage AUC = {meta_only_med['triage_roc_auc']:.4f} (0.5000 = pure chance!)"
    )

    # =========================================================================
    # SECTION 3: MODALITY ABLATION & PER-SOURCE EVALUATION ON VAL.CSV (P0-01/03)
    # =========================================================================
    print("\n[SECTION 3] Image-Only vs. De-Confounded Multimodal vs. Original Multimodal across Sources...")
    params_dino = {
        "lr": 0.0003,
        "weight_decay": 0.001,
        "gamma": 2.5,
        "class_weight_power": 0.75,
        "dropout": 0.15,
        "hidden_dim": 128,
        "clin_embed_dim": 32,
        "batch_size": 32,
        "activation": "silu",
    }
    params_biomed = {
        "lr": 0.002,
        "weight_decay": 0.001,
        "gamma": 2.5,
        "class_weight_power": 0.75,
        "dropout": 0.15,
        "hidden_dim": 256,
        "clin_embed_dim": 32,
        "batch_size": 64,
        "activation": "gelu",
    }

    # Also prepare DDI embeddings so we can evaluate the exact same trained models on DDI in Section 6!
    from scripts.evaluate_external_ddi_locked import find_image_file, locate_ddi_dataset

    ddi_df_raw, ddi_root = locate_ddi_dataset()
    file_col = "DDI_file" if "DDI_file" in ddi_df_raw.columns else "file_name"
    ddi_paths = []
    ddi_rows = []
    for _, r in ddi_df_raw.iterrows():
        p = find_image_file(ddi_root, str(r[file_col]).strip())
        if p is not None:
            ddi_paths.append(p)
            ddi_rows.append(r)
    ddi_df = pd.DataFrame(ddi_rows)
    ddi_dino = torch.load(cache_dir / "ddi_dinov2.pt", map_location="cpu", weights_only=True)
    ddi_biomed = get_or_extract_ddi_biomedclip(ddi_paths, device)

    ddi_clin_12d = torch.tensor(
        np.stack([extract_clinical_vector(r) for _, r in ddi_df.iterrows()]),
        dtype=torch.float32,
    )
    ddi_clin_7d = torch.tensor(
        np.stack([extract_deconfounded_clinical_vector(r) for _, r in ddi_df.iterrows()]),
        dtype=torch.float32,
    )

    tr_t_tensor = torch.tensor(y_triage_tr, dtype=torch.long)
    tr_s_tensor = torch.tensor(y_sub_tr, dtype=torch.long)
    tr_c12_tensor = torch.tensor(tr_clin_12d, dtype=torch.float32)
    va_c12_tensor = torch.tensor(va_clin_12d, dtype=torch.float32)
    tr_c7_tensor = torch.tensor(tr_clin_7d, dtype=torch.float32)
    va_c7_tensor = torch.tensor(va_clin_7d, dtype=torch.float32)

    # Accumulate predictions across 5 seeds for statistical stability and ensemble averaging
    preds_val = {
        "img_only_dino": ([], []),
        "img_only_biomed": ([], []),
        "img_only_consilium": ([], []),
        "deconfounded_mm_consilium": ([], []),
        "raw_mm_consilium_exp015": ([], []),
    }
    preds_ddi = {
        "img_only_dino": ([], []),
        "img_only_biomed": ([], []),
        "img_only_consilium": ([], []),
        "deconfounded_mm_consilium": ([], []),
        "raw_mm_consilium_exp015": ([], []),
    }

    for seed in SEEDS:
        # 1. Image-Only DINOv2 & BiomedCLIP
        (tp_d_va, sp_d_va), (tp_d_ddi, sp_d_ddi) = train_and_predict_image_only(
            tr_dino, tr_t_tensor, tr_s_tensor, [va_dino, ddi_dino], params_dino, seed, device
        )
        (tp_b_va, sp_b_va), (tp_b_ddi, sp_b_ddi) = train_and_predict_image_only(
            tr_biomed, tr_t_tensor, tr_s_tensor, [va_biomed, ddi_biomed], params_biomed, seed, device
        )
        preds_val["img_only_dino"][0].append(tp_d_va)
        preds_val["img_only_dino"][1].append(sp_d_va)
        preds_val["img_only_biomed"][0].append(tp_b_va)
        preds_val["img_only_biomed"][1].append(sp_b_va)
        preds_val["img_only_consilium"][0].append(0.5 * tp_d_va + 0.5 * tp_b_va)
        preds_val["img_only_consilium"][1].append(0.5 * sp_d_va + 0.5 * sp_b_va)

        preds_ddi["img_only_dino"][0].append(tp_d_ddi)
        preds_ddi["img_only_dino"][1].append(sp_d_ddi)
        preds_ddi["img_only_biomed"][0].append(tp_b_ddi)
        preds_ddi["img_only_biomed"][1].append(sp_b_ddi)
        preds_ddi["img_only_consilium"][0].append(0.5 * tp_d_ddi + 0.5 * tp_b_ddi)
        preds_ddi["img_only_consilium"][1].append(0.5 * sp_d_ddi + 0.5 * sp_b_ddi)

        # 2. De-Confounded Multimodal (7D without unknown flags + 50% metadata dropout)
        (tp_dc_d_va, sp_dc_d_va), (tp_dc_d_ddi, sp_dc_d_ddi) = train_and_predict_multimodal(
            tr_dino,
            tr_c7_tensor,
            tr_t_tensor,
            tr_s_tensor,
            [(va_dino, va_c7_tensor), (ddi_dino, ddi_clin_7d)],
            params_dino,
            seed,
            device,
            metadata_dropout_prob=0.50,
        )
        (tp_dc_b_va, sp_dc_b_va), (tp_dc_b_ddi, sp_dc_b_ddi) = train_and_predict_multimodal(
            tr_biomed,
            tr_c7_tensor,
            tr_t_tensor,
            tr_s_tensor,
            [(va_biomed, va_c7_tensor), (ddi_biomed, ddi_clin_7d)],
            params_biomed,
            seed,
            device,
            metadata_dropout_prob=0.50,
        )
        preds_val["deconfounded_mm_consilium"][0].append(0.5 * tp_dc_d_va + 0.5 * tp_dc_b_va)
        preds_val["deconfounded_mm_consilium"][1].append(0.5 * sp_dc_d_va + 0.5 * sp_dc_b_va)
        preds_ddi["deconfounded_mm_consilium"][0].append(0.5 * tp_dc_d_ddi + 0.5 * tp_dc_b_ddi)
        preds_ddi["deconfounded_mm_consilium"][1].append(0.5 * sp_dc_d_ddi + 0.5 * sp_dc_b_ddi)

        # 3. Original Multimodal Fusion (EXP-015 with 12D clinical vector)
        (tp_raw_d_va, sp_raw_d_va), (tp_raw_d_ddi, sp_raw_d_ddi) = train_and_predict_multimodal(
            tr_dino,
            tr_c12_tensor,
            tr_t_tensor,
            tr_s_tensor,
            [(va_dino, va_c12_tensor), (ddi_dino, ddi_clin_12d)],
            params_dino,
            seed,
            device,
            metadata_dropout_prob=0.0,
        )
        (tp_raw_b_va, sp_raw_b_va), (tp_raw_b_ddi, sp_raw_b_ddi) = train_and_predict_multimodal(
            tr_biomed,
            tr_c12_tensor,
            tr_t_tensor,
            tr_s_tensor,
            [(va_biomed, va_c12_tensor), (ddi_biomed, ddi_clin_12d)],
            params_biomed,
            seed,
            device,
            metadata_dropout_prob=0.0,
        )
        preds_val["raw_mm_consilium_exp015"][0].append(0.5 * tp_raw_d_va + 0.5 * tp_raw_b_va)
        preds_val["raw_mm_consilium_exp015"][1].append(0.5 * sp_raw_d_va + 0.5 * sp_raw_b_va)
        preds_ddi["raw_mm_consilium_exp015"][0].append(0.5 * tp_raw_d_ddi + 0.5 * tp_raw_b_ddi)
        preds_ddi["raw_mm_consilium_exp015"][1].append(0.5 * sp_raw_d_ddi + 0.5 * sp_raw_b_ddi)

    modality_source_table = {}
    val_pids_arr = val_df["patient_id"].astype(str).values

    for m_key in preds_val:
        mean_tp = np.mean(preds_val[m_key][0], axis=0)
        mean_sp = np.mean(preds_val[m_key][1], axis=0)

        # Also compute 5-seed mean +/- std of Macro F1 on global and PAD-UFES-20
        seed_f1_global = []
        seed_f1_pad = []
        for s_i in range(len(SEEDS)):
            g_m = compute_detailed_subset_metrics(
                preds_val[m_key][0][s_i], preds_val[m_key][1][s_i], y_triage_va, y_sub_va
            )
            p_m = compute_detailed_subset_metrics(
                preds_val[m_key][0][s_i][mask_pad_va],
                preds_val[m_key][1][s_i][mask_pad_va],
                y_triage_va[mask_pad_va],
                y_sub_va[mask_pad_va],
            )
            seed_f1_global.append(g_m["macro_f1_present"])
            seed_f1_pad.append(p_m["macro_f1_present"])

        m_global = compute_detailed_subset_metrics(
            mean_tp, mean_sp, y_triage_va, y_sub_va, val_pids_arr, compute_bootstrap=True
        )
        m_pad = compute_detailed_subset_metrics(
            mean_tp[mask_pad_va],
            mean_sp[mask_pad_va],
            y_triage_va[mask_pad_va],
            y_sub_va[mask_pad_va],
            val_pids_arr[mask_pad_va],
            compute_bootstrap=True,
        )
        m_med = compute_detailed_subset_metrics(
            mean_tp[mask_med_va],
            mean_sp[mask_med_va],
            y_triage_va[mask_med_va],
            y_sub_va[mask_med_va],
            val_pids_arr[mask_med_va],
            compute_bootstrap=True,
        )
        m_isic = compute_detailed_subset_metrics(
            mean_tp[mask_isic_va],
            mean_sp[mask_isic_va],
            y_triage_va[mask_isic_va],
            y_sub_va[mask_isic_va],
            val_pids_arr[mask_isic_va],
            compute_bootstrap=True,
        )

        modality_source_table[m_key] = {
            "seed_macro_f1_global_mean": float(np.mean(seed_f1_global)),
            "seed_macro_f1_global_std": float(np.std(seed_f1_global)),
            "seed_macro_f1_pad_mean": float(np.mean(seed_f1_pad)),
            "seed_macro_f1_pad_std": float(np.std(seed_f1_pad)),
            "global_val": m_global,
            "pad_ufes_20_only": m_pad,
            "med_node_only": m_med,
            "isic_clinical_only": m_isic,
        }

        pad_mel_rec = m_pad["per_class_recall_argmax"].get("mel", {})
        med_mel_rec = m_med["per_class_recall_argmax"].get("mel", {})
        isic_mel_rec = m_isic["per_class_recall_argmax"].get("mel", {})

        print(
            f"  [{m_key:25s}] Global F1={m_global['macro_f1_present']:.4f} (95% CI {m_global['macro_f1_bootstrap_95_ci'][0]:.4f}-{m_global['macro_f1_bootstrap_95_ci'][1]:.4f}) | "
            f"PAD-UFES-20 F1={m_pad['macro_f1_present']:.4f}, Triage AUC={m_pad['triage_roc_auc']:.4f}, "
            f"PAD Mel={pad_mel_rec.get('tp',0)}/{pad_mel_rec.get('n',0)} ({pad_mel_rec.get('recall',0)*100:.1f}%) | "
            f"MED-NODE AUC={m_med['triage_roc_auc']:.4f}, MED Mel={med_mel_rec.get('tp',0)}/{med_mel_rec.get('n',0)} ({med_mel_rec.get('recall',0)*100:.1f}%) | "
            f"ISIC AUC={m_isic['triage_roc_auc']:.4f}, ISIC Mel={isic_mel_rec.get('tp',0)}/{isic_mel_rec.get('n',0)}"
        )

    # =========================================================================
    # SECTION 4: LEAVE-ONE-SOURCE-OUT & PURE-PAD TRAINING (Leave-One-Source-Out Experiment)
    # =========================================================================
    print("\n[SECTION 4] Leave-One-Source-Out & Pure PAD-UFES-20 Training (Experiment E)...")
    mask_pad_tr = (train_df["source_dataset"] == "pad_ufes_20").values
    mask_med_tr = (train_df["source_dataset"] == "mednode").values
    mask_no_med_tr = ~mask_med_tr

    # Combine all 170 MED-NODE images (train + val) as a held-out zero-shot target
    med_all_dino = torch.cat([tr_dino[mask_med_tr], va_dino[mask_med_va]], dim=0)
    med_all_biomed = torch.cat([tr_biomed[mask_med_tr], va_biomed[mask_med_va]], dim=0)
    med_all_triage = np.concatenate([y_triage_tr[mask_med_tr], y_triage_va[mask_med_va]])
    med_all_sub = np.concatenate([y_sub_tr[mask_med_tr], y_sub_va[mask_med_va]])

    # 4A. Train Image-Only Fusion on PAD + ISIC (0% MED-NODE) -> Evaluate on all 170 MED-NODE & PAD val
    (tp_nomed_d_med, sp_nomed_d_med), (tp_nomed_d_pad, sp_nomed_d_pad) = train_and_predict_image_only(
        tr_dino[mask_no_med_tr],
        tr_t_tensor[mask_no_med_tr],
        tr_s_tensor[mask_no_med_tr],
        [med_all_dino, va_dino[mask_pad_va]],
        params_dino,
        42,
        device,
    )
    (tp_nomed_b_med, sp_nomed_b_med), (tp_nomed_b_pad, sp_nomed_b_pad) = train_and_predict_image_only(
        tr_biomed[mask_no_med_tr],
        tr_t_tensor[mask_no_med_tr],
        tr_s_tensor[mask_no_med_tr],
        [med_all_biomed, va_biomed[mask_pad_va]],
        params_biomed,
        42,
        device,
    )
    loso_med_metrics = compute_detailed_subset_metrics(
        0.5 * tp_nomed_d_med + 0.5 * tp_nomed_b_med,
        0.5 * sp_nomed_d_med + 0.5 * sp_nomed_b_med,
        med_all_triage,
        med_all_sub,
    )
    loso_pad_metrics = compute_detailed_subset_metrics(
        0.5 * tp_nomed_d_pad + 0.5 * tp_nomed_b_pad,
        0.5 * sp_nomed_d_pad + 0.5 * sp_nomed_b_pad,
        y_triage_va[mask_pad_va],
        y_sub_va[mask_pad_va],
    )
    print(
        f"  [4A. Leave-MED-NODE-Out (Train on PAD+ISIC -> Zero-Shot 170 MED-NODE)] "
        f"MED-NODE ROC-AUC = {loso_med_metrics['triage_roc_auc']:.4f}, "
        f"Binary F1 = {loso_med_metrics['macro_f1_present']:.4f}, "
        f"Mel Recall (tau=0.05) = {loso_med_metrics['mel_recall_tau005']['tp']}/{loso_med_metrics['mel_recall_tau005']['n']} "
        f"({loso_med_metrics['mel_recall_tau005']['recall']*100:.1f}%) | "
        f"PAD Val Macro F1 = {loso_pad_metrics['macro_f1_present']:.4f}"
    )

    # 4B. Train strictly on PAD-UFES-20 Only (1,834 smartphone images) -> Evaluate on PAD Val & 170 MED-NODE
    (tp_padonly_d_pad, sp_padonly_d_pad), (tp_padonly_d_med, sp_padonly_d_med) = train_and_predict_image_only(
        tr_dino[mask_pad_tr],
        tr_t_tensor[mask_pad_tr],
        tr_s_tensor[mask_pad_tr],
        [va_dino[mask_pad_va], med_all_dino],
        params_dino,
        42,
        device,
    )
    (tp_padonly_b_pad, sp_padonly_b_pad), (tp_padonly_b_med, sp_padonly_b_med) = train_and_predict_image_only(
        tr_biomed[mask_pad_tr],
        tr_t_tensor[mask_pad_tr],
        tr_s_tensor[mask_pad_tr],
        [va_biomed[mask_pad_va], med_all_biomed],
        params_biomed,
        42,
        device,
    )
    pure_pad_on_pad = compute_detailed_subset_metrics(
        0.5 * tp_padonly_d_pad + 0.5 * tp_padonly_b_pad,
        0.5 * sp_padonly_d_pad + 0.5 * sp_padonly_b_pad,
        y_triage_va[mask_pad_va],
        y_sub_va[mask_pad_va],
    )
    pure_pad_on_med = compute_detailed_subset_metrics(
        0.5 * tp_padonly_d_med + 0.5 * tp_padonly_b_med,
        0.5 * sp_padonly_d_med + 0.5 * sp_padonly_b_med,
        med_all_triage,
        med_all_sub,
    )
    print(
        f"  [4B. Train on PAD-UFES-20 ONLY] PAD Val Macro F1 = {pure_pad_on_pad['macro_f1_present']:.4f}, "
        f"Triage AUC = {pure_pad_on_pad['triage_roc_auc']:.4f} | "
        f"Zero-Shot on 170 MED-NODE ROC-AUC = {pure_pad_on_med['triage_roc_auc']:.4f}"
    )

    # =========================================================================
    # SECTION 5: 5-FOLD STRATIFIED GROUP CV ACROSS ALL 3,767 IMAGES (P0-04)
    # =========================================================================
    print("\n[SECTION 5] 5-Fold StratifiedGroupKFold Out-of-Fold Evaluation across all 3,767 Images (142 Melanomas)...")
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    full_dino = torch.cat([tr_dino, va_dino], dim=0)
    full_biomed = torch.cat([tr_biomed, va_biomed], dim=0)
    full_c7 = torch.cat([tr_c7_tensor, va_c7_tensor], dim=0)
    full_c12 = torch.cat([tr_c12_tensor, va_c12_tensor], dim=0)
    full_triage = np.concatenate([y_triage_tr, y_triage_va])
    full_sub = np.concatenate([y_sub_tr, y_sub_va])
    full_pids = full_df["patient_id"].astype(str).values
    full_sources = full_df["source_dataset"].astype(str).values

    # Stratify by source + subtype to guarantee balanced source & class representation per fold
    strat_key = np.array([f"{src}_{sub}" for src, sub in zip(full_sources, full_df["subtype_target"].fillna("unmapped"))])
    sgkf5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_img_tp = np.zeros((len(full_df), 2), dtype=np.float32)
    oof_img_sp = np.zeros((len(full_df), 6), dtype=np.float32)
    oof_mm_tp = np.zeros((len(full_df), 2), dtype=np.float32)
    oof_mm_sp = np.zeros((len(full_df), 6), dtype=np.float32)

    fold_f1_img = []
    fold_f1_mm = []

    for fold_idx, (f_tr_idx, f_va_idx) in enumerate(sgkf5.split(full_df, strat_key, full_pids)):
        f_tr_t = torch.tensor(full_triage[f_tr_idx], dtype=torch.long)
        f_tr_s = torch.tensor(full_sub[f_tr_idx], dtype=torch.long)

        # Image-only Fusion on fold
        (ftp_d, fsp_d), = train_and_predict_image_only(
            full_dino[f_tr_idx], f_tr_t, f_tr_s, [full_dino[f_va_idx]], params_dino, 42 + fold_idx, device
        )
        (ftp_b, fsp_b), = train_and_predict_image_only(
            full_biomed[f_tr_idx], f_tr_t, f_tr_s, [full_biomed[f_va_idx]], params_biomed, 42 + fold_idx, device
        )
        oof_img_tp[f_va_idx] = 0.5 * ftp_d + 0.5 * ftp_b
        oof_img_sp[f_va_idx] = 0.5 * fsp_d + 0.5 * fsp_b

        # De-confounded Multimodal Fusion on fold
        (mtp_d, msp_d), = train_and_predict_multimodal(
            full_dino[f_tr_idx],
            full_c7[f_tr_idx],
            f_tr_t,
            f_tr_s,
            [(full_dino[f_va_idx], full_c7[f_va_idx])],
            params_dino,
            42 + fold_idx,
            device,
            metadata_dropout_prob=0.50,
        )
        (mtp_b, msp_b), = train_and_predict_multimodal(
            full_biomed[f_tr_idx],
            full_c7[f_tr_idx],
            f_tr_t,
            f_tr_s,
            [(full_biomed[f_va_idx], full_c7[f_va_idx])],
            params_biomed,
            42 + fold_idx,
            device,
            metadata_dropout_prob=0.50,
        )
        oof_mm_tp[f_va_idx] = 0.5 * mtp_d + 0.5 * mtp_b
        oof_mm_sp[f_va_idx] = 0.5 * msp_d + 0.5 * msp_b

        f_m_img = compute_detailed_subset_metrics(
            oof_img_tp[f_va_idx], oof_img_sp[f_va_idx], full_triage[f_va_idx], full_sub[f_va_idx]
        )
        f_m_mm = compute_detailed_subset_metrics(
            oof_mm_tp[f_va_idx], oof_mm_sp[f_va_idx], full_triage[f_va_idx], full_sub[f_va_idx]
        )
        fold_f1_img.append(f_m_img["macro_f1_present"])
        fold_f1_mm.append(f_m_mm["macro_f1_present"])
        print(
            f"  [Fold {fold_idx+1}/5] Image-Only F1 = {f_m_img['macro_f1_present']:.4f} (AUC={f_m_img['triage_roc_auc']:.4f}) | "
            f"Deconfounded-MM F1 = {f_m_mm['macro_f1_present']:.4f} (AUC={f_m_mm['triage_roc_auc']:.4f})"
        )

    oof_img_global = compute_detailed_subset_metrics(
        oof_img_tp, oof_img_sp, full_triage, full_sub, full_pids, compute_bootstrap=True
    )
    oof_img_pad = compute_detailed_subset_metrics(
        oof_img_tp[full_sources == "pad_ufes_20"],
        oof_img_sp[full_sources == "pad_ufes_20"],
        full_triage[full_sources == "pad_ufes_20"],
        full_sub[full_sources == "pad_ufes_20"],
        full_pids[full_sources == "pad_ufes_20"],
        compute_bootstrap=True,
    )
    oof_img_med = compute_detailed_subset_metrics(
        oof_img_tp[full_sources == "mednode"],
        oof_img_sp[full_sources == "mednode"],
        full_triage[full_sources == "mednode"],
        full_sub[full_sources == "mednode"],
        full_pids[full_sources == "mednode"],
    )
    oof_mm_global = compute_detailed_subset_metrics(
        oof_mm_tp, oof_mm_sp, full_triage, full_sub, full_pids, compute_bootstrap=True
    )
    oof_mm_pad = compute_detailed_subset_metrics(
        oof_mm_tp[full_sources == "pad_ufes_20"],
        oof_mm_sp[full_sources == "pad_ufes_20"],
        full_triage[full_sources == "pad_ufes_20"],
        full_sub[full_sources == "pad_ufes_20"],
        full_pids[full_sources == "pad_ufes_20"],
        compute_bootstrap=True,
    )

    print(
        f"  -> [5-Fold OOF Image-Only (N=3,767, 142 Mel)] Global F1 = {oof_img_global['macro_f1_present']:.4f} "
        f"(Fold Mean = {np.mean(fold_f1_img):.4f} +/- {np.std(fold_f1_img):.4f}), "
        f"Mel Argmax Recall = {oof_img_global['per_class_recall_argmax']['mel']['tp']}/142 "
        f"({oof_img_global['per_class_recall_argmax']['mel']['recall']*100:.1f}%), "
        f"Mel Recall (tau=0.05) = {oof_img_global['mel_recall_tau005']['tp']}/142 "
        f"({oof_img_global['mel_recall_tau005']['recall']*100:.1f}%)"
    )
    print(
        f"  -> [5-Fold OOF Deconfounded-MM (N=3,767, 142 Mel)] Global F1 = {oof_mm_global['macro_f1_present']:.4f} "
        f"(Fold Mean = {np.mean(fold_f1_mm):.4f} +/- {np.std(fold_f1_mm):.4f}) | "
        f"PAD-UFES-20 OOF F1 = {oof_mm_pad['macro_f1_present']:.4f} (vs Image-Only PAD OOF F1 = {oof_img_pad['macro_f1_present']:.4f})"
    )

    # =========================================================================
    # SECTION 6: EXTERNAL STANFORD DDI AUDIT & THRESHOLD CALIBRATION (P0-05)
    # =========================================================================
    print("\n[SECTION 6] External Stanford DDI Audit (656 Images) & Non-Oracle Calibration...")
    mal_col = "malignant" if "malignant" in ddi_df.columns else "malignancy(malig=1)"
    tone_col = "skin_tone" if "skin_tone" in ddi_df.columns else "FST"
    disease_col = "disease" if "disease" in ddi_df.columns else "diagnosis"

    y_ddi_mal = ddi_df[mal_col].apply(
        lambda x: 1 if (x is True or x == 1 or str(x).lower() in ["true", "1"]) else 0
    ).values
    n_ddi = len(y_ddi_mal)
    n_ddi_mal = int(np.sum(y_ddi_mal == 1))
    n_ddi_ben = int(np.sum(y_ddi_mal == 0))

    # Map DDI diseases to 6-class taxonomy vs Open-Set
    def map_ddi_disease(d_str: str) -> int:
        s = str(d_str).lower().strip()
        if "melanoma" in s:
            return CLASS_TO_IDX["mel"]
        if "basal-cell-carcinoma" in s or "basal cell carcinoma" in s:
            return CLASS_TO_IDX["bcc"]
        if "squamous-cell-carcinoma" in s or "squamous cell carcinoma" in s:
            return CLASS_TO_IDX["scc"]
        if "nevus" in s or "nevi" in s:
            return CLASS_TO_IDX["nev"]
        if "seborrheic-keratosis" in s or "seborrheic keratosis" in s:
            return CLASS_TO_IDX["sek"]
        if "actinic-keratosis" in s or "actinic keratosis" in s:
            return CLASS_TO_IDX["ack"]
        return -1

    y_ddi_sub = np.array([map_ddi_disease(d) for d in ddi_df[disease_col]])
    in_scope_mask = y_ddi_sub >= 0
    open_set_mask = ~in_scope_mask

    ddi_comparison = {}
    for m_key in ["raw_mm_consilium_exp015", "deconfounded_mm_consilium", "img_only_consilium", "img_only_dino", "img_only_biomed"]:
        tp_ddi = np.mean(preds_ddi[m_key][0], axis=0)
        sp_ddi = np.mean(preds_ddi[m_key][1], axis=0)
        p_mal = tp_ddi[:, 1]

        auc_all = float(roc_auc_score(y_ddi_mal, p_mal))
        auc_in_scope = float(roc_auc_score(y_ddi_mal[in_scope_mask], p_mal[in_scope_mask]))
        auc_open_set = float(roc_auc_score(y_ddi_mal[open_set_mask], p_mal[open_set_mask]))

        # Operating points at tau=0.20, tau=0.50
        sens_020 = float(np.mean(p_mal[y_ddi_mal == 1] >= 0.20))
        spec_020 = float(np.mean(p_mal[y_ddi_mal == 0] < 0.20))
        sens_050 = float(np.mean(p_mal[y_ddi_mal == 1] >= 0.50))
        spec_050 = float(np.mean(p_mal[y_ddi_mal == 0] < 0.50))

        # Unsupervised Saerens EM Prior Shift Adaptation (zero target labels used!)
        pi_train = float(np.mean(y_triage_tr == 1))
        p_saerens, est_pi_target = saerens_em_prior_shift(p_mal, pi_source=pi_train)
        sens_saerens_050 = float(np.mean(p_saerens[y_ddi_mal == 1] >= 0.50))
        spec_saerens_050 = float(np.mean(p_saerens[y_ddi_mal == 0] < 0.50))

        # FST Breakdown
        fst_aucs = {}
        for tone in sorted(list(set(ddi_df[tone_col]))):
            t_mask = (ddi_df[tone_col] == tone).values
            if len(set(y_ddi_mal[t_mask])) > 1:
                fst_aucs[str(tone)] = float(roc_auc_score(y_ddi_mal[t_mask], p_mal[t_mask]))

        # Local Hospital Split Calibration (Protocol: 25% calibration split -> 75% locked test)
        skf_ddi = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        cal_sens_list, cal_spec_list, cal_tau_list = [], [], []
        for test_idx, cal_idx in skf_ddi.split(np.zeros(n_ddi), y_ddi_mal):
            # Note: skf.split returns (75% fold, 25% fold); use 25% fold (cal_idx) to pick threshold for >=75% sens / max J
            fpr_c, tpr_c, thr_c = roc_curve(y_ddi_mal[cal_idx], p_mal[cal_idx])
            j_idx = int(np.argmax(tpr_c - fpr_c))
            tau_local = float(thr_c[j_idx])
            cal_tau_list.append(tau_local)
            cal_sens_list.append(float(np.mean(p_mal[test_idx][y_ddi_mal[test_idx] == 1] >= tau_local)))
            cal_spec_list.append(float(np.mean(p_mal[test_idx][y_ddi_mal[test_idx] == 0] < tau_local)))

        # Three-Way Decision Triage (Calibrated Green / Yellow / Red triage zones on probabilities)
        # Green: p < 0.35 (Routine observation), Grey: 0.35 <= p < 0.65 (Planned Derm / Second Opinion), Red: p >= 0.65 (Urgent Oncology)
        green_mask = p_mal < 0.35
        grey_mask = (p_mal >= 0.35) & (p_mal < 0.65)
        red_mask = p_mal >= 0.65
        mal_in_red_or_grey = float(np.mean((red_mask | grey_mask)[y_ddi_mal == 1]))
        ben_spared_from_urgent_red = float(np.mean((~red_mask)[y_ddi_mal == 0]))

        ddi_comparison[m_key] = {
            "triage_roc_auc_all_656": auc_all,
            "triage_roc_auc_in_scope_301": auc_in_scope,
            "triage_roc_auc_open_set_355": auc_open_set,
            "fst_roc_auc": fst_aucs,
            "fixed_tau_020": {"sens": sens_020, "spec": spec_020},
            "fixed_tau_050": {"sens": sens_050, "spec": spec_050},
            "saerens_em_unsupervised_tau_050": {
                "estimated_pi_target": est_pi_target,
                "sens": sens_saerens_050,
                "spec": spec_saerens_050,
            },
            "local_hospital_25pct_calibrated_cv": {
                "mean_tau": float(np.mean(cal_tau_list)),
                "heldout_75pct_sens_mean": float(np.mean(cal_sens_list)),
                "heldout_75pct_spec_mean": float(np.mean(cal_spec_list)),
            },
            "three_way_triage_zones": {
                "green_fraction": float(np.mean(green_mask)),
                "grey_fraction": float(np.mean(grey_mask)),
                "red_fraction": float(np.mean(red_mask)),
                "malignant_captured_in_red_or_grey": mal_in_red_or_grey,
                "benign_spared_from_urgent_red": ben_spared_from_urgent_red,
            },
        }
        print(
            f"  [DDI: {m_key:25s}] AUC All={auc_all:.4f} (In-Scope={auc_in_scope:.4f}, Open-Set={auc_open_set:.4f}, FST56={fst_aucs.get('56',0):.4f}) | "
            f"@tau=0.20 Sens={sens_020*100:.1f}%/Spec={spec_020*100:.1f}% | "
            f"@tau=0.50 Sens={sens_050*100:.1f}%/Spec={spec_050*100:.1f}% | "
            f"Local-Calibrated Held-out Sens={np.mean(cal_sens_list)*100:.1f}%/Spec={np.mean(cal_spec_list)*100:.1f}%"
        )

    # Save complete EXP-017 report JSON
    out_dir = PROJECT_ROOT / "artifacts" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp017_fusion_diagnostics.json"

    report = {
        "experiment_id": "EXP-017",
        "title": "Technical Review P0 Diagnostics, De-Confounding & External DDI Audit",
        "elapsed_seconds": round(time.time() - t_start, 2),
        "section_1_data_integrity": {
            "patient_overlap_train_val": pid_overlap,
            "lesion_overlap_train_val": lid_overlap,
            "embedding_near_duplicates_gt_0995": high_sim_count,
            "cross_source_near_duplicates_gt_0990": cross_source_near_dups,
            "source_breakdown": source_breakdown,
        },
        "section_2_shortcut_probes": {
            "source_probes": source_probes,
            "metadata_only_classifier": {
                "global_val": meta_only_global,
                "pad_ufes_20_only": meta_only_pad,
                "med_node_only": meta_only_med,
            },
        },
        "section_3_modality_and_source_breakdown": modality_source_table,
        "section_4_leave_one_source_out": {
            "train_pad_plus_isic_eval_170_mednode": loso_med_metrics,
            "train_pad_plus_isic_eval_pad_val": loso_pad_metrics,
            "train_pure_pad_eval_pad_val": pure_pad_on_pad,
            "train_pure_pad_eval_170_mednode": pure_pad_on_med,
        },
        "section_5_5fold_stratified_group_cv_3767": {
            "image_only_consilium": {
                "fold_macro_f1_mean": float(np.mean(fold_f1_img)),
                "fold_macro_f1_std": float(np.std(fold_f1_img)),
                "oof_global_3767": oof_img_global,
                "oof_pad_ufes_20": oof_img_pad,
                "oof_med_node": oof_img_med,
            },
            "deconfounded_mm_consilium": {
                "fold_macro_f1_mean": float(np.mean(fold_f1_mm)),
                "fold_macro_f1_std": float(np.std(fold_f1_mm)),
                "oof_global_3767": oof_mm_global,
                "oof_pad_ufes_20": oof_mm_pad,
            },
        },
        "section_6_external_stanford_ddi_656": ddi_comparison,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 104)
    print(f"[+] EXP-017 Complete in {time.time() - t_start:.1f}s! Saved full report to: {out_path}")
    print("=" * 104)


if __name__ == "__main__":
    main()

