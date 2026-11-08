#!/usr/bin/env python3
"""EXP-015: Multi-Backbone Dual-Backbone Fusion & Ensemble Study (DINOv2 + BiomedCLIP + ConvNeXt-Tiny).

Purpose:
Systematically evaluate how to combine the complementary strengths of our top 3 distinct
vision families discovered in EXP-013 and EXP-014:
- Meta DINOv2 Small (384D SSL ViT): Best overall 6-class discrimination (Macro F1 = 0.7275, Bal Acc = 73.05%)
- Microsoft BiomedCLIP (512D Medical ViT-B/16): Domain-pretrained biomedical representations (Mel Recall = 91.7%)
- ConvNeXt-Tiny (768D Modern 7x7 CNN): Local convolutional border/texture inductive bias

Architectures & Fusion Strategies Compared (across 5 random seeds [0, 8, 27, 42, 2026]):
1. Single-Backbone Specialists (Optuna-Optimal baselines from EXP-014):
   - 1A. Meta DINOv2 Small Specialist
   - 1B. Microsoft BiomedCLIP Specialist
   - 1C. ConvNeXt-Tiny Specialist
2. Strategy A — Naive Feature Concatenation (Early Concatenation):
   - 2A. Concat [DINOv2 (384D) + BiomedCLIP (512D)] = 896D
   - 2B. Concat [DINOv2 (384D) + BiomedCLIP (512D) + ConvNeXt (768D)] = 1664D
3. Strategy B — Equal-Dimension Gated Attention Fusion (Learned Expert Gating):
   - Projects each backbone into an isometric 128D latent space and applies a learned
     per-sample Softmax Attention Gate over experts before fusing with clinical anamnesis.
   - 3A. Gated Attention [DINOv2 + BiomedCLIP]
   - 3B. Gated Attention [DINOv2 + BiomedCLIP + ConvNeXt]
4. Strategy C — Late Probability Fusion (Independent Specialists + Convex Probability Fusion):
   - Trains each specialist in its own optimal low-dimensional space (avoiding the curse of dimensionality)
     and combines calibrated probabilities.
   - 4A. Equal Dual Fusion (0.50 DINOv2 + 0.50 BiomedCLIP)
   - 4B. Equal Tri-Fusion (1/3 DINOv2 + 1/3 BiomedCLIP + 1/3 ConvNeXt)
   - 4C. Inner-Val Optimized Weighted Fusion (weights w1, w2, w3 tuned strictly on Inner-Val 570 images,
         evaluated once on untouched Hold-out Val 757 images).
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.multi_head import MaskedClassWeightedFocalLoss, MultiHeadLoss
from scripts.run_exp013_architecture_tournament import (
    CLASS_TO_IDX,
    SEEDS,
    aggregate_seed_metrics,
    evaluate_multimodal_predictions,
    extract_clinical_vector,
    set_seed,
)
from scripts.run_exp014_optuna_and_hyperparam_proofs import (
    ConfigurableMultimodalProbe,
    compute_powered_weights,
)


class GatedMultiBackboneProbe(nn.Module):
    """Projects heterogeneous vision backbones into an isometric space and fuses via learned attention gates."""

    def __init__(
        self,
        vis_dims: List[int],
        clin_dim: int = 12,
        proj_dim: int = 128,
        clin_embed_dim: int = 32,
        hidden_dim: int = 128,
        num_subtypes: int = 6,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.num_experts = len(vis_dims)
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Dropout(dropout),
                    nn.Linear(d, proj_dim),
                    nn.SiLU(),
                    nn.LayerNorm(proj_dim),
                )
                for d in vis_dims
            ]
        )
        # Gating network taking concatenated projected experts -> softmax weights
        self.gate_net = nn.Sequential(
            nn.Linear(proj_dim * self.num_experts, 64),
            nn.SiLU(),
            nn.Linear(64, self.num_experts),
        )
        self.clin_encoder = nn.Sequential(
            nn.BatchNorm1d(clin_dim),
            nn.Linear(clin_dim, clin_embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(clin_embed_dim, clin_embed_dim),
            nn.SiLU(),
            nn.BatchNorm1d(clin_embed_dim),
        )
        fused_dim = proj_dim + clin_embed_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.triage_head = nn.Linear(hidden_dim, 2)
        self.subtype_head = nn.Linear(hidden_dim, num_subtypes)

    def forward(self, vis_list: List[torch.Tensor], clin_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        proj_experts = [proj(v) for proj, v in zip(self.projectors, vis_list)]
        concat_proj = torch.cat(proj_experts, dim=-1)
        gate_weights = F.softmax(self.gate_net(concat_proj), dim=-1)  # (B, E)
        stacked = torch.stack(proj_experts, dim=1)  # (B, E, proj_dim)
        z_vis = torch.sum(stacked * gate_weights.unsqueeze(-1), dim=1)  # (B, proj_dim)

        z_clin = self.clin_encoder(clin_feat)
        z_fused = torch.cat([z_vis, z_clin], dim=-1)
        h = self.trunk(z_fused)
        return self.triage_head(h), self.subtype_head(h)


def train_and_predict_specialist(
    tr_vis: torch.Tensor,
    tr_clin: torch.Tensor,
    tr_t: torch.Tensor,
    tr_s: torch.Tensor,
    ev_vis: torch.Tensor,
    ev_clin: torch.Tensor,
    params: Dict[str, Any],
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a single specialist probe with its Optuna-optimal parameters and return predicted probabilities."""
    set_seed(seed)
    vis_dim = tr_vis.shape[1]
    weights_s = compute_powered_weights(tr_s.numpy(), 6, params["class_weight_power"], device)
    weights_t = compute_powered_weights(tr_t.numpy(), 2, params["class_weight_power"], device)

    train_loader = DataLoader(
        TensorDataset(tr_vis, tr_clin, tr_t, tr_s),
        batch_size=params["batch_size"],
        shuffle=True,
    )
    eval_loader = DataLoader(TensorDataset(ev_vis, ev_clin), batch_size=256, shuffle=False)

    probe = ConfigurableMultimodalProbe(
        vis_dim=vis_dim,
        clin_dim=12,
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

    for _ in range(60):
        probe.train()
        for vx, cx, tt, st in train_loader:
            vx, cx, tt, st = vx.to(device), cx.to(device), tt.to(device), st.to(device)
            optimizer.zero_grad()
            tl, sl = probe(vx, cx)
            loss, _, _ = loss_fn(tl, sl, tt, st)
            loss.backward()
            optimizer.step()
        scheduler.step()

    probe.eval()
    tp_list, sp_list = [], []
    with torch.no_grad():
        for vx, cx in eval_loader:
            vx, cx = vx.to(device), cx.to(device)
            tl, sl = probe(vx, cx)
            tp_list.append(torch.softmax(tl, dim=1).cpu())
            sp_list.append(torch.softmax(sl, dim=1).cpu())

    return torch.cat(tp_list, dim=0).numpy(), torch.cat(sp_list, dim=0).numpy()


def train_and_predict_gated_fusion(
    tr_vis_list: List[torch.Tensor],
    tr_clin: torch.Tensor,
    tr_t: torch.Tensor,
    tr_s: torch.Tensor,
    ev_vis_list: List[torch.Tensor],
    ev_clin: torch.Tensor,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train Gated Multi-Backbone Attention Fusion model and return predictions + mean gate weights."""
    set_seed(seed)
    vis_dims = [v.shape[1] for v in tr_vis_list]
    weights_s = compute_powered_weights(tr_s.numpy(), 6, 0.75, device)
    weights_t = compute_powered_weights(tr_t.numpy(), 2, 0.75, device)

    n_train = tr_clin.shape[0]
    indices = torch.arange(n_train)
    train_loader = DataLoader(TensorDataset(indices), batch_size=32, shuffle=True)

    model = GatedMultiBackboneProbe(
        vis_dims=vis_dims,
        clin_dim=12,
        proj_dim=128,
        clin_embed_dim=32,
        hidden_dim=128,
        num_subtypes=6,
        dropout=0.20,
    ).to(device)

    loss_fn = MultiHeadLoss(
        triage_weight=1.0,
        subtype_weight=1.0,
        triage_criterion=MaskedClassWeightedFocalLoss(alpha=weights_t, gamma=2.5),
        subtype_criterion=MaskedClassWeightedFocalLoss(alpha=weights_s, gamma=2.5),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

    for _ in range(60):
        model.train()
        for (batch_idx,) in train_loader:
            v_batch = [v[batch_idx].to(device) for v in tr_vis_list]
            cx = tr_clin[batch_idx].to(device)
            tt = tr_t[batch_idx].to(device)
            st = tr_s[batch_idx].to(device)

            optimizer.zero_grad()
            tl, sl = model(v_batch, cx)
            loss, _, _ = loss_fn(tl, sl, tt, st)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        v_ev = [v.to(device) for v in ev_vis_list]
        cx_ev = ev_clin.to(device)
        tl, sl = model(v_ev, cx_ev)
        tp = torch.softmax(tl, dim=1).cpu().numpy()
        sp = torch.softmax(sl, dim=1).cpu().numpy()

        # Also inspect learned gate weights on validation
        proj_experts = [proj(v) for proj, v in zip(model.projectors, v_ev)]
        gate_w = F.softmax(model.gate_net(torch.cat(proj_experts, dim=-1)), dim=-1).cpu().numpy()
        mean_gates = np.mean(gate_w, axis=0)

    return tp, sp, mean_gates


def main():
    print("=" * 100)
    print(" SkinAI EXP-015: Multi-Backbone Fusion Study (DINOv2 + BiomedCLIP + ConvNeXt-Tiny)")
    print(" Comparing Early Concatenation vs. Gated Attention Bottleneck vs. Late Probability Fusion")
    print("=" * 100)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    splits_dir = PROJECT_ROOT / "data" / "splits" / "development_pool_v2"
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")

    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
    strat_labels = train_df["subtype_target"].fillna("unmapped").astype(str).values
    groups = train_df["patient_id"].astype(str).values
    inner_train_idx, inner_val_idx = next(sgkf.split(train_df, strat_labels, groups))

    train_clin = torch.tensor(
        np.stack([extract_clinical_vector(r) for _, r in train_df.iterrows()]),
        dtype=torch.float32,
    )
    val_clin = torch.tensor(
        np.stack([extract_clinical_vector(r) for _, r in val_df.iterrows()]),
        dtype=torch.float32,
    )
    train_triage = torch.tensor(train_df["triage_target"].values, dtype=torch.long)
    val_triage = torch.tensor(val_df["triage_target"].values, dtype=torch.long)
    train_subtype = torch.tensor([CLASS_TO_IDX.get(str(s), -1) for s in train_df["subtype_target"]], dtype=torch.long)
    val_subtype = torch.tensor([CLASS_TO_IDX.get(str(s), -1) for s in val_df["subtype_target"]], dtype=torch.long)

    cache_dir = PROJECT_ROOT / "artifacts" / "embeddings"
    tr_dino = torch.load(cache_dir / "dev_pool_v2_train_dinov2.pt", map_location="cpu", weights_only=True)
    va_dino = torch.load(cache_dir / "dev_pool_v2_val_dinov2.pt", map_location="cpu", weights_only=True)

    tr_biomed = torch.load(cache_dir / "dev_pool_v2_train_biomedclip.pt", map_location="cpu", weights_only=True)
    va_biomed = torch.load(cache_dir / "dev_pool_v2_val_biomedclip.pt", map_location="cpu", weights_only=True)

    tr_conv = torch.load(cache_dir / "dev_pool_v2_train_convnext_tiny.pt", map_location="cpu", weights_only=True)
    va_conv = torch.load(cache_dir / "dev_pool_v2_val_convnext_tiny.pt", map_location="cpu", weights_only=True)

    # Proven Optuna-optimal parameters from EXP-014
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
    params_conv = {
        "lr": 0.0005,
        "weight_decay": 0.001,
        "gamma": 2.5,
        "class_weight_power": 0.75,
        "dropout": 0.25,
        "hidden_dim": 256,
        "clin_embed_dim": 32,
        "batch_size": 64,
        "activation": "gelu",
    }
    params_concat = {
        "lr": 0.0003,
        "weight_decay": 1e-5,
        "gamma": 2.5,
        "class_weight_power": 0.75,
        "dropout": 0.35,
        "hidden_dim": 512,
        "clin_embed_dim": 32,
        "batch_size": 32,
        "activation": "silu",
    }

    # -------------------------------------------------------------------------
    # Step 1: Find Optimal Convex Ensemble Weights (w_dino, w_biomed, w_conv) Strictly on Inner-Val
    # -------------------------------------------------------------------------
    print("\n[*] Step 1: Calibrating Late Fusion Ensemble Weights Strictly on Inner-Val (570 images)...")
    itr_clin, iva_clin = train_clin[inner_train_idx], train_clin[inner_val_idx]
    itr_t, iva_t = train_triage[inner_train_idx], train_triage[inner_val_idx]
    itr_s, iva_s = train_subtype[inner_train_idx], train_subtype[inner_val_idx]

    tp_dino_iva, sp_dino_iva = train_and_predict_specialist(
        tr_dino[inner_train_idx], itr_clin, itr_t, itr_s, tr_dino[inner_val_idx], iva_clin, params_dino, 42, device
    )
    tp_bio_iva, sp_bio_iva = train_and_predict_specialist(
        tr_biomed[inner_train_idx], itr_clin, itr_t, itr_s, tr_biomed[inner_val_idx], iva_clin, params_biomed, 42, device
    )
    tp_conv_iva, sp_conv_iva = train_and_predict_specialist(
        tr_conv[inner_train_idx], itr_clin, itr_t, itr_s, tr_conv[inner_val_idx], iva_clin, params_conv, 42, device
    )

    best_weights_dual = (0.5, 0.5)
    best_score_dual = -1.0
    for w_d in np.linspace(0.1, 0.9, 17):
        w_b = 1.0 - w_d
        tp_mix = w_d * tp_dino_iva + w_b * tp_bio_iva
        sp_mix = w_d * sp_dino_iva + w_b * sp_bio_iva
        m = evaluate_multimodal_predictions(tp_mix, sp_mix, iva_s.numpy())
        score = 0.65 * m["macro_f1"] + 0.20 * m["balanced_accuracy"] + 0.15 * m["mel_recall_tau_005"]
        if score > best_score_dual:
            best_score_dual = score
            best_weights_dual = (round(float(w_d), 2), round(float(w_b), 2))

    best_weights_tri = (0.5, 0.3, 0.2)
    best_score_tri = -1.0
    for w_d in np.linspace(0.2, 0.7, 11):
        for w_b in np.linspace(0.1, 0.6, 11):
            w_c = round(1.0 - w_d - w_b, 4)
            if w_c < 0.05 or w_c > 0.5:
                continue
            tp_mix = w_d * tp_dino_iva + w_b * tp_bio_iva + w_c * tp_conv_iva
            sp_mix = w_d * sp_dino_iva + w_b * sp_bio_iva + w_c * sp_conv_iva
            m = evaluate_multimodal_predictions(tp_mix, sp_mix, iva_s.numpy())
            score = 0.65 * m["macro_f1"] + 0.20 * m["balanced_accuracy"] + 0.15 * m["mel_recall_tau_005"]
            if score > best_score_tri:
                best_score_tri = score
                best_weights_tri = (round(float(w_d), 2), round(float(w_b), 2), round(float(w_c), 2))

    print(f"    [+] Inner-Val Optimal Dual Weights (DINOv2, BiomedCLIP): {best_weights_dual}")
    print(f"    [+] Inner-Val Optimal Tri Weights (DINOv2, BiomedCLIP, ConvNeXt): {best_weights_tri}")

    # -------------------------------------------------------------------------
    # Step 2: Evaluate All 9 Configurations Across 5 Seeds on Hold-out Val (757 images)
    # -------------------------------------------------------------------------
    print("\n[*] Step 2: Evaluating All Single, Early-Concat, Gated-Attention, and Late-Fusion Models (5 seeds)...")

    tr_concat_2 = torch.cat([tr_dino, tr_biomed], dim=-1)
    va_concat_2 = torch.cat([va_dino, va_biomed], dim=-1)

    tr_concat_3 = torch.cat([tr_dino, tr_biomed, tr_conv], dim=-1)
    va_concat_3 = torch.cat([va_dino, va_biomed, va_conv], dim=-1)

    configs_seeds: Dict[str, List[Dict[str, float]]] = {
        "1A_DINOv2_Specialist": [],
        "1B_BiomedCLIP_Specialist": [],
        "1C_ConvNeXt_Specialist": [],
        "2A_EarlyConcat_DINOv2_BiomedCLIP": [],
        "2B_EarlyConcat_DINOv2_BiomedCLIP_ConvNeXt": [],
        "3A_GatedAttention_DINOv2_BiomedCLIP": [],
        "3B_GatedAttention_DINOv2_BiomedCLIP_ConvNeXt": [],
        "4A_LateConsilium_Equal_DINOv2_BiomedCLIP": [],
        "4B_LateConsilium_OptWeighted_DINOv2_BiomedCLIP": [],
        "4C_LateConsilium_OptWeighted_Tri_DINOv2_Biomed_ConvNeXt": [],
    }
    gate_weights_2_log = []
    gate_weights_3_log = []

    y_val_np = val_subtype.numpy()

    for seed in SEEDS:
        print(f"    -> Running Seed {seed}...")
        # 1. Specialists
        tp_d, sp_d = train_and_predict_specialist(
            tr_dino, train_clin, train_triage, train_subtype, va_dino, val_clin, params_dino, seed, device
        )
        tp_b, sp_b = train_and_predict_specialist(
            tr_biomed, train_clin, train_triage, train_subtype, va_biomed, val_clin, params_biomed, seed, device
        )
        tp_c, sp_c = train_and_predict_specialist(
            tr_conv, train_clin, train_triage, train_subtype, va_conv, val_clin, params_conv, seed, device
        )

        configs_seeds["1A_DINOv2_Specialist"].append(evaluate_multimodal_predictions(tp_d, sp_d, y_val_np))
        configs_seeds["1B_BiomedCLIP_Specialist"].append(evaluate_multimodal_predictions(tp_b, sp_b, y_val_np))
        configs_seeds["1C_ConvNeXt_Specialist"].append(evaluate_multimodal_predictions(tp_c, sp_c, y_val_np))

        # 2. Early Feature Concatenation
        tp_ec2, sp_ec2 = train_and_predict_specialist(
            tr_concat_2, train_clin, train_triage, train_subtype, va_concat_2, val_clin, params_concat, seed, device
        )
        tp_ec3, sp_ec3 = train_and_predict_specialist(
            tr_concat_3, train_clin, train_triage, train_subtype, va_concat_3, val_clin, params_concat, seed, device
        )
        configs_seeds["2A_EarlyConcat_DINOv2_BiomedCLIP"].append(
            evaluate_multimodal_predictions(tp_ec2, sp_ec2, y_val_np)
        )
        configs_seeds["2B_EarlyConcat_DINOv2_BiomedCLIP_ConvNeXt"].append(
            evaluate_multimodal_predictions(tp_ec3, sp_ec3, y_val_np)
        )

        # 3. Gated Attention Bottleneck Fusion
        tp_ga2, sp_ga2, gw2 = train_and_predict_gated_fusion(
            [tr_dino, tr_biomed], train_clin, train_triage, train_subtype, [va_dino, va_biomed], val_clin, seed, device
        )
        tp_ga3, sp_ga3, gw3 = train_and_predict_gated_fusion(
            [tr_dino, tr_biomed, tr_conv],
            train_clin,
            train_triage,
            train_subtype,
            [va_dino, va_biomed, va_conv],
            val_clin,
            seed,
            device,
        )
        gate_weights_2_log.append(gw2.tolist())
        gate_weights_3_log.append(gw3.tolist())
        configs_seeds["3A_GatedAttention_DINOv2_BiomedCLIP"].append(
            evaluate_multimodal_predictions(tp_ga2, sp_ga2, y_val_np)
        )
        configs_seeds["3B_GatedAttention_DINOv2_BiomedCLIP_ConvNeXt"].append(
            evaluate_multimodal_predictions(tp_ga3, sp_ga3, y_val_np)
        )

        # 4. Late Probability Fusion Ensembles
        # 4A. Equal 50/50 Dual
        tp_lc_eq = 0.50 * tp_d + 0.50 * tp_b
        sp_lc_eq = 0.50 * sp_d + 0.50 * sp_b
        configs_seeds["4A_LateConsilium_Equal_DINOv2_BiomedCLIP"].append(
            evaluate_multimodal_predictions(tp_lc_eq, sp_lc_eq, y_val_np)
        )

        # 4B. Inner-Val Optimized Dual
        wd2, wb2 = best_weights_dual
        tp_lc_opt2 = wd2 * tp_d + wb2 * tp_b
        sp_lc_opt2 = wd2 * sp_d + wb2 * sp_b
        configs_seeds["4B_LateConsilium_OptWeighted_DINOv2_BiomedCLIP"].append(
            evaluate_multimodal_predictions(tp_lc_opt2, sp_lc_opt2, y_val_np)
        )

        # 4C. Inner-Val Optimized Tri-Fusion (DINOv2 + BiomedCLIP + ConvNeXt)
        wd3, wb3, wc3 = best_weights_tri
        tp_lc_opt3 = wd3 * tp_d + wb3 * tp_b + wc3 * tp_c
        sp_lc_opt3 = wd3 * sp_d + wb3 * sp_b + wc3 * sp_c
        configs_seeds["4C_LateConsilium_OptWeighted_Tri_DINOv2_Biomed_ConvNeXt"].append(
            evaluate_multimodal_predictions(tp_lc_opt3, sp_lc_opt3, y_val_np)
        )

    # Summarize all results
    summary_report: Dict[str, Any] = {}
    print("\n" + "=" * 120)
    print(" EXP-015 MULTI-BACKBONE Fusion RESULTS (HOLD-OUT VAL, 5 SEEDS)")
    print("=" * 120)
    print(
        f"{'Configuration':<52} | {'Macro F1':<15} | {'Bal Acc':<13} | {'ROC-AUC':<8} | "
        f"{'PR-AUC':<8} | {'Mel Recall (.05)':<16}"
    )
    print("-" * 120)

    for key, s_list in configs_seeds.items():
        agg = aggregate_seed_metrics(s_list)
        summary_report[key] = agg
        f1_s = f"{agg['macro_f1']['mean']:.4f}±{agg['macro_f1']['std']:.4f}"
        ba_s = f"{agg['balanced_accuracy']['mean']*100:.2f}%±{agg['balanced_accuracy']['std']*100:.2f}%"
        roc_s = f"{agg['macro_roc_auc']['mean']:.4f}"
        pr_s = f"{agg['macro_pr_auc']['mean']:.4f}"
        mel_s = f"{agg['mel_hits_mean']:.1f}/29 ({agg['mel_recall_tau_005']['mean']*100:.1f}%)"
        print(f"{key:<52} | {f1_s:<15} | {ba_s:<13} | {roc_s:<8} | {pr_s:<8} | {mel_s:<16}")

    print("=" * 120)
    mean_gw2 = np.mean(gate_weights_2_log, axis=0)
    mean_gw3 = np.mean(gate_weights_3_log, axis=0)
    print(f"[*] Learned Gated Attention Weights (Dual: DINOv2, BiomedCLIP): {[round(float(x), 3) for x in mean_gw2]}")
    print(
        f"[*] Learned Gated Attention Weights (Tri: DINOv2, BiomedCLIP, ConvNeXt): {[round(float(x), 3) for x in mean_gw3]}"
    )

    out_path = PROJECT_ROOT / "artifacts" / "experiments" / "exp015_multi_backbone_fusion.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": "EXP-015",
                "inner_val_calibrated_weights": {
                    "dual_dinov2_biomedclip": best_weights_dual,
                    "tri_dinov2_biomedclip_convnext": best_weights_tri,
                },
                "learned_attention_gates": {
                    "dual_mean_gates": [round(float(x), 4) for x in mean_gw2],
                    "tri_mean_gates": [round(float(x), 4) for x in mean_gw3],
                },
                "configurations": summary_report,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[+] Saved full EXP-015 Fusion benchmark to: {out_path}")


if __name__ == "__main__":
    main()

