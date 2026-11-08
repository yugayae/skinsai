#!/usr/bin/env python3
"""EXP-014: Rigorous Hyperparameter Justification via 1D Sensitivity Sweeps & Optuna Bayesian Optimization.

Purpose:
Provide mathematical and empirical proof for EVERY hyperparameter and threshold chosen in SkinAI,
so that no parameter is arbitrary ("magic number").

Methodology (Strict Zero-Leakage Protocol):
- All hyperparameter sweeps and Optuna trials are executed strictly on the patient-grouped
  Inner Split of train.csv (Inner-Train: 2,440 images / 1,115 patients -> Inner-Val: 570 images / 224 patients).
- The official Hold-out Validation set (val.csv: 757 images, 335 patients, 29 melanomas) is NEVER
  seen during grid sweeps or Optuna search; it is used only at the end to verify generalization.

What is Proven Empirically:
1. Optimizer choice: AdamW vs. SGD+Momentum vs. RMSprop (proving section 11.0.2 of diploma standard)
2. Activation function in Fusion Trunk: GELU vs. SiLU (Swish) vs. ReLU
3. Learning Rate (lr): [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 1e-2]
4. Weight Decay (wd): [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
5. Focal Loss Focusing Parameter (gamma): [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
6. Class-Weighting Exponent (power): [0.0 (unweighted), 0.5 (sqrt), 0.75, 1.0 (inverse freq), 1.25]
7. Dropout Rate (dropout): [0.0, 0.1, 0.2, 0.25, 0.35, 0.5]
8. Hidden Trunk Dimension (hidden_dim): [64, 128, 256, 512]
9. Clinical Embedding Dimension (clin_embed_dim): [16, 32, 64, 128]
10. Batch Size (batch_size): [16, 32, 64, 128, 256]
11. Training Epochs (epochs): [20, 40, 60, 80, 100]
12. Clinical Safety Thresholds:
    - Melanoma referral threshold tau_mel: [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    - Malignancy triage threshold tau_mal: [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
13. Optuna Bayesian Optimization (TPE Sampler, 50 trials per model) across Top-3 families
    (DINOv2 Small, BiomedCLIP, ConvNeXt-Tiny) + Multi-Backbone Ensemble (DINOv2 + BiomedCLIP).
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

optuna.logging.set_verbosity(optuna.logging.WARNING)
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
    evaluate_multimodal_predictions,
    extract_clinical_vector,
    set_seed,
    wilson_score_interval,
)

SWEEP_SEEDS = [0, 42, 2026]
FINAL_SEEDS = [0, 8, 27, 42, 2026]


def get_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


class ConfigurableMultimodalProbe(nn.Module):
    """Parameterized multimodal fusion head allowing ablation of activations and dimensions."""

    def __init__(
        self,
        vis_dim: int = 384,
        clin_dim: int = 12,
        clin_embed_dim: int = 64,
        hidden_dim: int = 256,
        num_subtypes: int = 6,
        dropout: float = 0.25,
        activation: str = "gelu",
    ):
        super().__init__()
        self.clin_encoder = nn.Sequential(
            nn.BatchNorm1d(clin_dim),
            nn.Linear(clin_dim, clin_embed_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(clin_embed_dim, clin_embed_dim),
            get_activation(activation),
            nn.BatchNorm1d(clin_embed_dim),
        )
        fused_dim = vis_dim + clin_embed_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim),
            get_activation(activation),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.triage_head = nn.Linear(hidden_dim, 2)
        self.subtype_head = nn.Linear(hidden_dim, num_subtypes)

    def forward(self, vis_feat: torch.Tensor, clin_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_clin = self.clin_encoder(clin_feat)
        z_fused = torch.cat([vis_feat, z_clin], dim=-1)
        h = self.trunk(z_fused)
        return self.triage_head(h), self.subtype_head(h)


def compute_powered_weights(targets: np.ndarray, num_classes: int, power: float, device: torch.device) -> torch.Tensor:
    valid = targets[targets >= 0]
    counts = Counter(valid)
    n_total = len(valid)
    raw = [((n_total / (num_classes * max(counts[i], 1))) ** power) for i in range(num_classes)]
    mean_w = float(np.mean(raw))
    normed = [w / max(mean_w, 1e-6) for w in raw]
    return torch.tensor(normed, dtype=torch.float32, device=device)


def run_single_config_on_split(
    itr_vis: torch.Tensor,
    itr_clin: torch.Tensor,
    itr_t: torch.Tensor,
    itr_s: torch.Tensor,
    eval_vis: torch.Tensor,
    eval_clin: torch.Tensor,
    eval_t: torch.Tensor,
    eval_s: torch.Tensor,
    device: torch.device,
    seeds: List[int],
    *,
    optimizer_name: str = "adamw",
    activation: str = "gelu",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gamma: float = 2.0,
    class_weight_power: float = 1.0,
    dropout: float = 0.25,
    hidden_dim: int = 256,
    clin_embed_dim: int = 64,
    batch_size: int = 64,
    epochs: int = 60,
    triage_loss_weight: float = 1.0,
) -> Dict[str, Any]:
    """Train and evaluate a specific hyperparameter configuration across given seeds."""
    vis_dim = itr_vis.shape[1]
    weights_s = compute_powered_weights(itr_s.numpy(), 6, class_weight_power, device)
    weights_t = compute_powered_weights(itr_t.numpy(), 2, class_weight_power, device)

    train_ds = TensorDataset(itr_vis, itr_clin, itr_t, itr_s)
    eval_ds = TensorDataset(eval_vis, eval_clin, eval_t, eval_s)
    eval_loader = DataLoader(eval_ds, batch_size=256, shuffle=False)

    seed_metrics = []
    for seed in seeds:
        set_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        probe = ConfigurableMultimodalProbe(
            vis_dim=vis_dim,
            clin_dim=12,
            clin_embed_dim=clin_embed_dim,
            hidden_dim=hidden_dim,
            num_subtypes=6,
            dropout=dropout,
            activation=activation,
        ).to(device)

        loss_fn = MultiHeadLoss(
            triage_weight=triage_loss_weight,
            subtype_weight=1.0,
            triage_criterion=MaskedClassWeightedFocalLoss(alpha=weights_t, gamma=gamma),
            subtype_criterion=MaskedClassWeightedFocalLoss(alpha=weights_s, gamma=gamma),
        )

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(probe.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for _ in range(epochs):
            probe.train()
            for vx, cx, t_tgt, s_tgt in train_loader:
                vx, cx, t_tgt, s_tgt = vx.to(device), cx.to(device), t_tgt.to(device), s_tgt.to(device)
                optimizer.zero_grad()
                t_l, s_l = probe(vx, cx)
                loss, _, _ = loss_fn(t_l, s_l, t_tgt, s_tgt)
                loss.backward()
                optimizer.step()
            scheduler.step()

        probe.eval()
        t_probs_list, s_probs_list = [], []
        with torch.no_grad():
            for vx, cx, _, _ in eval_loader:
                vx, cx = vx.to(device), cx.to(device)
                t_l, s_l = probe(vx, cx)
                t_probs_list.append(torch.softmax(t_l, dim=1).cpu())
                s_probs_list.append(torch.softmax(s_l, dim=1).cpu())

        t_probs = torch.cat(t_probs_list, dim=0).numpy()
        s_probs = torch.cat(s_probs_list, dim=0).numpy()
        m = evaluate_multimodal_predictions(t_probs, s_probs, eval_s.numpy())
        seed_metrics.append(m)

    return {
        "macro_f1_mean": round(float(np.mean([x["macro_f1"] for x in seed_metrics])), 4),
        "macro_f1_std": round(float(np.std([x["macro_f1"] for x in seed_metrics])), 4),
        "balanced_acc_mean": round(float(np.mean([x["balanced_accuracy"] for x in seed_metrics])), 4),
        "macro_roc_auc_mean": round(float(np.mean([x["macro_roc_auc"] for x in seed_metrics])), 4),
        "macro_pr_auc_mean": round(float(np.mean([x["macro_pr_auc"] for x in seed_metrics])), 4),
        "mel_recall_naive_mean": round(float(np.mean([x["mel_recall_naive"] for x in seed_metrics])), 4),
        "mel_recall_tau_005_mean": round(float(np.mean([x["mel_recall_tau_005"] for x in seed_metrics])), 4),
        "mel_spec_tau_005_mean": round(float(np.mean([x["mel_spec_tau_005"] for x in seed_metrics])), 4),
        "consensus_safety_recall_mean": round(float(np.mean([x["consensus_safety_recall"] for x in seed_metrics])), 4),
        "consensus_safety_spec_mean": round(float(np.mean([x["consensus_safety_spec"] for x in seed_metrics])), 4),
    }


def evaluate_threshold_sweeps(
    itr_vis: torch.Tensor,
    itr_clin: torch.Tensor,
    itr_t: torch.Tensor,
    itr_s: torch.Tensor,
    iva_vis: torch.Tensor,
    iva_clin: torch.Tensor,
    iva_t: torch.Tensor,
    iva_s: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Prove exact choice of tau_mel=0.05 and tau_mal=0.20 on Inner-Val."""
    set_seed(42)
    vis_dim = itr_vis.shape[1]
    weights_s = compute_powered_weights(itr_s.numpy(), 6, 1.0, device)
    weights_t = compute_powered_weights(itr_t.numpy(), 2, 1.0, device)
    train_loader = DataLoader(TensorDataset(itr_vis, itr_clin, itr_t, itr_s), batch_size=64, shuffle=True)

    probe = ConfigurableMultimodalProbe(vis_dim=vis_dim).to(device)
    loss_fn = MultiHeadLoss(
        triage_weight=1.0,
        subtype_weight=1.0,
        triage_criterion=MaskedClassWeightedFocalLoss(alpha=weights_t, gamma=2.0),
        subtype_criterion=MaskedClassWeightedFocalLoss(alpha=weights_s, gamma=2.0),
    )
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-5)

    for _ in range(60):
        probe.train()
        for vx, cx, tt, st in train_loader:
            vx, cx, tt, st = vx.to(device), cx.to(device), tt.to(device), st.to(device)
            opt.zero_grad()
            tl, sl = probe(vx, cx)
            loss, _, _ = loss_fn(tl, sl, tt, st)
            loss.backward()
            opt.step()
        sched.step()

    probe.eval()
    with torch.no_grad():
        tl, sl = probe(iva_vis.to(device), iva_clin.to(device))
        t_probs = torch.softmax(tl, dim=1).cpu().numpy()
        s_probs = torch.softmax(sl, dim=1).cpu().numpy()

    y_s = iva_s.numpy()
    valid = y_s >= 0
    y_v = y_s[valid]
    sp_v = s_probs[valid]
    tp_v = t_probs[valid]

    mel_idx = CLASS_TO_IDX["mel"]
    mel_mask = y_v == mel_idx
    mel_probs = sp_v[:, mel_idx]

    tau_mel_sweep = {}
    for tau in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        rec = float(np.sum((mel_probs >= tau) & mel_mask) / max(np.sum(mel_mask), 1))
        spec = float(np.sum((mel_probs < tau) & (~mel_mask)) / max(np.sum(~mel_mask), 1))
        f2 = (5 * rec * spec) / max(4 * spec + rec, 1e-6)  # safety-weighted F2 harmonic score
        tau_mel_sweep[f"{tau:.2f}"] = {
            "mel_recall": round(rec, 4),
            "mel_specificity": round(spec, 4),
            "clinical_f2_utility": round(f2, 4),
        }

    mal_mask = np.isin(y_v, MALIGNANT_INDICES)
    p_mal = tp_v[:, 1]
    tau_mal_sweep = {}
    for tau in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        rec = float(np.sum((p_mal >= tau) & mal_mask) / max(np.sum(mal_mask), 1))
        spec = float(np.sum((p_mal < tau) & (~mal_mask)) / max(np.sum(~mal_mask), 1))
        tau_mal_sweep[f"{tau:.2f}"] = {
            "malignancy_recall": round(rec, 4),
            "malignancy_specificity": round(spec, 4),
            "youden_j": round(rec + spec - 1.0, 4),
        }

    return {"tau_mel_sweep_inner_val": tau_mel_sweep, "tau_mal_sweep_inner_val": tau_mal_sweep}


def run_optuna_study_for_backbone(
    arch_title: str,
    itr_vis: torch.Tensor,
    itr_clin: torch.Tensor,
    itr_t: torch.Tensor,
    itr_s: torch.Tensor,
    iva_vis: torch.Tensor,
    iva_clin: torch.Tensor,
    iva_t: torch.Tensor,
    iva_s: torch.Tensor,
    full_tr_vis: torch.Tensor,
    full_tr_clin: torch.Tensor,
    full_tr_t: torch.Tensor,
    full_tr_s: torch.Tensor,
    val_vis: torch.Tensor,
    val_clin: torch.Tensor,
    val_t: torch.Tensor,
    val_s: torch.Tensor,
    device: torch.device,
    n_trials: int = 40,
) -> Dict[str, Any]:
    """Run Bayesian Optimization (Optuna TPE) strictly on Inner-Train -> Inner-Val, then evaluate on Hold-out Val."""
    print(f"\n[*] Running Optuna Bayesian Optimization ({n_trials} trials) for {arch_title} on Inner-Val...")
    t0 = time.time()

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_Categorical("lr", [3e-4, 5e-4, 1e-3, 2e-3, 3e-3]) if hasattr(trial, "suggest_Categorical") else trial.suggest_categorical("lr", [3e-4, 5e-4, 1e-3, 2e-3, 3e-3])
        weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 5e-4, 1e-3])
        gamma = trial.suggest_categorical("gamma", [1.0, 1.5, 2.0, 2.5, 3.0])
        class_weight_power = trial.suggest_categorical("class_weight_power", [0.5, 0.75, 1.0])
        dropout = trial.suggest_categorical("dropout", [0.15, 0.20, 0.25, 0.35])
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        clin_embed_dim = trial.suggest_categorical("clin_embed_dim", [32, 64, 128])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        activation = trial.suggest_categorical("activation", ["gelu", "silu"])

        res = run_single_config_on_split(
            itr_vis,
            itr_clin,
            itr_t,
            itr_s,
            iva_vis,
            iva_clin,
            iva_t,
            iva_s,
            device,
            seeds=[42],
            lr=lr,
            weight_decay=weight_decay,
            gamma=gamma,
            class_weight_power=class_weight_power,
            dropout=dropout,
            hidden_dim=hidden_dim,
            clin_embed_dim=clin_embed_dim,
            batch_size=batch_size,
            epochs=50,
            activation=activation,
        )
        # Composite clinical objective: 0.70 * Macro_F1 + 0.20 * Balanced_Acc + 0.10 * Mel_Recall_005
        score = (
            0.70 * res["macro_f1_mean"]
            + 0.20 * res["balanced_acc_mean"]
            + 0.10 * res["mel_recall_tau_005_mean"]
        )
        return float(score)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # Enqueue our canonical configuration as Trial #0 baseline reference
    study.enqueue_trial(
        {
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "gamma": 2.0,
            "class_weight_power": 1.0,
            "dropout": 0.25,
            "hidden_dim": 256,
            "clin_embed_dim": 64,
            "batch_size": 64,
            "activation": "gelu",
        }
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    print(f"    [+] Optuna finished in {time.time() - t0:.1f}s | Best Inner-Val Score: {study.best_value:.4f}")
    print(f"        Best Params: {best_params}")

    # Compute hyperparameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        importances = {k: round(float(v), 4) for k, v in importances.items()}
    except Exception:
        importances = {}

    # Evaluate Optuna Best Config on untouched Hold-out Val (5 seeds)
    holdout_eval = run_single_config_on_split(
        full_tr_vis,
        full_tr_clin,
        full_tr_t,
        full_tr_s,
        val_vis,
        val_clin,
        val_t,
        val_s,
        device,
        seeds=FINAL_SEEDS,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        gamma=best_params["gamma"],
        class_weight_power=best_params["class_weight_power"],
        dropout=best_params["dropout"],
        hidden_dim=best_params["hidden_dim"],
        clin_embed_dim=best_params["clin_embed_dim"],
        batch_size=best_params["batch_size"],
        epochs=60,
        activation=best_params["activation"],
    )
    print(
        f"        Hold-out Val (5 seeds) -> Macro F1: {holdout_eval['macro_f1_mean']:.4f}±{holdout_eval['macro_f1_std']:.4f} | "
        f"Bal Acc: {holdout_eval['balanced_acc_mean']*100:.2f}% | "
        f"Mel Recall (tau=0.05): {holdout_eval['mel_recall_tau_005_mean']*100:.1f}% | "
        f"Consensus Safety: {holdout_eval['consensus_safety_recall_mean']*100:.2f}%"
    )

    return {
        "n_trials": n_trials,
        "best_inner_val_objective": round(float(study.best_value), 4),
        "canonical_trial0_objective": round(float(study.trials[0].value), 4),
        "best_params": best_params,
        "param_importances": importances,
        "holdout_val_5seed_metrics": holdout_eval,
    }


def main():
    print("=" * 95)
    print(" SkinAI EXP-014: Complete Hyperparameter Proof & Optuna Bayesian Optimization")
    print(" Protocol: All Sweeps & Optuna Trials Run Strictly on Inner-Train (2,440) -> Inner-Val (570)")
    print("=" * 95)

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

    itr_vis, iva_vis = tr_dino[inner_train_idx], tr_dino[inner_val_idx]
    itr_clin, iva_clin = train_clin[inner_train_idx], train_clin[inner_val_idx]
    itr_t, iva_t = train_triage[inner_train_idx], train_triage[inner_val_idx]
    itr_s, iva_s = train_subtype[inner_train_idx], train_subtype[inner_val_idx]

    # -------------------------------------------------------------------------
    # Part 1: 1D Coordinate Sensitivity Sweeps on Inner-Val (Proving Every Parameter)
    # -------------------------------------------------------------------------
    print("\n[*] Part 1: Running 1D Sensitivity Sweeps on Inner-Val (3 seeds per point)...")
    sweeps: Dict[str, Dict[str, Any]] = {}

    def run_sweep(param_name: str, values: List[Any], **fixed_kwargs):
        print(f"    -> Sweeping '{param_name}': {values}")
        sweeps[param_name] = {}
        for val in values:
            kw = dict(fixed_kwargs)
            kw[param_name] = val
            res = run_single_config_on_split(
                itr_vis, itr_clin, itr_t, itr_s, iva_vis, iva_clin, iva_t, iva_s, device, SWEEP_SEEDS, **kw
            )
            sweeps[param_name][str(val)] = res
            print(
                f"       {param_name}={str(val):<8} | Inner-Val Macro F1={res['macro_f1_mean']:.4f} | "
                f"Bal Acc={res['balanced_acc_mean']*100:.2f}% | Mel Recall(0.05)={res['mel_recall_tau_005_mean']*100:.1f}%"
            )

    run_sweep("optimizer_name", ["adamw", "sgd", "rmsprop"])
    run_sweep("activation", ["gelu", "silu", "relu"])
    run_sweep("lr", [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 1e-2])
    run_sweep("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])
    run_sweep("gamma", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    run_sweep("class_weight_power", [0.0, 0.5, 0.75, 1.0, 1.25])
    run_sweep("dropout", [0.0, 0.1, 0.2, 0.25, 0.35, 0.5])
    run_sweep("hidden_dim", [64, 128, 256, 512])
    run_sweep("clin_embed_dim", [16, 32, 64, 128])
    run_sweep("batch_size", [16, 32, 64, 128, 256])
    run_sweep("epochs", [20, 40, 60, 80, 100])

    # -------------------------------------------------------------------------
    # Part 2: Clinical Threshold Operating Point Proofs (tau_mel and tau_mal)
    # -------------------------------------------------------------------------
    print("\n[*] Part 2: Evaluating Clinical Safety Threshold Operating Curves on Inner-Val...")
    threshold_proofs = evaluate_threshold_sweeps(
        itr_vis, itr_clin, itr_t, itr_s, iva_vis, iva_clin, iva_t, iva_s, device
    )
    for tau, m in threshold_proofs["tau_mel_sweep_inner_val"].items():
        print(
            f"    tau_mel={tau} -> Mel Recall={m['mel_recall']*100:.1f}%, "
            f"Mel Spec={m['mel_specificity']*100:.1f}%, F2 Utility={m['clinical_f2_utility']:.4f}"
        )

    # -------------------------------------------------------------------------
    # Part 3: Optuna Bayesian Optimization Across Top-3 Model Families + Fusion Ensemble
    # -------------------------------------------------------------------------
    tr_biomed = torch.load(cache_dir / "dev_pool_v2_train_biomedclip.pt", map_location="cpu", weights_only=True)
    va_biomed = torch.load(cache_dir / "dev_pool_v2_val_biomedclip.pt", map_location="cpu", weights_only=True)

    tr_conv = torch.load(cache_dir / "dev_pool_v2_train_convnext_tiny.pt", map_location="cpu", weights_only=True)
    va_conv = torch.load(cache_dir / "dev_pool_v2_val_convnext_tiny.pt", map_location="cpu", weights_only=True)

    # Dual Foundation Fusion (DINOv2 384D + BiomedCLIP 512D -> 896D)
    tr_consilium = torch.cat([tr_dino, tr_biomed], dim=-1)
    va_consilium = torch.cat([va_dino, va_biomed], dim=-1)

    optuna_studies = {}
    for key, title, tr_v, va_v in [
        ("dinov2_small", "Meta DINOv2 Small", tr_dino, va_dino),
        ("biomedclip", "Microsoft BiomedCLIP", tr_biomed, va_biomed),
        ("convnext_tiny", "ConvNeXt-Tiny", tr_conv, va_conv),
        ("consilium_dinov2_biomedclip", "Dual-Backbone Fusion (DINOv2 + BiomedCLIP)", tr_consilium, va_consilium),
    ]:
        optuna_studies[key] = run_optuna_study_for_backbone(
            title,
            tr_v[inner_train_idx],
            itr_clin,
            itr_t,
            itr_s,
            tr_v[inner_val_idx],
            iva_clin,
            iva_t,
            iva_s,
            tr_v,
            train_clin,
            train_triage,
            train_subtype,
            va_v,
            val_clin,
            val_triage,
            val_subtype,
            device,
            n_trials=40,
        )

    out_path = PROJECT_ROOT / "artifacts" / "experiments" / "exp014_optuna_hyperparameter_proofs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": "EXP-014",
                "protocol": "Strict Inner-Train (2,440) -> Inner-Val (570) Tuning; Hold-out Val (757) Verification",
                "one_factor_sensitivity_sweeps": sweeps,
                "clinical_threshold_proofs": threshold_proofs,
                "optuna_bayesian_studies": optuna_studies,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[+] Saved complete hyperparameter proofs & Optuna report to: {out_path}")
    print("=" * 95)


if __name__ == "__main__":
    main()

