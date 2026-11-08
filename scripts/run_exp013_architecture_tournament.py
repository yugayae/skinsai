#!/usr/bin/env python3
"""EXP-013: Multi-Architecture Tournament & Class-Imbalance Ablation on Development Pool v2.

Compares 6 distinct Computer Vision families under identical Multimodal Late-Fusion conditions:
1. EfficientNet-B0 (timm: efficientnet_b0, 224x224, 1280-D) — Compound-scaled CNN baseline
2. CSPDarknet-53 (timm: cspdarknet53, 256x256, 1024-D) — YOLO family Cross-Stage Partial backbone
3. ConvNeXt-Tiny (timm: convnext_tiny, 224x224, 768-D) — Modernized depthwise 7x7 CNN
4. Swin-Tiny (timm: swin_tiny_patch4_window7_224, 224x224, 768-D) — Hierarchical Windowed ViT
5. Microsoft BiomedCLIP (open_clip: ViT-B/16, 224x224, 512-D) — Biomedical Vision-Language model
6. Meta DINOv2 Small (timm: vit_small_patch14_dinov2, 518x518, 384-D) — Self-supervised ViT

Methodological Guarantees (Zero Validation Snooping):
- External Test Set (Stanford DDI) remains 100% locked and untouched.
- Protocol A (Inner-Val Strict Protocol): Splits train.csv (3,010 images) by patient_id into
  85% inner_train and 15% inner_val. Early stopping and Temperature Scaling (T) are tuned
  strictly on inner_val; val.csv (757 images, 335 patients, 29 melanomas) is used solely as a
  hold-out benchmark.
- Protocol B (Standard EXP-011 Fixed-60-Epoch Protocol): Trains on 100% of train.csv for 60 epochs
  for direct 1:1 comparability with EXP-011.
- Class-Imbalance Ablation Study: Evaluates Unweighted CE vs. Class-Weighted CE vs.
  MaskedClassWeightedFocalLoss (gamma=2.0) + Dual-Head Consensus + Threshold Calibration.
- Hardware Latency & FPS Benchmark: Measures real inference throughput on NVIDIA RTX 3060 GPU and CPU.
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
import open_clip
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

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
from ml.models.multimodal import MultimodalLateFusionProbe

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_ORDER)}
MALIGNANT_INDICES = [CLASS_TO_IDX[c] for c in MALIGNANT_CLASSES]
SEEDS = [0, 8, 27, 42, 2026]


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
    try:
        age_val = float(age) if age != "" and not pd.isna(age) else None
    except Exception:
        age_val = None

    if age_val is not None:
        vec.append((age_val - 50.0) / 25.0)
    else:
        vec.append(0.0)

    vec.append(1.0 if ("head" in site or "neck" in site) else 0.0)
    vec.append(1.0 if ("trunk" in site or "torso" in site) else 0.0)
    vec.append(1.0 if ("upper" in site or "arm" in site or "hand" in site) else 0.0)
    vec.append(1.0 if ("lower" in site or "leg" in site or "foot" in site) else 0.0)
    has_known_site = any(vec[-4:])
    vec.append(0.0 if has_known_site else 1.0)

    vec.append(1.0 if "female" in sex else 0.0)
    vec.append(1.0 if ("male" in sex and "female" not in sex) else 0.0)
    has_known_sex = vec[-2] == 1.0 or vec[-1] == 1.0
    vec.append(0.0 if has_known_sex else 1.0)

    vec.append(0.0)
    vec.append(1.0)
    vec.append(0.0)

    return np.array(vec, dtype=np.float32)


class ImageFileDataset(Dataset):
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


def wilson_score_interval(k: int, n: int) -> Tuple[float, float]:
    """Calculate exact 95% Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, float(centre - half_width)), min(1.0, float(centre + half_width))


def build_backbone_and_transform(arch_key: str, device: torch.device):
    """Instantiate pretrained backbone, transform, and metadata."""
    if arch_key == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0).to(device)
        input_res = 224
    elif arch_key == "cspdarknet53":
        model = timm.create_model("cspdarknet53", pretrained=True, num_classes=0).to(device)
        input_res = 256
    elif arch_key == "convnext_tiny":
        model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0).to(device)
        input_res = 224
    elif arch_key == "swin_tiny":
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0).to(device)
        input_res = 224
    elif arch_key == "dinov2_small":
        model = timm.create_model("vit_small_patch14_dinov2", pretrained=True, num_classes=0).to(device)
        input_res = 518
    elif arch_key == "biomedclip":
        model, preprocess = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        model = model.to(device)
        model.eval()
        return model, preprocess, 224, True
    else:
        raise ValueError(f"Unknown architecture: {arch_key}")

    model.eval()
    transform = transforms.Compose([
        transforms.Resize((input_res, input_res), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform, input_res, False


@torch.no_grad()
def benchmark_hardware_speed(
    model: nn.Module,
    input_res: int,
    is_biomedclip: bool,
    gpu_device: torch.device,
) -> Dict[str, float]:
    """Measure real GPU (RTX 3060) and CPU latency (ms) and throughput (FPS)."""
    dummy_b1 = torch.randn(1, 3, input_res, input_res, device=gpu_device)
    dummy_b16 = torch.randn(16, 3, input_res, input_res, device=gpu_device)

    def forward_fn(m, x):
        if is_biomedclip:
            return m.encode_image(x)
        return m(x)

    # Warmup GPU
    for _ in range(5):
        _ = forward_fn(model, dummy_b1)
    if gpu_device.type == "cuda":
        torch.cuda.synchronize()

    # GPU Single-Image Latency (30 iterations)
    t0 = time.perf_counter()
    n_iter_gpu = 30
    for _ in range(n_iter_gpu):
        _ = forward_fn(model, dummy_b1)
    if gpu_device.type == "cuda":
        torch.cuda.synchronize()
    gpu_single_ms = ((time.perf_counter() - t0) / n_iter_gpu) * 1000.0

    # GPU Batch-16 Throughput (15 batches = 240 frames)
    for _ in range(2):
        _ = forward_fn(model, dummy_b16)
    if gpu_device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_batch_gpu = 15
    for _ in range(n_batch_gpu):
        _ = forward_fn(model, dummy_b16)
    if gpu_device.type == "cuda":
        torch.cuda.synchronize()
    gpu_batch_fps = (n_batch_gpu * 16) / (time.perf_counter() - t0)

    # CPU Single-Image Latency (10 iterations)
    cpu_device = torch.device("cpu")
    model_cpu = model.to(cpu_device)
    dummy_cpu = torch.randn(1, 3, input_res, input_res, device=cpu_device)
    for _ in range(2):
        _ = forward_fn(model_cpu, dummy_cpu)
    t0 = time.perf_counter()
    n_iter_cpu = 8
    for _ in range(n_iter_cpu):
        _ = forward_fn(model_cpu, dummy_cpu)
    cpu_single_ms = ((time.perf_counter() - t0) / n_iter_cpu) * 1000.0
    cpu_fps = 1000.0 / max(cpu_single_ms, 1e-3)

    # Move back to GPU
    model.to(gpu_device)
    params_m = sum(p.numel() for p in (model.visual.parameters() if is_biomedclip else model.parameters())) / 1e6

    return {
        "params_millions": round(float(params_m), 2),
        "input_resolution": input_res,
        "gpu_latency_ms": round(float(gpu_single_ms), 2),
        "gpu_single_fps": round(float(1000.0 / max(gpu_single_ms, 1e-3)), 1),
        "gpu_batch16_fps": round(float(gpu_batch_fps), 1),
        "cpu_latency_ms": round(float(cpu_single_ms), 2),
        "cpu_fps": round(float(cpu_fps), 1),
    }


@torch.no_grad()
def get_or_extract_embeddings(
    arch_key: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Load cached embeddings or extract them and measure hardware speed."""
    cache_dir = PROJECT_ROOT / "artifacts" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if arch_key == "dinov2_small":
        train_cache = cache_dir / "dev_pool_v2_train_dinov2.pt"
        val_cache = cache_dir / "dev_pool_v2_val_dinov2.pt"
    else:
        train_cache = cache_dir / f"dev_pool_v2_train_{arch_key}.pt"
        val_cache = cache_dir / f"dev_pool_v2_val_{arch_key}.pt"

    speed_cache = cache_dir / f"speed_{arch_key}.json"

    if train_cache.exists() and val_cache.exists() and speed_cache.exists():
        print(f"    [*] Loading cached embeddings & speed profile for '{arch_key}'...")
        tr_emb = torch.load(train_cache, map_location="cpu", weights_only=True)
        va_emb = torch.load(val_cache, map_location="cpu", weights_only=True)
        with open(speed_cache, "r", encoding="utf-8") as f:
            speed_info = json.load(f)
        return tr_emb, va_emb, speed_info

    print(f"    [*] Instantiating '{arch_key}' on {device} for benchmark...")
    model, transform, input_res, is_biomedclip = build_backbone_and_transform(arch_key, device)

    print(f"    [*] Measuring RTX 3060 GPU & CPU inference speed for '{arch_key}' ({input_res}x{input_res})...")
    speed_info = benchmark_hardware_speed(model, input_res, is_biomedclip, device)
    with open(speed_cache, "w", encoding="utf-8") as f:
        json.dump(speed_info, f, indent=2)
    print(
        f"        -> Params: {speed_info['params_millions']}M | "
        f"GPU Latency: {speed_info['gpu_latency_ms']} ms ({speed_info['gpu_batch16_fps']} FPS batch) | "
        f"CPU Latency: {speed_info['cpu_latency_ms']} ms ({speed_info['cpu_fps']} FPS)"
    )

    def extract_split(df: pd.DataFrame, cache_p: Path, split_label: str) -> torch.Tensor:
        if cache_p.exists():
            return torch.load(cache_p, map_location="cpu", weights_only=True)
        print(f"    [*] Extracting '{arch_key}' embeddings for {split_label} ({len(df)} images)...")
        paths = [PROJECT_ROOT / r["file_path"] for _, r in df.iterrows()]
        ds = ImageFileDataset(paths, transform=transform)
        bs = 32 if input_res <= 256 else 16
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
        chunks = []
        t0 = time.time()
        for batch in loader:
            batch = batch.to(device)
            if is_biomedclip:
                feats = model.encode_image(batch)
            else:
                feats = model(batch)
                if isinstance(feats, (tuple, list)):
                    feats = feats[0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.cpu())
        emb = torch.cat(chunks, dim=0)
        torch.save(emb, cache_p)
        print(f"        [+] Saved {emb.shape} to {cache_p.name} in {time.time() - t0:.1f}s")
        return emb

    tr_emb = extract_split(train_df, train_cache, "Train")
    va_emb = extract_split(val_df, val_cache, "Validation")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return tr_emb, va_emb, speed_info


def evaluate_multimodal_predictions(
    triage_probs: np.ndarray,
    subtype_probs: np.ndarray,
    y_true_s: np.ndarray,
) -> Dict[str, float]:
    """Evaluate predictions strictly against ground truth 6-class targets."""
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

    # Melanoma metrics
    mel_idx = CLASS_TO_IDX["mel"]
    mel_mask = (y_true_v == mel_idx)
    mel_support = int(np.sum(mel_mask))
    mel_probs = s_probs_v[:, mel_idx]

    mel_hits_naive = int(np.sum((y_pred_s == mel_idx) & mel_mask))
    mel_recall_naive = float(mel_hits_naive / max(mel_support, 1))

    mel_hits_005 = int(np.sum((mel_probs >= 0.05) & mel_mask))
    mel_recall_005 = float(mel_hits_005 / max(mel_support, 1))
    mel_spec_005 = float(np.sum((mel_probs < 0.05) & (~mel_mask)) / max(np.sum(~mel_mask), 1))
    mel_ci_low, mel_ci_high = wilson_score_interval(mel_hits_005, mel_support)

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
        "mel_hits_naive": mel_hits_naive,
        "mel_recall_naive": mel_recall_naive,
        "mel_hits": mel_hits_005,
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


def aggregate_seed_metrics(seed_results: List[Dict[str, float]]) -> Dict[str, Any]:
    keys = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_roc_auc",
        "macro_pr_auc",
        "mel_recall_naive",
        "mel_recall_tau_005",
        "mel_spec_tau_005",
        "triage_roc_auc",
        "triage_recall_tau_020",
        "triage_spec_tau_020",
        "consensus_safety_recall",
        "consensus_safety_spec",
        "consensus_agreement_rate",
    ]
    summary: Dict[str, Any] = {}
    for k in keys:
        vals = [r[k] for r in seed_results]
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    mel_hits_mean = float(np.mean([r["mel_hits"] for r in seed_results]))
    mel_support = int(seed_results[0]["mel_support"])
    w_low, w_high = wilson_score_interval(int(round(mel_hits_mean)), mel_support)
    summary["mel_hits_mean"] = mel_hits_mean
    summary["mel_support"] = mel_support
    summary["mel_wilson_95_ci"] = [w_low, w_high]
    return summary


def run_protocol_b_full_train(
    train_vis: torch.Tensor,
    train_clin: torch.Tensor,
    train_triage: torch.Tensor,
    train_subtype: torch.Tensor,
    val_vis: torch.Tensor,
    val_clin: torch.Tensor,
    val_triage: torch.Tensor,
    val_subtype: torch.Tensor,
    weights_t_tensor: torch.Tensor,
    weights_s_tensor: torch.Tensor,
    device: torch.device,
    gamma: float = 2.0,
    use_class_weights: bool = True,
) -> Dict[str, Any]:
    """Protocol B: Standard 60-epoch training on 100% of train.csv (1:1 with EXP-011)."""
    vis_dim = train_vis.shape[1]
    train_ds = TensorDataset(train_vis, train_clin, train_triage, train_subtype)
    val_ds = TensorDataset(val_vis, val_clin, val_triage, val_subtype)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    t_w = weights_t_tensor.to(device) if use_class_weights else torch.ones(2, device=device)
    s_w = weights_s_tensor.to(device) if use_class_weights else torch.ones(6, device=device)

    seed_results = []
    for seed in SEEDS:
        set_seed(seed)
        probe = MultimodalLateFusionProbe(
            vis_dim=vis_dim,
            clin_dim=12,
            clin_embed_dim=64,
            hidden_dim=256,
            dropout=0.25,
            num_subtypes=6,
        ).to(device)

        loss_fn = MultiHeadLoss(
            triage_weight=1.0,
            subtype_weight=1.0,
            triage_criterion=MaskedClassWeightedFocalLoss(alpha=t_w, gamma=gamma),
            subtype_criterion=MaskedClassWeightedFocalLoss(alpha=s_w, gamma=gamma),
        )
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

        for _ in range(60):
            probe.train()
            for vx, cx, t_tgt, s_tgt in train_loader:
                vx, cx, t_tgt, s_tgt = vx.to(device), cx.to(device), t_tgt.to(device), s_tgt.to(device)
                optimizer.zero_grad()
                t_logits, s_logits = probe(vx, cx)
                loss, _, _ = loss_fn(t_logits, s_logits, t_tgt, s_tgt)
                loss.backward()
                optimizer.step()
            scheduler.step()

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

    return {"summary": aggregate_seed_metrics(seed_results), "seeds": seed_results}


def run_protocol_a_inner_split(
    train_vis: torch.Tensor,
    train_clin: torch.Tensor,
    train_triage: torch.Tensor,
    train_subtype: torch.Tensor,
    inner_train_idx: np.ndarray,
    inner_val_idx: np.ndarray,
    val_vis: torch.Tensor,
    val_clin: torch.Tensor,
    val_triage: torch.Tensor,
    val_subtype: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Protocol A: Strict Zero-Snooping Protocol.
    Epoch selection (Early Stopping) and Temperature Calibration (T) are tuned ONLY on inner_val.
    val.csv (757 images) is evaluated strictly once at the end!
    """
    vis_dim = train_vis.shape[1]

    itr_vis, itr_clin = train_vis[inner_train_idx], train_clin[inner_train_idx]
    itr_t, itr_s = train_triage[inner_train_idx], train_subtype[inner_train_idx]

    iva_vis, iva_clin = train_vis[inner_val_idx], train_clin[inner_val_idx]
    iva_t, iva_s = train_triage[inner_val_idx], train_subtype[inner_val_idx]

    # Compute class weights strictly from inner_train
    valid_s = itr_s[itr_s >= 0].numpy()
    counts_s = Counter(valid_s)
    weights_s = torch.tensor(
        [len(valid_s) / (6.0 * max(counts_s[i], 1)) for i in range(6)],
        dtype=torch.float32,
        device=device,
    )
    valid_t = itr_t[itr_t >= 0].numpy()
    counts_t = Counter(valid_t)
    weights_t = torch.tensor(
        [len(valid_t) / (2.0 * max(counts_t[i], 1)) for i in [0, 1]],
        dtype=torch.float32,
        device=device,
    )

    itr_loader = DataLoader(TensorDataset(itr_vis, itr_clin, itr_t, itr_s), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_vis, val_clin, val_triage, val_subtype), batch_size=128, shuffle=False)

    seed_results = []
    for seed in SEEDS:
        set_seed(seed)
        probe = MultimodalLateFusionProbe(
            vis_dim=vis_dim,
            clin_dim=12,
            clin_embed_dim=64,
            hidden_dim=256,
            dropout=0.25,
            num_subtypes=6,
        ).to(device)

        loss_fn = MultiHeadLoss(
            triage_weight=1.0,
            subtype_weight=1.0,
            triage_criterion=MaskedClassWeightedFocalLoss(alpha=weights_t, gamma=2.0),
            subtype_criterion=MaskedClassWeightedFocalLoss(alpha=weights_s, gamma=2.0),
        )
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

        best_inner_f1 = -1.0
        best_state = None

        for _ in range(60):
            probe.train()
            for vx, cx, t_tgt, s_tgt in itr_loader:
                vx, cx, t_tgt, s_tgt = vx.to(device), cx.to(device), t_tgt.to(device), s_tgt.to(device)
                optimizer.zero_grad()
                t_logits, s_logits = probe(vx, cx)
                loss, _, _ = loss_fn(t_logits, s_logits, t_tgt, s_tgt)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Early stopping check strictly on inner_val
            probe.eval()
            with torch.no_grad():
                _, s_l = probe(iva_vis.to(device), iva_clin.to(device))
                s_p = torch.softmax(s_l, dim=1).cpu().numpy()
                v_mask = iva_s.numpy() >= 0
                inner_f1 = float(
                    f1_score(iva_s.numpy()[v_mask], np.argmax(s_p[v_mask], axis=1), average="macro", zero_division=0)
                )
            if inner_f1 > best_inner_f1:
                best_inner_f1 = inner_f1
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

        if best_state is not None:
            probe.load_state_dict(best_state)

        # Calibrate temperature strictly on inner_val
        probe.eval()
        probe.calibrate_temperatures(
            iva_vis.to(device),
            iva_clin.to(device),
            iva_t.to(device),
            iva_s.to(device),
        )

        # Single locked evaluation on val.csv
        probe.eval()
        t_probs_list, s_probs_list = [], []
        with torch.no_grad():
            for vx, cx, _, _ in val_loader:
                vx, cx = vx.to(device), cx.to(device)
                t_p, s_p = probe.predict_proba(vx, cx)
                t_probs_list.append(t_p.cpu())
                s_probs_list.append(s_p.cpu())

        t_probs_all = torch.cat(t_probs_list, dim=0).numpy()
        s_probs_all = torch.cat(s_probs_list, dim=0).numpy()
        res = evaluate_multimodal_predictions(t_probs_all, s_probs_all, val_subtype.numpy())
        res["seed"] = seed
        res["inner_val_best_f1"] = best_inner_f1
        seed_results.append(res)

    return {"summary": aggregate_seed_metrics(seed_results), "seeds": seed_results}


def main():
    print("=" * 95)
    print(" SkinAI EXP-013: Multi-Architecture Tournament & Class-Imbalance Ablation Study")
    print(" Dataset: Unified Development Pool v2 (3,767 images, 1,674 patients, 142 melanomas)")
    print(" External Stanford DDI Test Set: 100% LOCKED & UNTOUCHED")
    print("=" * 95)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    splits_dir = PROJECT_ROOT / "data" / "splits" / "development_pool_v2"
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")

    # Verify zero patient leakage between train and val
    train_pats = set(train_df["patient_id"].astype(str))
    val_pats = set(val_df["patient_id"].astype(str))
    assert len(train_pats.intersection(val_pats)) == 0, "Patient leakage detected!"

    # Create strict patient-grouped inner split of train.csv (85% inner_train / 15% inner_val)
    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
    strat_labels = train_df["subtype_target"].fillna("unmapped").astype(str).values
    groups = train_df["patient_id"].astype(str).values
    inner_train_idx, inner_val_idx = next(sgkf.split(train_df, strat_labels, groups))
    inner_tr_pats = set(groups[inner_train_idx])
    inner_va_pats = set(groups[inner_val_idx])
    assert len(inner_tr_pats.intersection(inner_va_pats)) == 0, "Inner split patient leakage!"

    print(
        f"[*] Split Integrity Verified: Train={len(train_df)} (Inner-Train={len(inner_train_idx)}, "
        f"Inner-Val={len(inner_val_idx)}), Hold-out Val={len(val_df)} (0% patient overlap)"
    )

    # Extract clinical tabular vectors (12D)
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

    train_subtype = torch.tensor(
        [CLASS_TO_IDX.get(str(s), -1) for s in train_df["subtype_target"]],
        dtype=torch.long,
    )
    val_subtype = torch.tensor(
        [CLASS_TO_IDX.get(str(s), -1) for s in val_df["subtype_target"]],
        dtype=torch.long,
    )

    valid_subtypes = train_subtype[train_subtype >= 0].numpy()
    counts_s = Counter(valid_subtypes)
    weights_s_tensor = torch.tensor(
        [len(valid_subtypes) / (6.0 * max(counts_s[i], 1)) for i in range(6)],
        dtype=torch.float32,
    )
    valid_triage = train_triage[train_triage >= 0].numpy()
    counts_t = Counter(valid_triage)
    weights_t_tensor = torch.tensor(
        [len(valid_triage) / (2.0 * max(counts_t[i], 1)) for i in [0, 1]],
        dtype=torch.float32,
    )

    architectures = [
        ("efficientnet_b0", "EfficientNet-B0 (CNN Baseline)"),
        ("cspdarknet53", "CSPDarknet-53 (YOLO Backbone)"),
        ("convnext_tiny", "ConvNeXt-Tiny (Modern 7x7 CNN)"),
        ("swin_tiny", "Swin-Tiny (Windowed ViT)"),
        ("biomedclip", "Microsoft BiomedCLIP (ViT-B/16)"),
        ("dinov2_small", "Meta DINOv2 Small (SSL ViT)"),
    ]

    tournament_results: Dict[str, Any] = {}

    for arch_key, arch_title in architectures:
        print("\n" + "-" * 95)
        print(f"[*] Evaluating Architecture: {arch_title} [{arch_key}]")
        tr_vis, va_vis, speed_info = get_or_extract_embeddings(arch_key, train_df, val_df, device)

        print(f"    -> Running Protocol A (Strict Inner-Val Early Stopping & Temp Calibration, 5 seeds)...")
        proto_a = run_protocol_a_inner_split(
            tr_vis,
            train_clin,
            train_triage,
            train_subtype,
            inner_train_idx,
            inner_val_idx,
            va_vis,
            val_clin,
            val_triage,
            val_subtype,
            device,
        )

        print(f"    -> Running Protocol B (Standard Full-Train 60 Epochs, 5 seeds)...")
        proto_b = run_protocol_b_full_train(
            tr_vis,
            train_clin,
            train_triage,
            train_subtype,
            va_vis,
            val_clin,
            val_triage,
            val_subtype,
            weights_t_tensor,
            weights_s_tensor,
            device,
        )

        sa = proto_a["summary"]
        sb = proto_b["summary"]
        print(
            f"    [+] {arch_title} | Protocol A: Macro F1={sa['macro_f1']['mean']:.4f}±{sa['macro_f1']['std']:.4f}, "
            f"Bal Acc={sa['balanced_accuracy']['mean']*100:.2f}%, ROC-AUC={sa['macro_roc_auc']['mean']:.4f}, "
            f"Mel Recall={sa['mel_hits_mean']:.1f}/29 ({sa['mel_recall_tau_005']['mean']*100:.1f}%)"
        )
        print(
            f"    [+] {arch_title} | Protocol B: Macro F1={sb['macro_f1']['mean']:.4f}±{sb['macro_f1']['std']:.4f}, "
            f"Bal Acc={sb['balanced_accuracy']['mean']*100:.2f}%, ROC-AUC={sb['macro_roc_auc']['mean']:.4f}, "
            f"Mel Recall={sb['mel_hits_mean']:.1f}/29 ({sb['mel_recall_tau_005']['mean']*100:.1f}%)"
        )

        tournament_results[arch_key] = {
            "title": arch_title,
            "embed_dim": int(tr_vis.shape[1]),
            "hardware_speed": speed_info,
            "protocol_a_inner_split": proto_a,
            "protocol_b_full_train": proto_b,
        }

    # -------------------------------------------------------------------------
    # Class-Imbalance Ablation Study on Champion Backbone (DINOv2 Small)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print(" [*] Running Class-Imbalance Countermeasure Ablation Study (on Meta DINOv2 Small)")
    print("=" * 95)
    tr_dino, va_dino, _ = get_or_extract_embeddings("dinov2_small", train_df, val_df, device)

    # 1. Unweighted Cross-Entropy (gamma=0.0, use_class_weights=False)
    ablation_unweighted_ce = run_protocol_b_full_train(
        tr_dino,
        train_clin,
        train_triage,
        train_subtype,
        va_dino,
        val_clin,
        val_triage,
        val_subtype,
        weights_t_tensor,
        weights_s_tensor,
        device,
        gamma=0.0,
        use_class_weights=False,
    )
    # 2. Inverse-Frequency Class-Weighted Cross-Entropy (gamma=0.0, use_class_weights=True)
    ablation_weighted_ce = run_protocol_b_full_train(
        tr_dino,
        train_clin,
        train_triage,
        train_subtype,
        va_dino,
        val_clin,
        val_triage,
        val_subtype,
        weights_t_tensor,
        weights_s_tensor,
        device,
        gamma=0.0,
        use_class_weights=True,
    )
    # 3. Inverse-Frequency Class-Weighted Focal Loss (gamma=2.0, use_class_weights=True)
    ablation_weighted_focal = tournament_results["dinov2_small"]["protocol_b_full_train"]

    imbalance_ablation = {
        "1_unweighted_ce": ablation_unweighted_ce["summary"],
        "2_inverse_weighted_ce": ablation_weighted_ce["summary"],
        "3_class_weighted_focal_gamma2": ablation_weighted_focal["summary"],
    }

    for k, v in imbalance_ablation.items():
        print(
            f"    -> {k:<32}: Macro F1={v['macro_f1']['mean']:.4f} | "
            f"Bal Acc={v['balanced_accuracy']['mean']*100:.2f}% | "
            f"Mel Naive Argmax={v['mel_recall_naive']['mean']*100:.1f}% | "
            f"Mel tau=0.05={v['mel_recall_tau_005']['mean']*100:.1f}% | "
            f"Consensus Safety={v['consensus_safety_recall']['mean']*100:.2f}%"
        )

    # Print Final Summary Tables
    print("\n" + "=" * 115)
    print(" EXP-013 SUMMARY TABLE (PROTOCOL B — FULL TRAIN 60 EPOCHS, 5 SEEDS)")
    print("=" * 115)
    header = (
        f"{'Architecture':<32} | {'Macro F1':<15} | {'Bal Acc':<13} | {'ROC-AUC':<13} | "
        f"{'Mel Recall (tau=.05)':<20} | {'GPU FPS':<8} | {'CPU ms':<8}"
    )
    print(header)
    print("-" * 115)
    for arch_key, _ in architectures:
        entry = tournament_results[arch_key]
        s = entry["protocol_b_full_train"]["summary"]
        sp = entry["hardware_speed"]
        f1_str = f"{s['macro_f1']['mean']:.4f}±{s['macro_f1']['std']:.4f}"
        ba_str = f"{s['balanced_accuracy']['mean']*100:.2f}%±{s['balanced_accuracy']['std']*100:.2f}%"
        auc_str = f"{s['macro_roc_auc']['mean']:.4f}±{s['macro_roc_auc']['std']:.4f}"
        mel_str = f"{s['mel_hits_mean']:.1f}/29 ({s['mel_recall_tau_005']['mean']*100:.1f}%)"
        print(
            f"{entry['title']:<32} | {f1_str:<15} | {ba_str:<13} | {auc_str:<13} | "
            f"{mel_str:<20} | {sp['gpu_batch16_fps']:<8.1f} | {sp['cpu_latency_ms']:<8.1f}"
        )
    print("=" * 115)

    out_path = PROJECT_ROOT / "artifacts" / "experiments" / "exp013_architecture_tournament.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": "EXP-013",
                "dataset": "Development Pool v2 (3,010 Train / 757 Val, 0% patient overlap)",
                "external_test_status": "Stanford DDI Locked & Untouched",
                "architectures": tournament_results,
                "class_imbalance_ablation": imbalance_ablation,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[+] Saved full EXP-013 benchmark report to: {out_path}")


if __name__ == "__main__":
    main()

