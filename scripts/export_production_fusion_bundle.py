#!/usr/bin/env python3
"""Export the Champion De-Confounded Fusion Head (ONNX + PyTorch) and Biopsy Atlas k-NN Index
into `artifacts/models/smartphone_clinical/` for production serving.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

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
from scripts.run_exp013_architecture_tournament import CLASS_TO_IDX, set_seed
from scripts.run_exp014_optuna_and_hyperparam_proofs import (
    ConfigurableMultimodalProbe,
    compute_powered_weights,
)
from scripts.run_exp017_fusion_diagnostics import extract_deconfounded_clinical_vector


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    print("=" * 96)
    print(" Exporting Production Fusion Bundle (De-Confounded Head + Biopsy Atlas Index)")
    print("=" * 96)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    splits_dir = PROJECT_ROOT / "data" / "splits" / "development_pool_v2"
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")

    cache_dir = PROJECT_ROOT / "artifacts" / "embeddings"
    tr_dino = torch.load(cache_dir / "dev_pool_v2_train_dinov2.pt", map_location="cpu", weights_only=True)
    va_dino = torch.load(cache_dir / "dev_pool_v2_val_dinov2.pt", map_location="cpu", weights_only=True)
    tr_biomed = torch.load(cache_dir / "dev_pool_v2_train_biomedclip.pt", map_location="cpu", weights_only=True)
    va_biomed = torch.load(cache_dir / "dev_pool_v2_val_biomedclip.pt", map_location="cpu", weights_only=True)

    tr_vis = torch.cat([F.normalize(tr_dino.float(), dim=-1), F.normalize(tr_biomed.float(), dim=-1)], dim=-1)
    va_vis = torch.cat([F.normalize(va_dino.float(), dim=-1), F.normalize(va_biomed.float(), dim=-1)], dim=-1)

    tr_clin = torch.tensor(
        np.stack([extract_deconfounded_clinical_vector(row) for _, row in train_df.iterrows()]),
        dtype=torch.float32,
    )
    va_clin = torch.tensor(
        np.stack([extract_deconfounded_clinical_vector(row) for _, row in val_df.iterrows()]),
        dtype=torch.float32,
    )

    tr_t = torch.tensor(train_df["triage_target"].astype(int).values, dtype=torch.long)
    tr_s = torch.tensor([CLASS_TO_IDX.get(str(s), -1) for s in train_df["subtype_target"]], dtype=torch.long)

    params = {
        "clin_embed_dim": 32,
        "hidden_dim": 128,
        "dropout": 0.30,
        "activation": "silu",
        "gamma": 2.5,
        "class_weight_power": 0.50,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "batch_size": 32,
    }

    set_seed(42)
    weights_s = compute_powered_weights(tr_s.numpy(), 6, params["class_weight_power"], device)
    weights_t = compute_powered_weights(tr_t.numpy(), 2, params["class_weight_power"], device)

    probe = ConfigurableMultimodalProbe(
        vis_dim=896,
        clin_dim=7,
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
            keep_mask = (torch.rand((cx.shape[0], 1), device=device) >= 0.50).float()
            cx = cx * keep_mask
            optimizer.zero_grad()
            tl, sl = probe(vx, cx)
            loss, _, _ = loss_fn(tl, sl, tt, st)
            loss.backward()
            optimizer.step()
        scheduler.step()

    probe.eval()
    probe_cpu = probe.cpu()

    bundle_dir = PROJECT_ROOT / "artifacts" / "models" / "smartphone_clinical"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save PyTorch checkpoint and export ONNX Fusion Head
    pt_path = bundle_dir / "fusion_head.pt"
    onnx_head_path = bundle_dir / "fusion_head.onnx"
    torch.save(
        {
            "state_dict": probe_cpu.state_dict(),
            "params": params,
            "classes": CLASS_ORDER,
            "vis_dim": 896,
            "clin_dim": 7,
        },
        pt_path,
    )

    dummy_vis = torch.randn(1, 896, dtype=torch.float32)
    dummy_clin = torch.zeros(1, 7, dtype=torch.float32)
    torch.onnx.export(
        probe_cpu,
        (dummy_vis, dummy_clin),
        str(onnx_head_path),
        input_names=["vis_896d", "clin_7d"],
        output_names=["triage_logits", "subtype_logits"],
        dynamic_axes={
            "vis_896d": {0: "batch_size"},
            "clin_7d": {0: "batch_size"},
            "triage_logits": {0: "batch_size"},
            "subtype_logits": {0: "batch_size"},
        },
        opset_version=17,
    )
    onnx.checker.check_model(str(onnx_head_path))

    # Verify ONNX Runtime parity
    sess = ort.InferenceSession(str(onnx_head_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_t, pt_s = probe_cpu(dummy_vis, dummy_clin)
    ort_t, ort_s = sess.run(None, {"vis_896d": dummy_vis.numpy(), "clin_7d": dummy_clin.numpy()})
    max_diff = float(max(np.max(np.abs(pt_t.numpy() - ort_t)), np.max(np.abs(pt_s.numpy() - ort_s))))
    print(f"[+] Exported fusion_head.onnx (ONNX parity max_diff = {max_diff:.2e})")

    # 2. Export Compact Biopsy Atlas Index (for both 896D Fusion and 6D Class-Probability k-NN retrieval)
    valid_mask = train_df["subtype_target"].isin(CLASS_ORDER).values
    valid_indices = np.where(valid_mask)[0]
    valid_df = train_df.iloc[valid_indices].reset_index(drop=True)
    valid_vis = F.normalize(tr_vis[valid_indices], dim=-1)
    valid_clin = tr_clin[valid_indices]

    with torch.no_grad():
        _, atlas_s_logits = probe_cpu(valid_vis, valid_clin)
        atlas_probs_6d = F.normalize(torch.softmax(atlas_s_logits, dim=1), dim=-1)

    records = []
    for idx, row in valid_df.iterrows():
        sub = str(row["subtype_target"])
        mal_lbl = "Malignant" if sub in MALIGNANT_CLASSES else "Benign"
        age_v = row.get("age_approx")
        try:
            age_f = float(age_v) if pd.notna(age_v) and age_v != "" else None
        except Exception:
            age_f = None
        records.append(
            {
                "image_id": str(row.get("image_id", f"case_{idx}")),
                "patient_id": str(row.get("patient_id", "unknown")),
                "subtype_id": sub,
                "malignancy_label": mal_lbl,
                "age_approx": age_f,
                "sex": str(row.get("sex", "unknown")),
                "anatom_site": str(row.get("anatom_site_1", "unknown")),
                "dataset_source": str(row.get("source_dataset", "pad_ufes_20")),
            }
        )

    atlas_path = bundle_dir / "biopsy_atlas_index.pt"
    torch.save(
        {
            "vis_embeddings_896d": valid_vis.half(),  # float16 to keep file compact (~3.6 MB for 2,027 cases)
            "prob_signatures_6d": atlas_probs_6d.float(),
            "records": records,
            "class_order": CLASS_ORDER,
        },
        atlas_path,
    )
    print(f"[+] Exported biopsy_atlas_index.pt ({len(records)} biopsy-verified atlas cases, {atlas_path.stat().st_size / 1024:.1f} KB)")

    # 3. Update manifest.yaml with Fusion metadata while preserving backward compatibility
    manifest_path = bundle_dir / "manifest.yaml"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    manifest["version"] = "2.0.0-fusion"
    manifest["fusion_head_onnx"] = "fusion_head.onnx"
    manifest["fusion_head_sha256"] = compute_sha256(onnx_head_path)
    manifest["biopsy_atlas_index"] = "biopsy_atlas_index.pt"
    manifest["fusion_metrics_exp018"] = {
        "global_5fold_oof_macro_f1": 0.7161,
        "pad_ufes20_5fold_oof_macro_f1": 0.7079,
        "oof_triage_roc_auc_with_ack": 0.9453,
        "oof_triage_roc_auc_no_ack": 0.9917,
        "oof_melanoma_recall_tau015": 0.8944,
        "oof_melanoma_specificity_tau015": 0.9530,
        "oof_melanoma_recall_tau005": 0.9577,
        "stanford_ddi_in_scope_roc_auc": 0.8250,
        "stanford_ddi_all_656_roc_auc": 0.7348,
        "onnx_fusion_parity_max_diff": max_diff,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)

    print(f"[+] Updated {manifest_path} in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
