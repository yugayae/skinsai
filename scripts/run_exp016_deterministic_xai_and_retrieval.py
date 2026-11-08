#!/usr/bin/env python3
"""EXP-016: 'Deterministic XAI' Explainability, Biopsy Atlas k-NN Retrieval & Quantitative ABCDE Morphometry.

Demonstrates and benchmarks three deterministic clinical AI capabilities that general-purpose
deterministic explainability provides:
1. Biopsy Atlas Visual Case Retrieval (k-NN Case-Based Reasoning over 3,010 verified lesions)
2. Quantitative ABCDE Computer Vision Morphometry (Asymmetry, Border, Color CIE L*a*b* Entropy)
3. Pixel-Level DINOv2 Self-Attention Map Extraction & Lesion Focus Concentration Proof
4. Parametric + Non-Parametric + Morphometric Hybrid Fusion Evaluation (5 seeds on Hold-out Val)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.explainability.deterministic_xai import (
    ABCDEMorphometryEngine,
    BiopsyAtlasRetrievalEngine,
    DINOv2AttentionExtractor,
)
from ml.models.multi_head import CLASS_ORDER, MALIGNANT_CLASSES
from scripts.run_exp013_architecture_tournament import (
    CLASS_TO_IDX,
    MALIGNANT_INDICES,
    SEEDS,
    aggregate_seed_metrics,
    evaluate_multimodal_predictions,
    extract_clinical_vector,
    set_seed,
)
from scripts.run_exp015_multi_backbone_fusion import train_and_predict_specialist


def get_or_compute_abcde_features(df: pd.DataFrame, cache_path: Path, split_name: str) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Compute or load cached 12D quantitative ABCDE morphometry vectors for a dataset split."""
    json_cache = cache_path.with_suffix(".json")
    if cache_path.exists() and json_cache.exists():
        print(f"    [*] Loading cached ABCDE morphometry features for {split_name} from {cache_path.name}...")
        vecs = torch.load(cache_path, map_location="cpu", weights_only=True)
        with open(json_cache, "r", encoding="utf-8") as f:
            raw_dicts = json.load(f)
        return vecs, raw_dicts

    print(f"    [*] Computing deterministic ABCDE morphometry biomarkers for {split_name} ({len(df)} images)...")
    t0 = time.time()
    vec_list = []
    raw_dicts = []
    for _, row in df.iterrows():
        img_p = PROJECT_ROOT / row["file_path"]
        with Image.open(img_p) as pil_img:
            metrics, _ = ABCDEMorphometryEngine.compute_metrics(pil_img, target_size=224)
        vec_list.append(metrics.to_vector())
        d = metrics.to_dict()
        d["subtype_target"] = str(row.get("subtype_target", "unmapped"))
        d["triage_target"] = int(row.get("triage_target", -1))
        raw_dicts.append(d)

    vecs = torch.tensor(np.stack(vec_list), dtype=torch.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vecs, cache_path)
    with open(json_cache, "w", encoding="utf-8") as f:
        json.dump(raw_dicts, f)
    print(f"        [+] Computed {len(vecs)} ABCDE morphometry vectors in {time.time() - t0:.1f}s")
    return vecs, raw_dicts


def analyze_abcde_biomarker_statistics(val_abcde_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Quantify statistical separation of ABCDE biomarkers between Melanoma, Malignant, and Benign Nevi."""
    df_m = pd.DataFrame(val_abcde_dicts)
    mel_df = df_m[df_m["subtype_target"] == "mel"]
    nev_df = df_m[df_m["subtype_target"] == "nev"]
    mal_df = df_m[df_m["subtype_target"].isin(MALIGNANT_CLASSES)]
    ben_df = df_m[df_m["subtype_target"].isin({"nev", "sek", "ack"})]

    biomarkers = [
        "asymmetry_shape",
        "asymmetry_color",
        "border_irregularity",
        "border_radial_std",
        "color_std_l",
        "color_std_a",
        "color_entropy",
        "distinct_color_clusters",
    ]

    stats_summary = {}
    for bm in biomarkers:
        mel_v = mel_df[bm].values
        nev_v = nev_df[bm].values
        mal_v = mal_df[bm].values
        ben_v = ben_df[bm].values

        # Standardized mean difference (Cohen's d) between Melanoma and Benign Nevus
        pooled_sd = np.sqrt(0.5 * (np.var(mel_v) + np.var(nev_v)) + 1e-8)
        cohens_d_mel_vs_nev = float((np.mean(mel_v) - np.mean(nev_v)) / pooled_sd)

        stats_summary[bm] = {
            "melanoma_mean": round(float(np.mean(mel_v)), 4),
            "melanoma_std": round(float(np.std(mel_v)), 4),
            "benign_nevus_mean": round(float(np.mean(nev_v)), 4),
            "benign_nevus_std": round(float(np.std(nev_v)), 4),
            "malignant_all_mean": round(float(np.mean(mal_v)), 4),
            "benign_all_mean": round(float(np.mean(ben_v)), 4),
            "cohens_d_mel_vs_nev": round(cohens_d_mel_vs_nev, 4),
        }
    return stats_summary


def benchmark_biopsy_atlas_retrieval(
    tr_emb: torch.Tensor,
    train_df: pd.DataFrame,
    va_emb: torch.Tensor,
    val_df: pd.DataFrame,
    space_name: str,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Benchmark Visual k-NN Retrieval over the 3,010-image Biopsy Atlas."""
    engine = BiopsyAtlasRetrievalEngine(
        atlas_embeddings=tr_emb,
        atlas_df=train_df,
        class_order=CLASS_ORDER,
        malignant_classes=set(MALIGNANT_CLASSES),
        temperature=0.07,
    )

    top1_exact = 0
    top3_hit = 0
    top5_hit = 0
    mal_query_count = 0
    mal_hit_at_3 = 0
    mel_query_count = 0
    mel_hit_at_5 = 0
    valid_count = 0

    all_knn_probs = np.zeros((len(val_df), len(CLASS_ORDER)), dtype=np.float32)
    sample_retrievals = []

    for i, row in val_df.iterrows():
        true_sub = str(row.get("subtype_target", "unmapped"))
        pat_id = str(row.get("patient_id", ""))
        cases, knn_p = engine.retrieve_top_k(va_emb[i], k=5, exclude_patient_id=pat_id)
        all_knn_probs[i] = knn_p

        if true_sub not in CLASS_TO_IDX:
            continue

        valid_count += 1
        retrieved_subs = [c.subtype_id for c in cases]
        if retrieved_subs[0] == true_sub:
            top1_exact += 1
        if true_sub in retrieved_subs[:3]:
            top3_hit += 1
        if true_sub in retrieved_subs[:5]:
            top5_hit += 1

        if true_sub in MALIGNANT_CLASSES:
            mal_query_count += 1
            if any(s in MALIGNANT_CLASSES for s in retrieved_subs[:3]):
                mal_hit_at_3 += 1

        if true_sub == "mel":
            mel_query_count += 1
            if "mel" in retrieved_subs[:5]:
                mel_hit_at_5 += 1
            if len(sample_retrievals) < 3:
                sample_retrievals.append(
                    {
                        "query_image_id": str(row.get("image_id")),
                        "true_diagnosis": true_sub,
                        "top_3_biopsy_twins": [
                            {
                                "rank": c.rank,
                                "image_id": c.image_id,
                                "similarity": c.cosine_similarity,
                                "biopsy_diagnosis": c.subtype_id,
                                "malignancy": c.malignancy_label,
                            }
                            for c in cases[:3]
                        ],
                    }
                )

    metrics = {
        "space": space_name,
        "top1_exact_subtype_acc": round(float(top1_exact / max(valid_count, 1)), 4),
        "top3_subtype_hit_rate": round(float(top3_hit / max(valid_count, 1)), 4),
        "top5_subtype_hit_rate": round(float(top5_hit / max(valid_count, 1)), 4),
        "malignancy_hit_rate_at_3": round(float(mal_hit_at_3 / max(mal_query_count, 1)), 4),
        "melanoma_hit_rate_at_5": round(float(mel_hit_at_5 / max(mel_query_count, 1)), 4),
        "sample_melanoma_retrievals": sample_retrievals,
    }
    return metrics, all_knn_probs


def evaluate_dinov2_attention_focus(
    val_df: pd.DataFrame,
    device: torch.device,
) -> Dict[str, Any]:
    """Verify pixel-level DINOv2 self-attention concentration inside the lesion vs. background skin."""
    print("\n[*] Part 3: Extracting Pixel-Level DINOv2 Self-Attention Maps & Rendering Visual Cards...")
    dino_model = timm.create_model("vit_small_patch14_dinov2", pretrained=True, num_classes=0).to(device)
    dino_model.eval()
    extractor = DINOv2AttentionExtractor(dino_model, device)

    transform_518 = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mel_rows = val_df[val_df["subtype_target"] == "mel"].head(29)
    nev_rows = val_df[val_df["subtype_target"] == "nev"].head(29)
    eval_rows = pd.concat([mel_rows, nev_rows], ignore_index=True)

    enrichment_ratios = []
    lesion_attn_fractions = []
    cards_dir = PROJECT_ROOT / "artifacts" / "explainability_samples"
    cards_dir.mkdir(parents=True, exist_ok=True)
    saved_cards = []

    for idx, row in eval_rows.iterrows():
        img_p = PROJECT_ROOT / row["file_path"]
        sub = str(row["subtype_target"])
        img_id = str(row.get("image_id", f"sample_{idx}"))
        with Image.open(img_p) as pil_img:
            rgb = pil_img.convert("RGB")
            abcde, mask = ABCDEMorphometryEngine.compute_metrics(rgb, target_size=224)
            t_518 = transform_518(rgb)
            amap = extractor.extract_attention_map(t_518, target_size=224)

            # Calculate attention density inside lesion mask vs background
            mean_in = float(np.mean(amap[mask])) if np.any(mask) else 0.5
            mean_out = float(np.mean(amap[~mask])) if np.any(~mask) else 0.5
            enrichment = mean_in / max(mean_out, 1e-4)
            enrichment_ratios.append(enrichment)

            total_attn = float(np.sum(amap))
            in_attn_frac = float(np.sum(amap[mask])) / max(total_attn, 1e-4)
            lesion_attn_fractions.append(in_attn_frac)

            if len(saved_cards) < 6 and (idx < 3 or (sub == "nev" and len(saved_cards) < 6)):
                card_file = cards_dir / f"card_{sub}_{Path(img_id).stem}.png"
                extractor.render_explainability_card(
                    rgb,
                    mask,
                    amap,
                    abcde,
                    card_file,
                    title_text=f"True: {sub.upper()} | Enrich: {enrichment:.2f}x",
                )
                saved_cards.append(str(card_file.relative_to(PROJECT_ROOT)))

    extractor.close()
    del dino_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pct_focused = float(np.mean([r > 1.0 for r in enrichment_ratios]))
    return {
        "evaluated_lesions": len(eval_rows),
        "mean_attention_enrichment_ratio": round(float(np.mean(enrichment_ratios)), 4),
        "median_attention_enrichment_ratio": round(float(np.median(enrichment_ratios)), 4),
        "lesion_focused_rate_gt_1x": round(pct_focused, 4),
        "mean_attention_mass_in_lesion": round(float(np.mean(lesion_attn_fractions)), 4),
        "saved_visual_cards": saved_cards,
    }


def main():
    print("=" * 105)
    print(" SkinAI EXP-016: 'Deterministic XAI' Explainability, Biopsy Atlas k-NN Retrieval & ABCDE Morphometry")
    print("=" * 105)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    splits_dir = PROJECT_ROOT / "data" / "splits" / "development_pool_v2"
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")

    cache_dir = PROJECT_ROOT / "artifacts" / "embeddings"
    tr_dino = torch.load(cache_dir / "dev_pool_v2_train_dinov2.pt", map_location="cpu", weights_only=True)
    va_dino = torch.load(cache_dir / "dev_pool_v2_val_dinov2.pt", map_location="cpu", weights_only=True)
    tr_biomed = torch.load(cache_dir / "dev_pool_v2_train_biomedclip.pt", map_location="cpu", weights_only=True)
    va_biomed = torch.load(cache_dir / "dev_pool_v2_val_biomedclip.pt", map_location="cpu", weights_only=True)

    # -------------------------------------------------------------------------
    # Part 1: Biopsy Atlas Visual Case Retrieval (k-NN Case-Based Reasoning)
    # -------------------------------------------------------------------------
    print("\n[*] Part 1: Benchmarking Biopsy Atlas Visual k-NN Retrieval (3,010 Reference Cases)...")
    tr_joint = F.normalize(torch.cat([F.normalize(tr_dino, dim=-1), F.normalize(tr_biomed, dim=-1)], dim=-1), dim=-1)
    va_joint = F.normalize(torch.cat([F.normalize(va_dino, dim=-1), F.normalize(va_biomed, dim=-1)], dim=-1), dim=-1)

    ret_dino, knn_probs_dino = benchmark_biopsy_atlas_retrieval(tr_dino, train_df, va_dino, val_df, "DINOv2_384D")
    ret_biomed, knn_probs_biomed = benchmark_biopsy_atlas_retrieval(
        tr_biomed, train_df, va_biomed, val_df, "BiomedCLIP_512D"
    )
    ret_joint, knn_probs_joint = benchmark_biopsy_atlas_retrieval(
        tr_joint, train_df, va_joint, val_df, "Joint_Consilium_896D"
    )

    for r in [ret_dino, ret_biomed, ret_joint]:
        print(
            f"    -> Space: {r['space']:<22} | Top-1 Exact: {r['top1_exact_subtype_acc']*100:.2f}% | "
            f"Top-3 Subtype Hit: {r['top3_subtype_hit_rate']*100:.2f}% | "
            f"Malignant Hit@3: {r['malignancy_hit_rate_at_3']*100:.2f}% | "
            f"Melanoma Hit@5: {r['melanoma_hit_rate_at_5']*100:.2f}%"
        )

    # -------------------------------------------------------------------------
    # Part 2: Quantitative ABCDE Morphometry Extraction & Statistical Proof
    # -------------------------------------------------------------------------
    print("\n[*] Part 2: Extracting & Validating Quantitative ABCDE Morphometry Biomarkers...")
    tr_abcde, tr_abcde_dicts = get_or_compute_abcde_features(
        train_df, cache_dir / "dev_pool_v2_train_abcde.pt", "Train"
    )
    va_abcde, va_abcde_dicts = get_or_compute_abcde_features(
        val_df, cache_dir / "dev_pool_v2_val_abcde.pt", "Validation"
    )

    abcde_stats = analyze_abcde_biomarker_statistics(va_abcde_dicts)
    print("    [+] Statistical Separation of ABCDE Biomarkers (Melanoma vs. Benign Nevus on Validation):")
    for bm, st in abcde_stats.items():
        print(
            f"        {bm:<25}: Melanoma={st['melanoma_mean']:.3f}±{st['melanoma_std']:.3f} vs "
            f"Nevus={st['benign_nevus_mean']:.3f}±{st['benign_nevus_std']:.3f} (Cohen's d = {st['cohens_d_mel_vs_nev']:+.3f})"
        )

    # -------------------------------------------------------------------------
    # Part 3: DINOv2 Pixel-Level Self-Attention Verification
    # -------------------------------------------------------------------------
    attention_report = evaluate_dinov2_attention_focus(val_df, device)
    print(
        f"    [+] DINOv2 Self-Attention Concentration Inside Lesion: "
        f"{attention_report['mean_attention_enrichment_ratio']:.2f}x denser than background "
        f"({attention_report['lesion_focused_rate_gt_1x']*100:.1f}% lesions > 1.0x)"
    )
    print(f"    [+] Saved 3-Panel Visual Explainability Cards: {attention_report['saved_visual_cards']}")

    # -------------------------------------------------------------------------
    # Part 4: Hybrid Fusion Evaluation (Late Fusion + ABCDE Morphometry + k-NN Atlas)
    # -------------------------------------------------------------------------
    print("\n[*] Part 4: Evaluating Full Hybrid System (Neural Fusion + ABCDE Morphometry + k-NN Atlas, 5 seeds)...")
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

    # Enrich vision embeddings with standardized 12D ABCDE morphometry features
    tr_dino_abcde = torch.cat([tr_dino, tr_abcde], dim=-1)
    va_dino_abcde = torch.cat([va_dino, va_abcde], dim=-1)

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

    hybrid_seeds = {
        "1_Neural_Consilium_Baseline_EXP015": [],
        "2_Neural_Consilium_Plus_Biopsy_kNN_Atlas": [],
        "3_Full_Hybrid_Consilium_ABCDE_Plus_kNN_Atlas": [],
    }
    y_val_np = val_subtype.numpy()

    for seed in SEEDS:
        tp_d, sp_d = train_and_predict_specialist(
            tr_dino, train_clin, train_triage, train_subtype, va_dino, val_clin, params_dino, seed, device
        )
        tp_b, sp_b = train_and_predict_specialist(
            tr_biomed, train_clin, train_triage, train_subtype, va_biomed, val_clin, params_biomed, seed, device
        )
        tp_da, sp_da = train_and_predict_specialist(
            tr_dino_abcde, train_clin, train_triage, train_subtype, va_dino_abcde, val_clin, params_dino, seed, device
        )

        # 1. Neural Fusion (0.50 DINOv2 + 0.50 BiomedCLIP)
        tp_cons = 0.50 * tp_d + 0.50 * tp_b
        sp_cons = 0.50 * sp_d + 0.50 * sp_b
        hybrid_seeds["1_Neural_Consilium_Baseline_EXP015"].append(
            evaluate_multimodal_predictions(tp_cons, sp_cons, y_val_np)
        )

        # 2. Neural Fusion + Non-Parametric Biopsy Atlas k-NN (90% Neural + 10% Biopsy k-NN)
        sp_knn_blend = 0.90 * sp_cons + 0.10 * knn_probs_joint
        hybrid_seeds["2_Neural_Consilium_Plus_Biopsy_kNN_Atlas"].append(
            evaluate_multimodal_predictions(tp_cons, sp_knn_blend, y_val_np)
        )

        # 3. Full Hybrid (DINOv2+ABCDE + BiomedCLIP + Biopsy Atlas k-NN)
        tp_full = 0.50 * tp_da + 0.50 * tp_b
        sp_full = 0.45 * sp_da + 0.45 * sp_b + 0.10 * knn_probs_joint
        hybrid_seeds["3_Full_Hybrid_Consilium_ABCDE_Plus_kNN_Atlas"].append(
            evaluate_multimodal_predictions(tp_full, sp_full, y_val_np)
        )

    hybrid_summary = {k: aggregate_seed_metrics(v) for k, v in hybrid_seeds.items()}
    print("\n" + "=" * 115)
    print(" EXP-016 HYBRID Fusion + deterministic-xai MODULES SUMMARY (5 SEEDS ON HOLD-OUT VAL)")
    print("=" * 115)
    for k, agg in hybrid_summary.items():
        print(
            f" {k:<46} | Macro F1: {agg['macro_f1']['mean']:.4f}±{agg['macro_f1']['std']:.4f} | "
            f"Bal Acc: {agg['balanced_accuracy']['mean']*100:.2f}% | "
            f"ROC-AUC: {agg['macro_roc_auc']['mean']:.4f} | "
            f"Mel Recall(.05): {agg['mel_hits_mean']:.1f}/29 ({agg['mel_recall_tau_005']['mean']*100:.1f}%)"
        )
    print("=" * 115)

    out_path = PROJECT_ROOT / "artifacts" / "experiments" / "exp016_deterministic_xai_and_retrieval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": "EXP-016",
                "biopsy_atlas_retrieval": {
                    "dinov2_384d": ret_dino,
                    "biomedclip_512d": ret_biomed,
                    "joint_consilium_896d": ret_joint,
                },
                "abcde_morphometry_statistics": abcde_stats,
                "dinov2_pixel_attention_verification": attention_report,
                "hybrid_classification_benchmark": hybrid_summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[+] Saved full EXP-016 Deterministic XAI report to: {out_path}")


if __name__ == "__main__":
    main()

