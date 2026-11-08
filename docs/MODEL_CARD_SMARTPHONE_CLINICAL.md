# Model Card: SkinAI Smartphone Clinical Profile (`v2.0.0-fusion`)

## 1. Model Details
- **Model Name**: SkinAI De-Confounded Dual-Backbone Fusion (`DINOv2 Small + BiomedCLIP` + Dual-Head MLP) & Fast ONNX Runtime Engine
- **Visual Backbones**:
  - `Meta DINOv2 Small` (`ViT-S/14`, `384D` L2-normalized embeddings + `[CLS]` Self-Attention map)
  - `Microsoft BiomedCLIP` (`ViT-B/16` PMC-15M, `512D` L2-normalized embeddings)
  - Fast CPU Edge Fallback: `efficientnet_b0` (`model.onnx`) + `fusion_head.onnx` (`ONNX parity max_diff = 3.58e-07`)
- **De-Confounded Clinical Context (`7D`)**: `[age_norm, site_head_neck, site_trunk, site_upper_ext, site_lower_ext, sex_female, sex_male]` trained with `50% Metadata Dropout` (zero dataset-shortcut `unknown` flags)
- **Explainability ("Deterministic XAI" Engine)**:
  - **DS-5 Morphometry**: Morphological `Black-Hat` hair removal (`9x9`) + Telea inpainting + `DINOv2` Attention-Seeded `cv2.grabCut` contour segmentation $\to$ 12 quantitative `ABCDE` metrics in perceptual `CIE L*a*b*` space
  - **Biopsy Reference Atlas (`k-NN`)**: `biopsy_atlas_index.pt` (`2,027` histologically verified reference cases returning `Top-3` clinical twins)
- **Detailed Evidence**: [`docs/SENIOR_CONSILIUM_AUDIT_REPORT.md`](file:///c:/HomeWork/Medical/Dermatology/docs/SENIOR_CONSILIUM_AUDIT_REPORT.md) and [`docs/EXPERIMENTS_LOG.md`](file:///c:/HomeWork/Medical/Dermatology/docs/EXPERIMENTS_LOG.md)

## 2. Intended Use & Safety Scope
- **Target Profile**: Consumer smartphone close-up and clinical macroscopic skin lesion photos (`PAD-UFES-20`, `MED-NODE`, `ISIC Clinical Close-Up`)
- **License Scope**: `commercial_ok` / `public_ok` (Permissive `CC BY 4.0` / `CC0` provenance documented in [`docs/DATA_GOVERNANCE.md`](file:///c:/HomeWork/Medical/Dermatology/docs/DATA_GOVERNANCE.md))
- **Clinical Disclaimer**: Experimental research tool. NOT a diagnostic medical device.
- **Out-of-Distribution (`CIE L*a*b*` OOD) & Quality Gate**: Rejects blurry/corrupt inputs (`Laplacian < 8.0`, `< 128x128`) and abstains on non-skin / out-of-distribution inputs (`skin_pixel_ratio < 0.25` or high-frequency noise).
- **Decoupled 3-Zone Clinical Triage (`Red / Yellow / Green`)**:
  - `🔴 Red Zone` (`urgent_referral`): $\tau_{\text{mel}} = 0.15$ (or high-vigilance $\tau_{\text{mel}} = 0.05$), $\tau_{\text{mal}} = 0.62$, or NICE NG12 Glasgow score $\ge 3$.
  - `🟡 Yellow Zone` (`routine_consultation`): Pre-cancerous Actinic Keratosis ($\tau_{\text{ack}} = 0.20$, capturing **99.73% (`728/730`)** of all `ACK` cases), moderate risk, or OOD/uncertainty abstention.
  - `🟢 Green Zone` (`routine_no_high_risk`): High-confidence benign lesion with Glasgow score $< 3$.

## 3. Evaluation Metrics (`EXP-017` & `EXP-018`: 5-Fold StratifiedGroupKFold CV, `3,767` images, `1,674` patients)

| Метрика | `De-Confounded Fusion (903D)` | `Hybrid (0.70 Fusion + 0.30 LogReg)` | `Pure Image Fusion (896D)` |
|---|---|---|---|
| **Global 5-Fold OOF Macro F1 (6 классов)** | **`0.7161 ± 0.0080`** | `0.7091` | `0.6561 ± 0.0110` |
| **PAD-UFES-20 OOF Macro F1 ($N=2,298$)** | **`0.7079`** | `0.7010` | `0.6434` |
| **OOF Triage ROC-AUC (With ACK, $N=3,767$)** | **`0.9453`** | **`0.9576`** | `0.8898` |
| **OOF Triage ROC-AUC (No ACK, $N=3,037$)** | **`0.9917`** | `0.9902` | `0.9870` |
| **External Blind Test: Stanford DDI In-Scope AUC ($N=301$)** | **`0.8250`** | **`0.8266`** | `0.8241` |
| **External Blind Test: Stanford DDI All AUC ($N=656$)** | **`0.7348`** | `0.7313` | **`0.7383`** |

### Paired Operating Points for Melanoma Detection (`5-Fold OOF`, $N_{\text{mel}} = 142$ across `2,547` subtyped lesions)

| Рабочий порог $\tau_{\text{mel}}$ | Чувствительность (Recall, 95% CI) | Специфичность (Specificity, 95% CI) | PPV (Точность положительного) | NPV (Предсказательная ценность отрицательного) | FPR (Ложноположительная доля) |
|---|---|---|---|---|---|
| **$\tau_{\text{mel}} = 0.05$ (Максимальная онкобдительность)** | **`95.77%` (`136/142`, `[91.1%, 98.1%]`)** | **`88.73%` (`[87.4%, 89.9%]`)** | `33.42%` | **`99.72%`** | `11.27%` |
| **$\tau_{\text{mel}} = 0.10$ (Сбалансированный скрининг)** | **`90.85%` (`129/142`, `[85.0%, 94.6%]`)** | **`93.31%` (`[92.2%, 94.2%]`)** | `44.48%` | **`99.42%`** | `6.69%` |
| **$\tau_{\text{mel}} = 0.15$ (Порог Red Zone по умолчанию)** | **`89.44%` (`127/142`, `[83.3%, 93.5%]`)** | **`95.30%` (`[94.4%, 96.1%]`)** | `52.92%` | **`99.35%`** | `4.70%` |
| **$\tau_{\text{mel}} = 0.25$ (Высокая специфичность)** | **`82.39%` (`117/142`, `[75.3%, 87.8%]`)** | **`97.30%` (`[96.6%, 97.9%]`)** | `64.29%` | **`98.94%`** | `2.70%` |

