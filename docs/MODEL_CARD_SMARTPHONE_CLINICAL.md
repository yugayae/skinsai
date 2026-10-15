# Model Card: SkinAI Smartphone Clinical Profile

## 1. Model Details
- **Model Name**: SkinAI EfficientNet-B0 (Smartphone Clinical Profile)
- **Backbone**: `efficientnet_b0` (timm ImageNet pre-trained)
- **Task**: 6-class skin lesion image classification from smartphone photos
- **Input Resolution**: 224x224x3
- **Calibrated Temperature**: $T = 1.5813$
- **Expected Calibration Error (ECE)**: **5.49%** (Reduced from 14.24%, 61.4% improvement)
- **ONNX Model SHA-256**: `9e66c808308df2a954b6da50bac262e6864997bfee7bb6dd79830da8de3faace`
- **Training Hardware**: NVIDIA GeForce RTX 3060 12GB (PyTorch AMP)
- **Detailed Evidence**: [`docs/SCIENTIFIC_VALIDATION_REPORT.md`](docs/SCIENTIFIC_VALIDATION_REPORT.md)

## 2. Intended Use & Safety Scope
- **Target Profile**: Consumer smartphone close-up skin lesion photos (PAD-UFES-20 baseline)
- **License Scope**: `commercial_ok` (Derived 100% from CC BY 4.0 PAD-UFES-20 data)
- **Clinical Disclaimer**: Experimental research tool. NOT a diagnostic medical device.
- **Abstention Policy**: Model rejects input if max calibrated probability < 0.40 or image fails quality filters.
- **Decoupled Clinical Triage**: Clinical decisions do not rely on naive argmax; safety referral thresholds ($\tau_{\text{mel}} = 0.05$–$0.20$, $\tau_{\text{mal}} = 0.20$–$0.30$) and Glasgow anamnesis are used.

## 3. Evaluation Metrics (Untouched Test Set, 346 images, 205 patients)

- **Overall Accuracy**: 64.74%
- **Balanced Accuracy**: 51.41%
- **Macro F1-Score**: 0.5089
- **Macro ROC-AUC**: **0.8552** (One-vs-Rest)
- **Macro PR-AUC**: **0.5772**

### Detailed Per-Class Performance

| Класс | Precision | Recall | F1-Score | ROC-AUC | Тестовая выборка (Support) |
|---|---|---|---|---|---|
| `bcc` | 0.735 | 0.764 | 0.749 | 0.8928 | 127 |
| `scc` | 0.080 | 0.074 | 0.077 | 0.6471 | 27 |
| `ack` | 0.750 | 0.673 | 0.709 | 0.8754 | 107 |
| `sek` | 0.545 | 0.649 | 0.593 | 0.9259 | 37 |
| `mel` | 0.286 | 0.250 | 0.267 | 0.8351 | 8 |
| `nev` | 0.643 | 0.675 | 0.659 | 0.9547 | 40 |

### Clinical Threshold Performance (Multi-Threshold Safety)

| Назначение | Порог $\tau$ | Чувствительность (Recall) | Специфичность | Доля направленных к врачу |
|---|---|---|---|---|
| **Меланома (Высокая бдительность)** | $\tau_{\text{mel}} = 0.05$ | **75.0% (6/8)** | **87.28%** | 14.16% |
| **Меланома (Канонический порог)** | $\tau_{\text{mel}} = 0.20$ | **50.0% (4/8)** | **97.63%** | 3.47% |
| **Злокачественная группа (`bcc`+`scc`+`mel`)** | $\tau_{\text{mal}} = 0.20$ | **95.68% (155/162)** | 54.89% | 46.24% |
| **Злокачественная группа (`bcc`+`scc`+`mel`)** | $\tau_{\text{mal}} = 0.30$ | **91.98% (149/162)** | 67.39% | 34.68% |
