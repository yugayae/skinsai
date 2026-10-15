# Architecture Specification: SkinAI

## 1. System Overview
SkinAI is an end-to-end computer vision and clinical decision support research platform for skin lesion classification. It decouples the client tier (Telegram Bot & Telegram Mini App), the service tier (FastAPI Backend), and the core ML engine (ONNX Runtime / PyTorch).

```mermaid
flowchart TD
    subgraph Client ["Client Layer"]
        TG["Telegram Bot (aiogram 3 - RU / EN / KO)"]
        MA["Telegram Mini App (Vite / TS - RU / EN / KO)"]
        PDF["Doctor Clinical Memo PDF Generator"]
    end

    subgraph Service ["Backend Service (FastAPI)"]
        API["FastAPI App (/v1/predict)"]
        VAL["Image Quality & Magic Byte Validator"]
        REG["Model Registry & Profile Manager"]
        DB[("PostgreSQL 16 (Salted Telemetry)")]
    end

    subgraph ML ["Core ML Engine"]
        PRE["Canonical Preprocessor (224x224 RGB)"]
        M_SMART["Smartphone EfficientNet-B0 ONNX Model"]
        CAL["Temperature Calibration (T=1.5813)"]
        TRIAGE["Decoupled 4-Tier Clinical Triage Engine"]
    end

    TG -->|Async HTTP Multipart| API
    MA -->|Standalone or WebApp| API
    TG --> PDF
    MA --> PDF
    API --> VAL
    VAL --> PRE
    PRE --> REG
    REG --> M_SMART
    M_SMART --> CAL
    CAL --> TRIAGE
    TRIAGE --> API
    API --> DB
    API --> TG
    API --> MA
```

---

## 2. Component Specifications

### 2.1 Telegram Bot (`bot/`)
- **Technology**: Python 3.11+, `aiogram 3.x`.
- **Localization Engine**: Trilingual support across all UI flows, dialogs, keyboards, and clinical nosology names:
  - 🇷🇺 **Russian** (`ru`) — Primary interface language;
  - 🇬🇧 **English** (`en`) — International clinical reference;
  - 🇰🇷 **Korean** (`ko`) — East-Asian clinical reference.
- **Interactive Triage & Anamnesis**:
  - Implements the **Glasgow 7-Point Checklist** (NICE Guideline NG12) via aiogram FSM.
  - Generates an in-memory doctor encounter memo PDF (`bot/services/pdf_report.py`) with zero cognitive anchoring (patient photo, answers, questions for doctor, no model probabilities).
- **Security & Privacy**:
  - Zero raw image retention on disk.
  - Private testing allowlist (`TELEGRAM_ALLOWED_USERS`) to prevent unauthorized access.

### 2.2 Telegram Mini App (`miniapp/`)
- **Technology**: Vite 6, TypeScript, Vanilla CSS design system.
- **Features**:
  - Direct integration with Telegram WebApp SDK (`Telegram.WebApp.ready()`, `expand()`, `HapticFeedback`, `MainButton`).
  - Adaptive Dark/Light theme switching.
  - Interactive Glasgow 7-point checklist progress meter (0/7 to 7/7).
  - ABCD educational cards for patient understanding.
  - In-browser client PDF generation (`jspdf`) with zero cognitive anchoring.

### 2.3 Backend Service (`backend/`)
- **Technology**: Python 3.11+, `FastAPI`, `Pydantic v2`, `SQLAlchemy 2`.
- **Endpoints**:
  - `GET /health`: Readiness and liveness probes (DB ping, model loading check).
  - `GET /v1/profiles`: Metadata on active model profiles, target classes, and release gate status.
  - `POST /v1/predict`: Multipart endpoint accepting image bytes, profile string, and optional Glasgow anamnesis.
- **Input Gating**:
  - Strict magic byte validation (`detect_image_format_from_bytes`) rejecting polyglots and executable scripts.
  - Pillow decompression bomb protection (`Image.MAX_IMAGE_PIXELS = 25_000_000`, max 15 MB).
  - Laplacian variance blur detection (rejects blurry images below threshold 8.0).
- **Telemetry Storage**:
  - PostgreSQL 16 storing salted anonymous client hashes (`HMAC-SHA256(user_id, TELEMETRY_SALT)`), timestamp, predicted class, triage tier, and latency. **No raw photos are ever saved to disk or DB**.
  - Ephemeral execution in `tmpfs` RAM disk (`/tmp:size=2G`).

### 2.4 ML Inference & Decoupled Triage Engine (`ml/`)
- **Inference Runtime**: CPU-optimized `ONNX Runtime` with FP32/quantized execution.
- **Canonical Preprocessing**: Standardized letterbox resize (224x224), ImageNet normalization ($[0.485, 0.456, 0.406]$, $[0.229, 0.224, 0.225]$).
- **Temperature Calibration**: Calibrated via validation NLL optimization ($T = 1.5813$), reducing Expected Calibration Error (ECE) from 14.24% to 5.49%.
- **Decoupled 4-Tier Clinical Triage Engine**:
  - Replaces naive heuristic multiplication with deterministic clinical arbitration:
    - 🔴 **`URGENT_REFERRAL`**: Triggered if $P(\text{mel}) \ge \tau_{\text{mel}}$ ($0.05$–$0.20$), or Glasgow score $\ge 3$, or any major symptom reported.
    - 🔵 **`ROUTINE_CONSULTATION`**: Triggered if malignancy risk $\sum P(\text{bcc}, \text{scc}, \text{mel}) \ge 0.20$–$0.30$, or Glasgow score is 1–2.
    - 🟢 **`SELF_MONITORING`**: Benign profile, low malignancy risk, and Glasgow score $= 0$.
    - ⚪ **`INCONCLUSIVE_QUALITY`**: Technical flaw in upload (blur, blank, wrong format).

---

## 3. The Model Artifact Bundle
Model checkpoints are distributed as self-contained bundles:
```
artifacts/models/{profile}/
├── model.onnx            # CPU-optimized ONNX model (16 MB)
├── best_model.pt         # PyTorch checkpoint weights
└── manifest.yaml         # Complete specification, classes, thresholds & SHA-256
```

### Bundle Manifest (`manifest.yaml`) Schema:
```yaml
profile: smartphone_clinical
version: 1.0.0
backbone: efficientnet_b0
classes:
  - bcc
  - scc
  - ack
  - sek
  - mel
  - nev
input_size: [3, 224, 224]
normalization:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
temperature: 1.58128
thresholds:
  tau_abstain: 0.40
  tau_referral_melanoma: 0.20
sha256: 9e66c808308df2a954b6da50bac262e6864997bfee7bb6dd79830da8de3faace
```

---

## 4. API Contract (`POST /v1/predict`)

### Request:
- Form field `file`: binary image data (JPEG, PNG, WebP)
- Form field `profile`: `"smartphone_clinical"` | `"dermoscopy"`
- Form field `glasgow_answers` (optional): JSON string of Glasgow 7-point checklist answers

### Response Body Example:
```json
{
  "profile": "smartphone_clinical",
  "status": "success",
  "predicted_class_id": "nev",
  "predicted_class_name_en": "Melanocytic nevus",
  "predicted_class_name_ru": "Меланоцитарный невус",
  "predicted_class_name_ko": "멜라닌세포 모반",
  "confidence": 0.675,
  "class_probabilities": {
    "bcc": 0.082,
    "scc": 0.015,
    "ack": 0.041,
    "sek": 0.062,
    "mel": 0.025,
    "nev": 0.675
  },
  "melanoma_risk_score": 0.025,
  "referral_recommended": false,
  "latency_ms": 48.2,
  "quality_metrics": {
    "is_valid": true,
    "blur_score": 42.8,
    "dimensions": [750, 750]
  },
  "triage": {
    "decision_tier": "SELF_MONITORING",
    "referral_recommended": false,
    "urgency": "low",
    "rationale": "Benign characteristics, low malignancy risk, zero Glasgow alarm features"
  }
}
```
