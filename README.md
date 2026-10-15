# SkinAI

Research-oriented skin lesion classification and clinical decision-support platform featuring dual-modality computer vision, trilingual localization (RU / EN / KO), a Telegram bot (aiogram 3), an interactive Telegram Mini App (Vite + TypeScript), a FastAPI backend, and rigorous data governance.

> **Clinical & Regulatory Disclaimer**: SkinAI is strictly an experimental research and educational tool. It is **not** a certified medical device and does **not** provide clinical diagnoses or triage advice. It provides model class probability estimates or abstains when image quality is substandard. Always consult a certified dermatologist or oncologist for professional medical evaluation.

---

## Architecture Overview

SkinAI operates under a decoupled, privacy-first microservice architecture:
1. **Client Tier**:
   - **Telegram Bot (`bot/`)**: aiogram 3 bot with trilingual dialogs (🇷🇺 Русский / 🇬🇧 English / 🇰🇷 한국어), Glasgow 7-point checklist anamnesis (NICE NG12), fail-closed user allowlist, and in-memory Doctor Clinical Encounter Memo PDF generator with zero cognitive anchoring.
   - **Telegram Mini App (`miniapp/`)**: Embedded interactive web application (Vite, TypeScript, Vanilla CSS) featuring visual risk gauges, interactive Glasgow progress meters, ABCD educational guides, and dark/light themes.
2. **Service Tier**:
   - **Backend API (`backend/`)**: FastAPI service executing magic byte validation, blur detection (Laplacian variance), ONNX Runtime forward passes, and temperature scaling ($T=1.5813$).
   - **Database**: PostgreSQL 16 storing salted anonymous telemetry (`HMAC-SHA256`). Raw user photos are **never** persisted to disk or DB.
   - **RAM Buffering**: Docker `tmpfs` mounts (`/tmp:size=2G`) prevent disk image leakage.
3. **ML Core (`ml/`)**:
   - `smartphone_clinical`: Close-up consumer smartphone photos of skin lesions (PAD-UFES-20 baseline, 6 classes: `bcc`, `scc`, `ack`, `sek`, `mel`, `nev`).
   - `dermoscopy`: Magnified illuminated dermoscopic images (ISIC Archive CC0/CC-BY baseline).
   - **Decoupled 4-Tier Clinical Triage Engine**:
     - 🔴 `URGENT_REFERRAL` (Priority consultation within 1–2 weeks)
     - 🔵 `ROUTINE_CONSULTATION` (Scheduled dermatologist check)
     - 🟢 `SELF_MONITORING` (Routine self-monitoring)
     - ⚪ `INCONCLUSIVE_QUALITY` (Quality rejection)

---

## System Requirements
- **Development Workstation**: Windows 11 with NVIDIA GPU (RTX 3060 12GB) for local model training.
- **Python**: 3.11.x (64-bit).
- **Node.js**: 20+ (for `miniapp/` development).
- **Inference Server (VPS)**: Ubuntu 22.04 / 24.04 (2–4 vCPU, 4–8 GB RAM, Docker & Docker Compose).

---

## Quickstart

### 1. Verify Environment
```powershell
python scripts/check_environment.py
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your credentials:
```powershell
Copy-Item .env.example .env
```

### 3. Run Backend & Bot Tests
```powershell
pytest -v
```
*(74 passed in ~14s, 100% ruff clean)*

### 4. Run Telegram Mini App Locally
```powershell
cd miniapp
npm install
npm run dev
```

### 5. Production Docker Staging
```bash
docker compose --env-file .env -f deploy/docker-compose.prod.yml config
docker compose --env-file .env -f deploy/docker-compose.prod.yml up -d --build
```

---

## Repository Structure
```text
Dermatology/
├── AGENTS.md                  # Root rules for Antigravity agents
├── .agents/                   # Modular rules and role skill definitions
├── artifacts/                 # Versioned model ONNX checkpoints & validation benchmarks
│   ├── benchmarks/            # 5-fold CV results & clinical validation JSONs
│   └── models/                # EfficientNet-B0 ONNX bundles & manifest.yaml
├── backend/app/               # FastAPI inference service & PostgreSQL telemetry
├── bot/                       # aiogram 3 Telegram bot & in-memory PDF generator
│   └── locales/               # ru.json, en.json, ko.json (100% parity)
├── configs/                   # Profile YAML configurations (dermoscopy, smartphone)
├── data/                      # Dataset splits and metadata (PAD-UFES-20)
├── deploy/                    # Dockerfile.api, Dockerfile.bot, docker-compose.prod.yml, Nginx
├── docs/                      # Architectural, scientific & clinical documentation
│   ├── ARCHITECTURE.md        # Comprehensive system specification
│   ├── DATA_GOVERNANCE.md     # Provenance and licensing register
│   ├── DECISIONS.md           # Architectural Decision Records (ADRs 001–010)
│   ├── DEPLOYMENT.md          # VPS Staging and production runbook
│   ├── EXECUTIVE_SUMMARY_FOR_COLLEAGUES.md # Consilium summary & clinical review
│   ├── MODEL_CARD_SMARTPHONE_CLINICAL.md  # Official model card
│   ├── SCIENTIFIC_BENCHMARK_REPORT.md     # 5-Fold Stratified Cross-Validation
│   ├── SCIENTIFIC_VALIDATION_REPORT.md    # Multi-threshold test set calibration report
│   ├── SECURITY.md            # STRIDE threat model & privacy architecture
│   └── STATUS.md              # Project status and roadmap gate tracking
├── miniapp/                   # Telegram Mini App (Vite + TypeScript + Vanilla CSS)
├── ml/                        # Canonical preprocessing, PyTorch models, ONNX inference
├── scripts/                   # Verification, training & evaluation utilities
└── pyproject.toml             # Project dependency specification
```
