# Project Status & Phase Tracking: SkinAI

## Current State: Phases 0–7 & Sprints P0–P1 Completed / Production Staging Ready
- **Active Phase**: Phase 6 (VPS Deployment, Staging & Docker Orchestration)
- **Primary Workstation**: Windows 11, NVIDIA RTX 3060 12GB (Driver: 616.92), Python 3.11.9, Docker 29.4.2
- **Last Updated**: 2026-10-14

---

## Phase Roadmap & Gates

| Phase | Title | Lead Role | Status | Acceptance Gate |
|---|---|---|---|---|
| **Phase 0** | **Foundation & Architecture** | Tech Lead | **COMPLETED** | `AGENTS.md`, `.agents/rules/`, `.agents/skills/`, `docs/`, `configs/` created. Workspace established. |
| **Phase 0.5** | **Environment & CUDA Verification** | ML Engineer | **COMPLETED** | Python 3.11.9, RTX 3060 12GB (Driver 616.92), PyTorch 2.5.1+cu121 verified; 15/15 unit tests passing. |
| **Phase 1** | **Data Governance & Split Pipeline** | Data Engineer | **COMPLETED** | PAD-UFES-20 (Collection 406, CC-BY, 2,298 images, 1,373 patients) split with 0 patient leakage into Train/Val/Test; Dermoscopy CC-0/CC-BY cohorts cataloged; core `ml/datasets/` pipeline complete. |
| **Phase 2** | **ML Training, Calibration & ONNX Export** | ML Engineer | **COMPLETED** | EfficientNet-B0 trained on RTX 3060 with Focal Loss; Temperature calibrated (T=1.581); ONNX parity verified (diff < 1e-4); 23/23 tests pass; Model card and OnnxPredictor ready. |
| **Phase 3** | **FastAPI Backend & Inference Service** | Backend Engineer | **COMPLETED** | API contracts implemented; validator and telemetry working; OnnxPredictor integrated; health & prediction endpoints verified; 36/36 tests passing. |
| **Phase 4** | **Telegram Bot MVP (RU / EN / KO)** | Telegram Engineer | **COMPLETED** | aiogram 3 bot implemented; localized dialog flows (RU/EN/KO); Glasgow checklist FSM; ephemeral zero-disk image ingestion; live bot @Ai_skins_bot tested; 46/46 tests passing. |
| **Phase 5** | **QA, Security & Adversarial Auditing** | QA/Security Engineer | **COMPLETED** | STRIDE threat model in `docs/SECURITY.md`; decompression bomb, polyglot, fuzzing tests; 68/68 automated tests pass; ruff 100% clean. |
| **Sprint P0** | **Clinical Safety & Decoupled Triage (Consilium Consensus)** | Tech Lead / ML / Backend | **COMPLETED** | Eliminated heuristic probability multiplication; implemented 4-tier clinical triage engine (`URGENT_REFERRAL`, `ROUTINE_CONSULTATION`, `SELF_MONITORING`, `INCONCLUSIVE_QUALITY`); bot fail-closed allowlist; tmpfs buffering; 69/69 automated tests pass. |
| **Sprint P1** | **Clinical Encounter Memo PDF (Zero Anchoring)** | Telegram / ML / UX | **COMPLETED** | In-memory multilingual PDF generator for in-person doctor visits (RU/EN/KO); embedded lesion photo, Glasgow checklist, consultation questions; zero model probability leakage; 74/74 tests pass. |
| **Phase 7** | **Telegram Mini App** | Frontend / UX | **COMPLETED** | Vite + TypeScript Mini App with trilingual UI (RU/EN/KO), 4-tier triage cards, interactive Glasgow 7-point checklist, ABCD dermatologist guide, dark/light theme, and in-memory client PDF doctor memo export. |
| **Phase 2.5** | **Scientific Test Calibration & Safety Thresholds** | ML Engineer | **COMPLETED** | Evaluated 346 test images (205 patients); Macro ROC-AUC 0.8552; ECE reduced to 5.49% (-61.4%); melanoma recall up to 75% at tau=0.05; malignancy recall 95.68% at tau=0.20; `docs/SCIENTIFIC_VALIDATION_REPORT.md` published. |
| **Phase 6** | **VPS Deployment & Production Staging** | Tech Lead | **ACTIVE** | Docker Compose operational with healthchecks, Nginx reverse proxy template, staging deployment runbook. |

---

## Active Blockers & Resolutions
1. **Drive Space on `C:\`**: `C:\` has ~24.9 GB available. Resolved by configuring `.env` storage redirection for `data/` and `artifacts/` to secondary storage (drive `G:\` or designated external directory).
2. **Terminal Sandbox**: Terminal execution verified working in Antigravity.
