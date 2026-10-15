# Architectural Decision Records (ADR)

## ADR-001: Dual-Profile Separation (`dermoscopy` & `smartphone_clinical`)
- **Status**: Approved
- **Context**: Dermoscopic images (specialized illuminated magnification) differ fundamentally in optical characteristics, artifacts, and patient population from smartphone cameras. Training a single mixed network risks severe domain shift and inflated offline test accuracy that degrades on real phones.
- **Decision**: Treat dermoscopy and smartphone clinical imaging as two distinct profiles with separate models, separate datasets, independent evaluation benchmarks, and distinct release gates.
- **Consequences**: Avoids deceptive performance claims. Allows each profile to advance independently.

---

## ADR-002: Interactive User Profile Selection in MVP
- **Status**: Approved
- **Context**: An automatic classifier distinguishing "dermoscopy vs smartphone vs non-skin" introduces its own misclassification rate and development complexity.
- **Decision**: In the initial MVP, the user explicitly chooses the image type via an interactive keyboard in Telegram (`Дерматоскоп` vs `Фото со смартфона`). Auto-detection is deferred to future iterations.
- **Consequences**: Simpler, auditable UX. Explicit user intent.

---

## ADR-003: Commercial Licensing Prudence (`license_scope`)
- **Status**: Approved
- **Context**: Open research datasets have divergent licensing terms. Many ISIC challenge archives (e.g., ISIC 2018 Task 3) carry `CC BY-NC` (Non-Commercial). Calling a bot "free" does not immunize against commercial disputes if future monetization or service association is planned.
- **Decision**: Establish strict `license_scope` metadata (`research_only`, `public_ok`, `commercial_ok`). Only data sub-cohorts with verifiable permissive licenses (e.g. ISIC CC0/CC-BY filtered subset, PAD-UFES-20 CC BY 4.0) will be used for the production models. Non-commercial data is segregated strictly for research baseline comparisons.
- **Consequences**: Clean legal provenance for future monetization.

---

## ADR-004: Training on Windows GPU, Inference on Linux CPU via ONNX Runtime
- **Status**: Approved
- **Context**: The developer machine possesses an NVIDIA RTX 3060 12GB GPU running Windows 11. Production deployment will reside on a cost-effective Linux VPS without a dedicated GPU.
- **Decision**: Perform model training natively in Windows using PyTorch + CUDA + mixed precision (AMP). Export approved checkpoints to ONNX Runtime FP32/quantized format. Deploy the API and Telegram Bot on Linux VPS via Docker Compose utilizing CPU ONNX Runtime.
- **Consequences**: Minimizes VPS hosting expenses. Keeps training fast locally. Requires explicit parity testing between PyTorch and ONNX outputs.

---

## ADR-005: Telegram-First Interface with Deferred Mini App
- **Status**: Approved (Superseded by ADR-010 for Phase 7 implementation)
- **Context**: Building a full web application or Telegram Mini App before validating the ML pipeline and bot core adds frontend surface area and security overhead.
- **Decision**: Phase 4 focused on a Telegram Bot (aiogram 3). Phase 7 implemented a Telegram Mini App (Vite + TypeScript) with interactive cards after the backend API and bot flows were proven.
- **Consequences**: Fast MVP launch followed by rich interactive web experience.

---

## ADR-006: Zero Image Retention Policy
- **Status**: Approved
- **Context**: User privacy and medical data sensitivity require minimizing data footprint.
- **Decision**: Incoming user photos are analyzed in memory and immediately discarded. No persistent storage of raw or processed user images. Only anonymized telemetry (salted hash of user ID, timestamp, profile, predicted top classes, latency) is persisted to PostgreSQL.
- **Consequences**: Enhanced user trust, reduced compliance exposure.

---

## ADR-007: Decoupled 4-Tier Clinical Triage Engine (Sprint P0)
- **Status**: Approved
- **Context**: The consilium consensus (Claude, GPT, Gemini, DeepSeek, Grok) emphasized that multiplying neural network softmax probabilities by anamnesis multipliers creates false pseudo-Bayesian probabilities that confuse clinicians and risk dangerous under/over-referral.
- **Decision**: Decouple pure vision probabilities from anamnesis. Introduce a deterministic, evidence-based 4-tier clinical triage engine (`URGENT_REFERRAL`, `ROUTINE_CONSULTATION`, `SELF_MONITORING`, `INCONCLUSIVE_QUALITY`) incorporating NICE NG12 Glasgow scoring rules independently of vision output.
- **Consequences**: Transparent clinical logic, defense-in-depth safety, zero mathematical distortion of calibrated vision probabilities.

---

## ADR-008: In-Memory Clinical Encounter Memo PDF with Zero Cognitive Anchoring (Sprint P1)
- **Status**: Approved
- **Context**: When patients present AI classification scores (e.g., "78% melanoma" or "90% benign") to an examining dermatologist, clinicians may experience cognitive anchoring bias, compromising their objective dermoscopic evaluation.
- **Decision**: Provide an in-memory PDF export designed specifically for in-person doctor visits. The memo contains the lesion photograph, date, Glasgow 7-point checklist responses, suggested consultation questions, and regulatory disclaimer, while strictly omitting raw model softmax percentages and AI disease assertions.
- **Consequences**: Preserves doctor objectivity, bridges telemedicine with clinical practice, zero permanent storage on server disk.

---

## ADR-009: Trilingual Localization Architecture (RU / EN / KO)
- **Status**: Approved
- **Context**: SkinAI serves multi-regional research and clinical needs. The project requires Russian as the primary interface language, English as the international standard, and Korean for clinical and medical-technological parity.
- **Decision**: Implement a unified 3-language localization architecture across the Telegram Bot (`ru.json`, `en.json`, `ko.json`), FastAPI responses (`CLASS_DISPLAY_NAMES`), PDF reports (supporting TrueType Hangul fonts `Malgun Gothic` and `NanumGothic`), and Telegram Mini App.
- **Consequences**: Broad accessibility, robust TrueType font discovery across Windows and Linux Docker containers.

---

## ADR-010: Telegram Mini App Implementation (Phase 7)
- **Status**: Approved
- **Context**: Users and clinicians benefit from an interactive visual summary of lesion features, Glasgow risk score meters, and educational ABCD guides directly embedded inside Telegram.
- **Decision**: Build `miniapp/` using Vite, TypeScript, and Vanilla CSS. Integrate Telegram WebApp SDK (`HapticFeedback`, `MainButton`, `themeParams`), implement trilingual UI toggles, and provide client-side PDF export fallback via `jspdf`.
- **Consequences**: High-quality visual presentation, lightweight zero-framework frontend, mobile-responsive layout.
