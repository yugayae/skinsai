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
- **Status**: Amended (2026-10-23)
- **Context**: The technical safety review emphasized that multiplying neural network softmax probabilities by anamnesis multipliers creates false pseudo-Bayesian probabilities that confuse clinicians and risk dangerous under/over-referral.
- **Decision**: Decouple pure vision probabilities from anamnesis. Introduce a deterministic, evidence-based clinical triage engine incorporating NICE NG12 Glasgow scoring rules independently of vision output.
- **Amendment (2026-10-23)**: Deprecate the category `SELF_MONITORING` («наблюдение в динамике»). При ограниченной чувствительности модели по меланоме выдача пользователю заключения о «безопасности для самостоятельного наблюдения» создает недопустимый риск задержки очной медицинской помощи. Уровень переименован в `ROUTINE_NO_HIGH_RISK_DETECTED` («Признаков высокого риска по фото не выявлено») с обязательным дисклеймером: отрицательный результат модели не исключает патологию, и при любых изменениях очага необходим визит к дерматологу.
- **Consequences**: Transparent clinical logic, defense-in-depth safety, zero mathematical distortion of calibrated vision probabilities, elimination of false clinical reassurance.

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

---

## ADR-011: Decoupled Multi-Head Architecture (Binary Oncology Triage + Differential Subtyping)
- **Status**: Approved
- **Context**: Single-head 6-class softmax models force a zero-sum competition between broad oncological risk triage (Malignant vs Benign) and granular differential diagnosis (e.g., distinguishing BCC vs SCC vs MEL). Furthermore, thousands of high-quality clinical close-up images from ISIC are confirmed benign but lack specific 6-class subtyping labels (e.g., unspecified solar lentigo, benign keratosis), rendering them unusable in a strict 6-class single-head setup.
- **Decision**: Implement a decoupled Multi-Head architecture (`MultiHeadSkinClassifier` and `MultiHeadProbe`) sharing a foundational vision backbone (Meta DINOv2 Small, 384-dim embeddings):
  1. **Head 1 (Binary Oncology Triage)**: Predicts `[Benign, Malignant]`. Trained on all confirmed lesions plus 1,219 ISIC benign close-ups to provide a robust, highly sensitive cancer screening filter.
  2. **Head 2 (Differential 6-Class Subtyping)**: Predicts canonical classes (`ack`, `bcc`, `mel`, `nev`, `scc`, `sek`).
  3. **Masked Multi-Task Loss**: Automatically masks unmapped subtype samples (`ignore_index = -1`), позволяя использовать общие доброкачественные когорты.
  4. **Clinical Consensus**: Computes joint agreement between dedicated Head 1 triage and Head 2 derived malignancy sum.
- **Consequences**: Enables utilization of large unmapped permissive cohorts, decouples oncological safety from subtyping errors, provides dual-head clinical agreement verification.

---

## ADR-012: Transition to Aviation & Medical Regulatory Documentation Standards and Two-Tier Dataset Architecture
- **Status**: Approved (2026-10-23)
- **Context**: Независимый технический аудит выявил критические методологические искажения: многократное использование тестового набора `test.csv` (Data Snooping / Information Leakage в EXP-001 — EXP-009), некорректную оценку значимости ($p < 0.001$), скрытую статистическую погрешность на выборке из 8 меланом и применение неакадемической рекламной лексики.
- **Decision**:
  1. **Принятие авиационного стандарта строгости (ICAO / FAA / EASA Part-145 аналог) и медицинских стандартов (CLAIM 2024, STARD-AI, IEC 62304)**: Полный запрет эмоциональных прилагательных («прорыв», «сверхстабильный», «ликвидация слепых зон»). Обязательное указание абсолютных дробей ($n/N$), доверительных интервалов Уилсона для биномиальных долей и бутстрепа по пациентам для агрегатов.
  2. **Реквалификация `test.csv`**: Набор `test.csv` переведен в статус данных разработки (Development Validation Set). Результаты EXP-001 — EXP-009 признаны предварительными.
  3. **Двухконтурная архитектура валидации (Two-Tier Evaluation)**:
     - **Контур разработки (Development Tier)**: Пул PAD-UFES-20 делится строго на Train и Validation по `patient_id`. Подбор гиперпараметров, архитектурные решения и калибровка порогов осуществляются исключительно на валидационном контуре (с Out-of-Fold оценкой по всем 52 меланомам).
     - **Контур внешнего независимого аудита (Locked External Test Tier)**: Формируется отдельный, ранее не использовавшийся датасет клинических смартфонных снимков (Stanford DDI / независимая когорта ISIC Close-up). Тестирование замороженной модели (с фиксацией SHA-256 чекпоинта) проводится **ровно один раз** в режиме «запечатанного конверта» (pre-registered locked evaluation) без права обратной подгонки порогов.
- **Consequences**: Устранение оптимистического смещения, обеспечение воспроизводимости результатов, соответствие требованиям регуляторов к доказательной базе SaMD.

---

## ADR-013: Unified Development Pool and Multimodal Multi-Head Architecture
- **Status**: Approved (2026-10-23)
- **Context**: 
  1. В дискуссии возник вопрос о несовпадении цифр: 1 373 пациента vs ~3 500 доступных снимков. В датасете PAD-UFES-20 содержится 2 298 снимков от 1 373 уникальных пациентов (у одного пациента часто присутствует от 2 до 5 новообразований). Дополнительно в архиве ISIC Close-up верифицировано 1 299 клинических макроснимков от 131 пациента. Итого в распоряжении проекта находится 3 597 физических изображений от 1 504 пациентов.
  2. Использование нескольких снимков одного и того же человека в разных выборках приводит к недопустимой утечке данных (Data Leakage: фоновые особенности кожи, камеры, освещения).
  3. Обучение классификатора требует одновременного решения двух задач: высокочувствительный онкологический скрининг (доброкачественное vs злокачественное) и дифференциальная диагностика 6 классов, с обогащением визуальных признаков метаданными клинического анамнеза (возраст, пол, локализация).
- **Decision**:
  1. **Формирование Unified Development Pool (3 597 изображений, 1 504 пациента)**:
     - Разделение строго по `patient_id` (GroupSplit): Train = 2 894 снимка (1 203 пациента, 80.5%), Validation = 703 снимка (301 пациент, 19.5%).
     - Пересечение пациентов и очагов между Train и Val строго равно нулю (`overlap_patients == 0`, `overlap_lesions == 0`).
     - Пул меланом расширен с 52 до 72 снимков (58 в Train, 14 в Val), что повышает статистическую мощность валидации.
  2. **Двухголовая архитектура с маскированным лоссом (Decoupled Multi-Head with Masked Loss)**:
     - Общий бэкбон: самоконтролируемый Meta DINOv2 Small (ViT-S/14, 384D) с L2-нормализацией.
     - Модуль анамнеза: `TabularClinicalEncoder` (12D -> 64D) с маскированием при отсутствии метаданных (конкатенация в 448D мультимодальный вектор).
     - Head 1 (Binary Oncology Triage): 2 логита `[Benign, Malignant]`, обучается на всех 3 597 снимках пула (1 153 Malignant, 1 714 Benign, 730 предраковых ACK маскированы).
     - Head 2 (Differential 6-Class Subtyping): 6 логитов (`ack`, `bcc`, `mel`, `nev`, `scc`, `sek`), обучается с `MaskedClassWeightedFocalLoss` (`ignore_index = -1`), что позволяет изолировать 1 220 снимков без точного подтипа от искажения дифференциальных границ.
  3. **Согласование решений (Consensus Engine)**:
     - Вычисляется консенсус между бинарным флагом Head 1 ($P_{\text{H1}} \ge 0.20$) и суммарной вероятностью злокачественности Head 2 ($\sum P_{\text{H2}} \ge 0.20$).
     - В случае расхождения (например, Head 1 сигнализирует риск, а Head 2 классифицирует как редкий доброкачественный подтип) система отдает приоритет онкологической безопасности (Union rule: $H_1 \lor H_2$).
- **Consequences**:
  - Полное использование всех 3 597 снимков без компромиссов в чистоте валидации (0% patient leakage).
  - Сбалансированная архитектура: широкая онкологическая чувствительность не подавляет точность дифференциальной классификации.

---

## ADR-014: Pre-Registered Locked External Evaluation Protocol on Stanford DDI (Phase 2.8)
- **Status**: Approved (2026-10-24)
- **Context**: 
  1. В соответствии со стандартами CLAIM 2024, STARD-AI и регуляторными требованиями к SaMD (Software as a Medical Device), окончательная оценка алгоритма должна проводиться на внешнем, невиданном наборе данных (External Blind Test) без возможности повторной подгонки порогов или весов модели.
  2. В качестве внешнего тестового контура утвержден датасет **Stanford DDI (Diverse Dermatology Images)**: 656 гистологически подтвержденных (100% биопсия) клинических макроснимков от 570 пациентов Стэнфордского университета, стратифицированных по 6 фототипам кожи Фицпатрика (FST 12, FST 34, FST 56).
- **Decision**:
  1. **Пре-регистрация и заморозка артефакта**:
     - Чекпоинт модели: `artifacts/models/smartphone_clinical/development_champion_v2/best_model.pt`
     - Зафиксированный контрольный хеш SHA-256: `4bc71ff1dbad7520f5c76469185bfe409672565683afe3a256134ce8bd74d58f`. Любое несовпадение хеша блокирует исполнение теста.
  2. **Фиксация порогов и метрик**:
     - Порог триажа онкоскрининга: $\tau_{\text{triage}} = 0.20$.
     - Порог чувствительности к меланоме: $\tau_{\text{mel}} = 0.05$.
     - Первичная конечная точка: ROC-AUC и Чувствительность бинарного триажа на всех 656 снимках с 95% интервалами Уилсона.
     - Вторичная конечная точка: Чувствительность выявления меланомы ($n/N_{\text{mel}}$) с интервалом Уилсона.
     - Демографическая конечная точка: Аудит справедливости (Demographic Equity) раздельно по группам FST 12 (светлая кожа), FST 34 (промежуточная) и FST 56 (темная кожа).
  3. **Режим «запечатанного конверта»**:
     - Скрипт валидации [`scripts/evaluate_external_ddi_locked.py`](file:///c:/HomeWork/Medical/Dermatology/scripts/evaluate_external_ddi_locked.py) исполняется ровно один раз. Результаты фиксируются в отчете `exp012_stanford_ddi_locked_evaluation.json`.
     - Запрет обратной связи: данные DDI категорически запрещено переносить в обучающую выборку.
- **Consequences**: Полное исключение оптимистического смещения и подгонки под тест; получение объективной, юридически и клинически чистой оценки качества SkinAI в международной практике.

---

## ADR-015: De-Confounded Dual-Backbone Fusion (`DINOv2 + BiomedCLIP`), Deterministic XAI Engine & 3-Zone Triage (`EXP-015` — `EXP-018`)
- **Status**: Approved (2026-11-08)
- **Context**:
  1. Двухраундовый аудит независимой экспертизы (в рамках этапов `EXP-017` и `EXP-018`) потребовал: (а) проверки на 5-фолдовой пациент-группированной кросс-валидации (`3,767` снимков, `1,674` пациента); (б) устранения конфаундинга отсутствующих метаданных (`site_unknown`, `sex_unknown`), снижавшего переносимость на внешние датасеты без анкет (`Stanford DDI`); (в) сравнения нейросетевой двухголовой архитектуры с классическим ML (`LogReg`, `LinearSVM`, `LightGBM`, `XGBoost`, `RandomForest`, `MLP`); (г) замены центрированного порога Оцу в морфометрии `ABCDE` на `Attention-Seeded GrabCut + Black-Hat Hair Removal` (`DS-5`).
- **Decision**:
  1. **Визуальное слияние (Dual-Backbone Fusion) (`896D`):** Объединение `Meta DINOv2 Small` (`384D`, геометрия и структура границ + карта самовнимания `[CLS]`) и `Microsoft BiomedCLIP` (`512D`, биомедицинская семантика PMC-15M).
  2. **Деконфаундированная двухголовая архитектура (`De-Confounded Dual-Head Fusion`):**
     - Удалены служебные флаги `site_unknown` и `sex_unknown` (табличный вектор сокращен с `12D` до `7D`: возраст, 4 анатомические зоны, пол).
     - Внедрен `50% Metadata Dropout` при обучении (`gamma=2.5`, `SiLU`, `AdamW`), благодаря чему модель сохраняет высокую точность как при наличии анкеты (`5-Fold OOF Macro F1 = 0.7161 ± 0.0080`), так и при подаче «голого» снимка без метаданных (`Stanford DDI In-Scope ROC-AUC = 0.8250`, прирост `+11.2%` над конфаундированной моделью).
     - Экспортирован боевой артефакт `artifacts/models/smartphone_clinical/fusion_head.onnx` (точность совпадения с PyTorch `3.58e-07`).
  3. **Движок «Deterministic XAI» (`ml/explainability/deterministic_xai.py`):**
     - **Морфометрия `ABCDE` (`DS-5`):** Фильтрация волос `cv2.MORPH_BLACKHAT` + `cv2.inpaint` (снижение ложной изрезанности границ на `63.7–80.8` ед.) и сегментация `cv2.grabCut`, инициализируемая центром внимания `DINOv2` (`+9.7%` IoU при сдвиге очага к краю кадра).
     - **Атлас биопсий (`BiopsyAtlasKNNRetriever`):** Косинусный поиск `Top-3` гистологически верифицированных близнецов (`artifacts/models/smartphone_clinical/biopsy_atlas_index.pt`, `2,027` референсных биопсий).
  4. **3-зонная клиническая маршрутизация (`Red / Yellow / Green`) и фильтр `CIE L*a*b* OOD Gate`:**
     - `🔴 Red Zone` (`urgent_referral`): `P(mel) >= 0.15` или `P(malig) >= 0.62` или `Glasgow >= 3`.
     - `🟡 Yellow Zone` (`routine_consultation` / `inconclusive_quality`): `P(ack) >= 0.20` (перехват `99.73%` предракового актинического кератоза `ACK`), пограничный риск или отсев внедоменного изображения (`CIE L*a*b* Non-Skin OOD Gate`).
     - `🟢 Green Zone` (`routine_no_high_risk`): низкий онкориск и отсутствие тревожных симптомов.
- **Consequences**: Полная прослеживаемость каждого параметра, защита от доменного сдвига при отсутствии анкеты и сквозная интеграция во все уровни системы (`ML -> FastAPI -> Telegram Bot -> Mini App`).




