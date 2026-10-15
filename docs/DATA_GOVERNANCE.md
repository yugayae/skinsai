# Data Governance, Provenance & Licensing Register

## 1. Governance Principles
- Raw datasets are immutable and stored strictly outside version control.
- Image records must include explicit provenance tracking: source, original diagnosis, diagnostic confirmation type (e.g., histopathology, clinical consensus, follow-up), and licensing rights.
- Any model bundle slated for production release must be derived solely from data marked `commercial_ok` or `public_ok`.

---

## 2. Dataset Registry

### Registry Entry D-01: PAD-UFES-20 (Smartphone Clinical Profile)
- **Source**: Mendeley Data / Federal University of Espírito Santo (UFES), Brazil.
- **Publication**: Pacheco, A. G., et al. "PAD-UFES-20: A skin lesion dataset composed of patient data and images collected with smartphones." Data in Brief 32 (2020): 106221.
- **License**: Creative Commons Attribution 4.0 International (`CC BY 4.0`).
- **Commercial Feasibility**: Permitted under CC BY 4.0 with mandatory attribution. Status: `commercial_ok`.
- **Modality**: Smartphone cameras (various consumer smartphones: iPhone, Samsung Galaxy, Motorola).
- **Volume**: 2,298 images, 1,891 lesions, 1,373 patients (Total download footprint: ~220 MB).
- **Classes**:
  1. `BCC`: Basal Cell Carcinoma (845 images, 36.8%)
  2. `SCC`: Squamous Cell Carcinoma (192 images, 8.4%)
  3. `ACK`: Actinic Keratosis (730 images, 31.8%)
  4. `SEK`: Seborrheic Keratosis (235 images, 10.2%)
  5. `MEL`: Melanoma (52 images, 2.3%)
  6. `NEV`: Melanocytic Nevus (244 images, 10.6%)
- **Grouping Key**: `patient_id` (1,373 unique patients).
- **Zero-Leakage Split Generated**:
  - `Train`: 1,599 images (69.6%), 962 patients (BCC: 588, ACK: 512, NEV: 166, SEK: 166, SCC: 133, MEL: 34)
  - `Val`: 353 images (15.4%), 206 patients (BCC: 130, ACK: 111, NEV: 38, SEK: 32, SCC: 32, MEL: 10)
  - `Test`: 346 images (15.1%), 205 patients (BCC: 127, ACK: 107, NEV: 40, SEK: 37, SCC: 27, MEL: 8)
  - **Leakage Audit**: `len(train_pts & val_pts) == 0`, `len(train_pts & test_pts) == 0`, `len(val_pts & test_pts) == 0` (100% isolated).
  - **Manifest File**: `data/splits/smartphone_clinical/manifest.csv` (Hash: `e01475d0b3a2191a...`).

---

### Registry Entry D-02: ISIC Archive Permissive Sub-cohort (Dermoscopy Profile)
- **Source**: International Skin Imaging Collaboration (ISIC) Archive API.
- **License Structure**: The ISIC Archive hosts multiple datasets with heterogeneous licenses:
  - `CC-0`: Public Domain dedication.
  - `CC-BY`: Attribution 4.0.
  - `CC-BY-NC`: Non-Commercial (e.g. ISIC 2018 Task 3 / HAM10000).
- **Commercial Feasibility**:
  - Filtered sub-cohort containing ONLY images flagged with `copyright_license in ['CC-0', 'CC-BY']` qualifies for `commercial_ok` / `public_ok`.
  - The legacy HAM10000 / ISIC 2018 challenge training set contains CC BY-NC images and is strictly tagged `research_only`.
- **Target Diagnostic Taxonomy**:
  1. `MEL`: Melanoma
  2. `NV`: Melanocytic Nevus
  3. `BCC`: Basal Cell Carcinoma
  4. `AKIEC`: Actinic Keratosis / Intraepithelial Carcinoma
  5. `BKL`: Benign Keratosis (Seborrheic Keratosis, Solar Lentigo, Lichen-like Keratosis)
  6. `DF`: Dermatofibroma
  7. `VASC`: Vascular Lesions
- **Grouping Key**: `lesion_id` and `patient_id` where available.
- **Action for Phase 1**: Execute an automated metadata audit via the ISIC REST API to determine exact image counts by class under `CC-0` and `CC-BY` licenses.

---

## 3. Disallowed / Excluded Datasets
| Dataset | Reason for Exclusion |
|---|---|
| **DermNet** | Terms of service explicitly forbid training AI/ML models without commercial licensing agreement. |
| **Fitzpatrick17k** | License terms require non-commercial academic research agreement. |
| **DDI (Diverse Dermatology Images)** | Restricted non-commercial academic agreement. |
