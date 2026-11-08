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
### Registry Entry D-03: ISIC Archive Permissive Clinical Close-up Cohort (Smartphone Clinical Profile Extension)
- **Source**: International Skin Imaging Collaboration (ISIC) Archive API (AWS S3 Image Repository).
- **Modality**: `clinical: close-up` (macroscopic / clinical camera & smartphone photography).
- **License**: Creative Commons CC-0 and CC-BY 4.0 (`commercial_ok` / `public_ok`).
- **Volume**: 1,299 non-overlapping images (1,774.24 MB), 0 failures during ingestion.
- **Physical Location**: `data/raw/isic_clinical/`
- **Diagnostic Distribution**:
  - `Benign / Benign - Other`: 1,217 images (histopathologically & clinically confirmed negative controls)
  - `BCC` (Basal Cell Carcinoma): 29 images
  - `MEL` (Melanoma): 20 images
  - `SCC` (Squamous Cell Carcinoma): 15 images
  - `NEV` (Melanocytic Nevus): 15 images
  - `Scar / Atypical`: 3 images
- **Manifest & Hash Audit**:
  - `data/metadata/isic_clinical_manifest.csv` (complete 3,597 close-up inventory)
  - `data/metadata/isic_clinical_download_log.csv` (SHA-256 integrity logs for 1,299 files)
- **Role in Architecture**: Direct reinforcement of minority malignant classes and primary negative baseline for the decoupled Binary Malignancy / Triage Head (`malignant vs benign`).

### Registry Entry D-04: Unified Development Pool (PAD-UFES-20 + ISIC Close-up Permissive + MED-NODE)
- **Source**: Harmonized pool of PAD-UFES-20 (D-01), ISIC Clinical Close-up (D-03), and MED-NODE (D-05).
- **Volume**: 3,767 confirmed physical images across 1,674 unique patient clusters.
- **License Compliance**: CC-BY 4.0, CC-0, and Research Attribution (Giotis et al., 2015).
- **Taxonomy & Targets**:
  - 6 differential classes (`bcc`, `scc`, `mel`, `ack`, `sek`, `nev`) + `unmapped` (general benign cohort).
  - Binary triage target (`1`=Malignant: 1,223; `0`=Benign: 1,814; `-1`=Masked ACK: 730).
- **Zero-Leakage Split (Stratified Group by `patient_id`)**:
  - Grouped strictly by patient ID to ensure 0% patient leakage.
  - Melanoma pool: 142 confirmed melanomas (nearly double the original 72).
- **Physical Manifests**:
  - `data/splits/development_pool_v2/manifest.csv`
  - `data/splits/development_pool_v2/train.csv`
  - `data/splits/development_pool_v2/val.csv`

---

### Registry Entry D-05: MED-NODE (Macroscopic Melanoma & Naevi Dataset)
- **Source**: Department of Dermatology, University Medical Center Groningen (UMCG), Netherlands.
- **Publication**: Giotis, I., et al. "MED-NODE: A computer-assisted melanoma diagnosis system using non-dermoscopic images." Expert Systems with Applications 42.17-18 (2015): 6578-6585.
- **License**: Academic Research Use with mandatory citation. Status: `research_ok`.
- **Modality**: Macroscopic (clinical digital camera photography, non-dermoscopic).
- **Volume**: 170 images (24.67 MB, SHA-256 archive: `e2dde3d04fee3402e821d2dbc343344685fe2acc63248d3b246986ad54f89dc6`):
  - `Melanoma`: 70 images (100% histopathologically confirmed via surgical biopsy at UMCG).
  - `Naevus`: 100 images (clinically and histopathologically verified melanocytic naevi).
- **Physical Location**: `data/raw/mednode/`
- **Manifest**: `data/metadata/mednode_manifest.csv`
- **Role in Architecture**: Direct reinforcement of the critical minority class `mel` (melanoma), resolving sample scarcity in macroscopic photography.

---

## 3. Disallowed / Excluded Datasets in Training
| Dataset | Reason for Exclusion from Production Training | Permissible Research Use |
|---|---|---|
| **DermNet** | Terms of service explicitly forbid training AI/ML models without commercial licensing agreement. | Strictly prohibited. |
| **Fitzpatrick17k** | CC BY-NC license; high label noise from web atlas scraping. | Excluded from training; exploratory demographic audit only. |
| **DDI (Diverse Dermatology Images)** | Released under CC BY-NC 4.0. | Excluded from production model weights. Permissible strictly as read-only external benchmark for research audit with attribution. |

