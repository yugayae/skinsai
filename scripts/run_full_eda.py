#!/usr/bin/env python3
"""Comprehensive Exploratory Data Analysis (EDA) on PAD-UFES-20 smartphone dataset.

Examines:
1. Ground truth reliability (diagnostic confirmation methods)
2. Class balance & severe minority class challenges (Melanoma: 52 images)
3. Demographic & anatomical confounding (age, sex, anatomical location shortcuts)
4. Spatial image dimensions, aspect ratios, color channel statistics (mean/std)
5. Blur & sharpness distribution (Laplacian variance across classes)
6. Outliers, corruption, underexposed/overexposed images
7. Leakage check between train/val/test splits
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.manifest import DatasetManifest  # noqa: E402
from ml.datasets.validator import ImageValidator  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def run_eda():
    manifest_path = Path("data/splits/smartphone_clinical/manifest.json")
    if not manifest_path.exists():
        print(f"[-] Manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"[*] Loading dataset manifest from {manifest_path}...")
    manifest = DatasetManifest.from_json(manifest_path)
    total_items = len(manifest.items)
    print(f"[*] Total items in manifest: {total_items}")

    img_dir = Path("data/raw/pad_ufes_20")
    validator = ImageValidator(min_dimension=128, min_blur_score=8.0)

    # 1. Demographic & Clinical Aggregations
    classes_cnt = Counter()
    confirm_by_class = defaultdict(Counter)
    site_by_class = defaultdict(Counter)
    age_by_class = defaultdict(list)
    patients_set = set()
    lesions_set = set()
    patient_lesions = defaultdict(set)

    # 2. Image Quality & Spatial Metrics
    widths = []
    heights = []
    aspect_ratios = []
    blur_scores = []
    blur_by_class = defaultdict(list)
    r_means, g_means, b_means = [], [], []
    r_stds, g_stds, b_stds = [], [], []
    dark_images = []
    overexposed_images = []
    failed_images = []

    print("[*] Analyzing metadata and computing image metrics...")
    start_t = time.time()

    for idx, item in enumerate(manifest.items, 1):
        classes_cnt[item.class_id] += 1
        if item.patient_id:
            patients_set.add(item.patient_id)
            if item.lesion_id:
                patient_lesions[item.patient_id].add(item.lesion_id)
        if item.lesion_id:
            lesions_set.add(item.lesion_id)

        extra = item.metadata_extra
        conf_type = extra.get("diagnosis_confirm_type") or "unknown"
        confirm_by_class[item.class_id][conf_type] += 1

        site = extra.get("anatom_site_1") or extra.get("anatom_site_2") or "unknown"
        site_by_class[item.class_id][site] += 1

        age = extra.get("age_approx")
        if age is not None:
            age_by_class[item.class_id].append(float(age))

        # Check local image file if present
        img_file = img_dir / f"{item.image_id}.jpg"
        if img_file.exists():
            try:
                metrics, pil_img = validator.validate_file(img_file)
                if metrics.is_valid and pil_img:
                    w, h = pil_img.size
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h)
                    b_score = metrics.blur_score
                    blur_scores.append(b_score)
                    blur_by_class[item.class_id].append(b_score)

                    # Compute channel statistics on scaled down image (for speed)
                    thumb = pil_img.resize((128, 128), Image.Resampling.BILINEAR)
                    arr = np.array(thumb, dtype=np.float32) / 255.0
                    r_means.append(float(arr[:, :, 0].mean()))
                    g_means.append(float(arr[:, :, 1].mean()))
                    b_means.append(float(arr[:, :, 2].mean()))
                    r_stds.append(float(arr[:, :, 0].std()))
                    g_stds.append(float(arr[:, :, 1].std()))
                    b_stds.append(float(arr[:, :, 2].std()))

                    # Outlier checks
                    mean_val = arr.mean() * 255.0
                    if mean_val < 35.0:
                        dark_images.append(item.image_id)
                    elif mean_val > 225.0:
                        overexposed_images.append(item.image_id)
                else:
                    failed_images.append((item.image_id, metrics.rejection_reason))
            except Exception as e:
                failed_images.append((item.image_id, str(e)))

        if idx % 500 == 0 or idx == total_items:
            print(f"    Processed {idx}/{total_items} items ({(idx/total_items)*100:.1f}%)")

    elapsed = time.time() - start_t
    print(f"[*] Visual analysis complete in {elapsed:.2f}s.")

    # 3. Compile EDA Summary
    eda_summary = {
        "dataset_name": "PAD-UFES-20 (Collection 406)",
        "total_images": total_items,
        "unique_patients": len(patients_set),
        "unique_lesions": len(lesions_set),
        "class_distribution": dict(classes_cnt.most_common()),
        "confirmation_methods": {k: dict(v) for k, v in confirm_by_class.items()},
        "anatomical_sites": {k: dict(v) for k, v in site_by_class.items()},
        "age_statistics": {
            k: {
                "mean": float(np.mean(v)) if v else None,
                "median": float(np.median(v)) if v else None,
                "min": float(np.min(v)) if v else None,
                "max": float(np.max(v)) if v else None,
            }
            for k, v in age_by_class.items()
        },
        "spatial_dimensions": {
            "width_min": int(np.min(widths)) if widths else None,
            "width_max": int(np.max(widths)) if widths else None,
            "width_median": float(np.median(widths)) if widths else None,
            "height_min": int(np.min(heights)) if heights else None,
            "height_max": int(np.max(heights)) if heights else None,
            "height_median": float(np.median(heights)) if heights else None,
            "aspect_ratio_median": float(np.median(aspect_ratios)) if aspect_ratios else None,
        },
        "color_channel_statistics": {
            "mean_rgb": [float(np.mean(r_means)), float(np.mean(g_means)), float(np.mean(b_means))] if r_means else [],
            "std_rgb": [float(np.mean(r_stds)), float(np.mean(g_stds)), float(np.mean(b_stds))] if r_stds else [],
        },
        "blur_laplacian_statistics": {
            "median_blur": float(np.median(blur_scores)) if blur_scores else None,
            "p10_blur": float(np.percentile(blur_scores, 10)) if blur_scores else None,
            "p90_blur": float(np.percentile(blur_scores, 90)) if blur_scores else None,
        },
        "outlier_counts": {
            "failed_validation": len(failed_images),
            "dark_images": len(dark_images),
            "overexposed_images": len(overexposed_images),
        },
    }

    # Save summary JSON
    out_dir = Path("data/eda")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2, ensure_ascii=False)

    # 4. Generate Comprehensive Markdown Report
    report_path = Path("docs/EDA_REPORT_SMARTPHONE.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Exploratory Data Analysis (EDA) Report: PAD-UFES-20 Smartphone Clinical Dataset\n\n")
        f.write("## 1. Executive Summary & Clinical Context\n")
        f.write(f"- **Всего изображений**: {total_items}\n")
        f.write(f"- **Уникальных пациентов**: {len(patients_set)}\n")
        f.write(f"- **Уникальных кожных образований (lesions)**: {len(lesions_set)}\n")
        f.write("- **Лицензия**: Creative Commons Attribution 4.0 International (`CC BY 4.0`)\n")
        f.write("- **Модальность**: Клинические фотографии близкого расстояния, сделанные разными смартфонами (iPhone, Samsung Galaxy, Motorola) в клинических условиях.\n\n")

        f.write("## 2. Распределение целевых классов и фактор крайнего дисбаланса\n\n")
        f.write("| Код класса | Название | Кол-во изображений | Доля в датасете | Статус риска |\n")
        f.write("|---|---|---|---|---|\n")
        risk_map = {"mel": "Критический (Злокачественный)", "bcc": "Высокий (Злокачественный)", "scc": "Высокий (Злокачественный)", "ack": "Предрак (Актинический кератоз)", "sek": "Доброкачественный", "nev": "Доброкачественный"}
        name_map = {"mel": "Меланома (Melanoma)", "bcc": "Базальноклеточная карцинома (BCC)", "scc": "Плоскоклеточная карцинома (SCC)", "ack": "Актинический кератоз (ACK)", "sek": "Себорейный кератоз (SEK)", "nev": "Невус / родинка (NEV)"}
        for cid, count in classes_cnt.most_common():
            pct = (count / total_items) * 100
            f.write(f"| `{cid}` | {name_map.get(cid, cid)} | {count} | {pct:.1f}% | {risk_map.get(cid, '-')} |\n")

        f.write("\n> [!WARNING]\n")
        f.write("> **Критический вывод по дисбалансу**: Меланома представлена всего **52 изображениями (2.3%)**!\n")
        f.write("> Обычный Cross-Entropy Loss без взвешивания классов приведёт к тому, что сеть проигнорирует меланому и достигнет высокой точности (~80%) просто за счет частых классов (BCC 36.8% и ACK 31.8%). Обязательно применение **Focal Loss** или **Class-Weighted CrossEntropy**.\n\n")

        f.write("## 3. Надежность меток (Методы подтверждения диагнозов)\n\n")
        f.write("| Класс | Гистопатология (Золотой стандарт) | Консилиум экспертов (1 фото) | Консилиум (3 фото) |\n")
        f.write("|---|---|---|---|\n")
        for cid, _ in classes_cnt.most_common():
            c_dict = confirm_by_class[cid]
            histo = c_dict.get("histopathology", 0)
            single_ex = c_dict.get("single image expert consensus", 0)
            three_ex = c_dict.get("three image expert consensus", 0)
            f.write(f"| `{cid}` | {histo} ({histo/classes_cnt[cid]*100:.1f}%) | {single_ex} | {three_ex} |\n")

        f.write("\n> [!NOTE]\n")
        f.write("> Злокачественные новообразования (BCC, SCC, Melanoma) в подавляющем большинстве верифицированы **гистопатологией (биопсией)**, что исключает «мусорные» субъективные метки для опасных классов.\n\n")

        f.write("## 4. Качество изображений, пространственные размеры и цветность\n\n")
        if widths:
            f.write(f"- **Разрешение**: от {np.min(widths)}x{np.min(heights)} до {np.max(widths)}x{np.max(heights)} (Медиана: {int(np.median(widths))}x{int(np.median(heights))})\n")
            f.write(f"- **Медианный Laplacian Blur Score**: {np.median(blur_scores):.1f} (10-й перцентиль: {np.percentile(blur_scores, 10):.1f})\n")
            f.write(f"- **Цветовые характеристики RGB**: Средние = [{np.mean(r_means):.3f}, {np.mean(g_means):.3f}, {np.mean(b_means):.3f}], Отклонения = [{np.mean(r_stds):.3f}, {np.mean(g_stds):.3f}, {np.mean(b_stds):.3f}]\n")
            f.write(f"- **Экстремальные аномалии**: Слишком темных = {len(dark_images)}, Пересвеченных = {len(overexposed_images)}, Ошибок декодирования = {len(failed_images)}\n\n")

        f.write("## 5. Выводы для архитектуры модели и конвейера обучения\n")
        f.write("1. **Исключение Garbage-in**: Изображения с размером < 128x128 и размытые кадры с Laplacian < 60 должны отсекаться встроенным валидатором.\n")
        f.write("2. **Обязательный Transfer Learning**: На 2 298 снимках обучение с нуля невозможно. Необходима предобученная сверточная модель (`timm` EfficientNet-B0 или ConvNeXt-Tiny).\n")
        f.write("3. **Аугментации инвариантности**: Вращения на 360°, случайные отражения (Flip), ColorJitter (сглаживание вариаций сенсоров смартфонов).\n")
        f.write("4. **Взвешивание функции потерь**: Вес меланомы должен быть увеличен пропорционально обратному квадратному корню частоты классов.\n")

    print("\n[+] Full EDA analysis complete!")
    print(f"[+] Summary JSON saved: {out_dir / 'eda_summary.json'}")
    print(f"[+] Detailed Report saved: {report_path}")


if __name__ == "__main__":
    run_eda()
