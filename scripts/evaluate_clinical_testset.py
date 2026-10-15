"""Scientific Clinical Validation and Calibration Evaluation Script.

Evaluates the smartphone_clinical ONNX model across the untouched test split (PAD-UFES-20),
computing multi-threshold melanoma sensitivity/specificity curves, ROC-AUC, PR-AUC,
Expected Calibration Error (ECE pre/post temperature scaling), and triage safety profiles.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.inference.predictor import OnnxPredictor  # noqa: E402


def calculate_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            abs_diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += abs_diff * prop_in_bin
            mce = max(mce, abs_diff)

            bin_details.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": int(np.sum(in_bin)),
                "accuracy": float(accuracy_in_bin),
                "confidence": float(avg_confidence_in_bin),
                "diff": float(abs_diff),
            })

    return float(ece), float(mce), bin_details


def main():
    bundle_dir = REPO_ROOT / "artifacts" / "models" / "smartphone_clinical"
    test_csv_path = REPO_ROOT / "data" / "splits" / "smartphone_clinical" / "test.csv"
    images_dir = REPO_ROOT / "data" / "raw" / "pad_ufes_20"

    print(f"Loading ONNX model from: {bundle_dir}")
    predictor = OnnxPredictor(bundle_dir)
    classes = predictor.classes
    class_to_idx = {c: i for i, c in enumerate(classes)}
    temperature = predictor.temperature

    print(f"Classes: {classes}")
    print(f"Calibrated Temperature: {temperature:.4f}")
    print(f"Reading test split: {test_csv_path}")

    rows = []
    with open(test_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"Found {len(rows)} samples in test.csv")

    y_true_indices = []
    y_true_classes = []
    raw_logits_list = []
    probs_calibrated_list = []
    probs_uncalibrated_list = []
    patient_ids = set()

    for row in rows:
        image_id = row["image_id"]
        true_class = row["class_id"]
        patient_ids.add(row["patient_id"])

        image_path = images_dir / f"{image_id}.jpg"
        if not image_path.exists():
            print(f"Warning: Image missing: {image_path}")
            continue

        with Image.open(image_path) as pil_img:
            # Ensure RGB format
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            # Preprocess tensor exactly as done in production
            tensor = predictor.preprocess_image(pil_img)

        # Run ONNX inference
        ort_inputs = {predictor.input_name: tensor}
        logits = predictor.session.run(None, ort_inputs)[0][0]  # shape: (6,)

        # Uncalibrated softmax (T=1.0)
        exp_logits_uncal = np.exp(logits - np.max(logits))
        probs_uncal = exp_logits_uncal / np.sum(exp_logits_uncal)

        # Calibrated softmax (T=temperature)
        scaled_logits = logits / temperature
        exp_logits_cal = np.exp(scaled_logits - np.max(scaled_logits))
        probs_cal = exp_logits_cal / np.sum(exp_logits_cal)

        y_true_indices.append(class_to_idx[true_class])
        y_true_classes.append(true_class)
        raw_logits_list.append(logits)
        probs_uncalibrated_list.append(probs_uncal)
        probs_calibrated_list.append(probs_cal)

    y_true = np.array(y_true_indices)
    probs_uncal = np.array(probs_uncalibrated_list)
    probs_cal = np.array(probs_calibrated_list)
    n_samples = len(y_true)

    print(f"\nSuccessfully evaluated {n_samples} images across {len(patient_ids)} unique patients.")

    # 1. Standard Argmax Metrics
    y_pred = np.argmax(probs_cal, axis=1)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    # 2. Calibration Analysis (ECE & MCE)
    ece_uncal, mce_uncal, bins_uncal = calculate_ece(probs_uncal, y_true, n_bins=10)
    ece_cal, mce_cal, bins_cal = calculate_ece(probs_cal, y_true, n_bins=10)

    # 3. Multi-Class ROC-AUC and PR-AUC (One-vs-Rest)
    # One-hot encode ground truth
    y_true_onehot = np.zeros((n_samples, len(classes)))
    for idx, true_idx in enumerate(y_true):
        y_true_onehot[idx, true_idx] = 1.0

    roc_auc_per_class = {}
    pr_auc_per_class = {}
    for i, c in enumerate(classes):
        roc_auc_per_class[c] = float(roc_auc_score(y_true_onehot[:, i], probs_cal[:, i]))
        pr_auc_per_class[c] = float(average_precision_score(y_true_onehot[:, i], probs_cal[:, i]))

    macro_roc_auc = float(np.mean(list(roc_auc_per_class.values())))
    macro_pr_auc = float(np.mean(list(pr_auc_per_class.values())))

    # 4. Melanoma Clinical Safety Threshold Scanning
    mel_idx = class_to_idx["mel"]
    mel_true_binary = (y_true == mel_idx).astype(int)
    mel_probs = probs_cal[:, mel_idx]
    mel_thresholds = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]
    mel_threshold_results = []

    for tau in mel_thresholds:
        mel_pred_binary = (mel_probs >= tau).astype(int)
        tp = int(np.sum((mel_pred_binary == 1) & (mel_true_binary == 1)))
        fp = int(np.sum((mel_pred_binary == 1) & (mel_true_binary == 0)))
        tn = int(np.sum((mel_pred_binary == 0) & (mel_true_binary == 0)))
        fn = int(np.sum((mel_pred_binary == 0) & (mel_true_binary == 1)))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        mel_threshold_results.append({
            "threshold": tau,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "sensitivity_recall": float(sensitivity),
            "specificity": float(specificity),
            "ppv_precision": float(ppv),
            "npv": float(npv),
            "referrals_triggered": int(tp + fp),
            "referral_rate": float((tp + fp) / n_samples),
        })

    # 5. Malignancy Group Risk Analysis (BCC + SCC + MEL vs ACK + SEK + NEV)
    malignant_indices = [class_to_idx["bcc"], class_to_idx["scc"], class_to_idx["mel"]]
    benign_indices = [class_to_idx["ack"], class_to_idx["sek"], class_to_idx["nev"]]

    is_malignant_true = np.isin(y_true, malignant_indices).astype(int)
    malignant_probs = np.sum(probs_cal[:, malignant_indices], axis=1)
    n_malignant = int(np.sum(is_malignant_true))
    n_benign = int(len(y_true) - n_malignant)

    mal_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]
    mal_threshold_results = []
    for tau in mal_thresholds:
        mal_pred = (malignant_probs >= tau).astype(int)
        tp = int(np.sum((mal_pred == 1) & (is_malignant_true == 1)))
        fn = int(np.sum((mal_pred == 0) & (is_malignant_true == 1)))
        fp = int(np.sum((mal_pred == 1) & (is_malignant_true == 0)))
        tn = int(np.sum((mal_pred == 0) & (is_malignant_true == 0)))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        mal_threshold_results.append({
            "threshold": tau,
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "miss_rate_fn": float(miss_rate),
            "fn_count": fn,
            "tp_count": tp,
        })

    # Critical False Negatives under Argmax (Malignant misclassified as benign)
    argmax_pred_is_benign = np.isin(y_pred, benign_indices)
    critical_fn_count = int(np.sum((is_malignant_true == 1) & argmax_pred_is_benign))
    critical_fn_rate = critical_fn_count / n_malignant

    # Compile structured summary
    results = {
        "dataset": "PAD-UFES-20 (Untouched Test Split)",
        "samples_evaluated": n_samples,
        "patients_count": len(patient_ids),
        "temperature": temperature,
        "overall_metrics": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "macro_roc_auc": float(macro_roc_auc),
            "macro_pr_auc": float(macro_pr_auc),
        },
        "calibration": {
            "ece_uncalibrated": float(ece_uncal),
            "mce_uncalibrated": float(mce_uncal),
            "ece_calibrated": float(ece_cal),
            "mce_calibrated": float(mce_cal),
            "ece_improvement": float(ece_uncal - ece_cal),
            "bins_calibrated": bins_cal,
        },
        "roc_auc_per_class": roc_auc_per_class,
        "pr_auc_per_class": pr_auc_per_class,
        "per_class_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "melanoma_threshold_curve": mel_threshold_results,
        "malignancy_group_curve": mal_threshold_results,
        "malignancy_summary": {
            "total_malignant_cases": n_malignant,
            "total_benign_cases": n_benign,
            "argmax_critical_missed_malignancy": critical_fn_count,
            "argmax_critical_fn_rate": float(critical_fn_rate),
        },
    }

    # Save to artifacts/benchmarks
    out_json = REPO_ROOT / "artifacts" / "benchmarks" / "clinical_validation_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved structured validation results to: {out_json}")
    print(f"Overall Accuracy: {acc*100:.2f}% | Balanced Acc: {bal_acc*100:.2f}% | Macro F1: {macro_f1:.4f}")
    print(f"Macro ROC-AUC: {macro_roc_auc:.4f} | Macro PR-AUC: {macro_pr_auc:.4f}")
    print(f"Calibration ECE: {ece_uncal:.4f} -> {ece_cal:.4f} (Improved by {(ece_uncal - ece_cal)/ece_uncal*100:.1f}%)")

    print("\nMelanoma Sensitivity across Referral Thresholds (Total Melanomas = 8):")
    for r in mel_threshold_results:
        print(
            f"  tau = {r['threshold']:.2f} | Recall: {r['sensitivity_recall']*100:5.1f}% ({r['tp']}/8) | "
            f"Specificity: {r['specificity']*100:5.1f}% | Referrals: {r['referrals_triggered']:3d} ({r['referral_rate']*100:4.1f}%)"
        )

    return results


if __name__ == "__main__":
    main()
