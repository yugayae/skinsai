#!/usr/bin/env python3
"""Train, calibrate, evaluate and export the Smartphone Clinical profile model on NVIDIA RTX 3060."""

from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.export.exporter import ModelExporter  # noqa: E402
from ml.models.classifier import SkinClassifier  # noqa: E402
from ml.models.dataset import SkinLesionDataset, get_canonical_transforms  # noqa: E402
from ml.models.losses import ClassWeightedFocalLoss, compute_class_weights  # noqa: E402
from ml.training.trainer import SkinTrainer  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

CLASS_ORDER = ["bcc", "scc", "ack", "sek", "mel", "nev"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


def main():
    print("=" * 65)
    print(" SkinAI Training: Smartphone Clinical Profile (PAD-UFES-20)")
    print("=" * 65)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device : {device}")
    if device.type == "cuda":
        print(f"[*] GPU Name       : {torch.cuda.get_device_name(0)}")
        print(f"[*] VRAM Available : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Load splits
    split_dir = Path("data/splits/smartphone_clinical")
    train_df = pd.read_csv(split_dir / "train.csv", keep_default_na=False)
    val_df = pd.read_csv(split_dir / "val.csv", keep_default_na=False)
    test_df = pd.read_csv(split_dir / "test.csv", keep_default_na=False)

    print(f"[*] Dataset Splits : Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Compute class weights from training distribution
    train_counts = Counter(train_df["class_id"])
    weights = compute_class_weights(train_counts, CLASS_ORDER, power=0.5)
    print(f"[*] Class Weights  : {dict(zip(CLASS_ORDER, [round(float(w), 3) for w in weights]))}")

    # Datasets and Loaders
    img_dir = Path("data/raw/pad_ufes_20")
    train_dataset = SkinLesionDataset(
        manifest_items=train_df.to_dict("records"),
        class_to_idx=CLASS_TO_IDX,
        transform=get_canonical_transforms(is_train=True),
        image_dir=img_dir,
    )
    val_dataset = SkinLesionDataset(
        manifest_items=val_df.to_dict("records"),
        class_to_idx=CLASS_TO_IDX,
        transform=get_canonical_transforms(is_train=False),
        image_dir=img_dir,
    )
    test_dataset = SkinLesionDataset(
        manifest_items=test_df.to_dict("records"),
        class_to_idx=CLASS_TO_IDX,
        transform=get_canonical_transforms(is_train=False),
        image_dir=img_dir,
    )

    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    # Initialize model
    print("[*] Initializing EfficientNet-B0 with ImageNet pretrained weights...")
    model = SkinClassifier(
        backbone_name="efficientnet_b0",
        num_classes=len(CLASS_ORDER),
        pretrained=True,
        drop_rate=0.3,
        drop_path_rate=0.1,
    )

    # Loss, Optimizer, Scheduler
    criterion = ClassWeightedFocalLoss(
        alpha=weights,
        gamma=2.0,
        label_smoothing=0.05,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Trainer
    save_dir = Path("artifacts/models/smartphone_clinical")
    trainer = SkinTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=True,
        class_names=CLASS_ORDER,
    )

    # Fit model
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_dir=save_dir,
        checkpoint_name="best_model.pt",
    )

    # Load best checkpoint for final evaluation on Test Set
    best_ckpt_path = save_dir / "best_model.pt"
    print(f"\n[*] Loading best checkpoint from {best_ckpt_path}...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    calibrated_temp = float(ckpt.get("temperature", 1.0))
    print(f"[*] Calibrated Temperature T = {calibrated_temp:.3f}")

    # Evaluate on un-touched Test Set
    print("\n" + "=" * 65)
    print(" FINAL EVALUATION ON UN-TOUCHED TEST SET")
    print("=" * 65)
    test_loss, test_acc, test_f1, test_report, _, _ = trainer.evaluate(test_loader)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"Test Macro F1 : {test_f1:.4f}")
    print("\nPer-class Metrics on Test Set:")
    for c in CLASS_ORDER:
        c_stats = test_report.get(c, {})
        print(
            f"  - {c.upper():<4}: Precision={c_stats.get('precision', 0):.3f} | "
            f"Recall={c_stats.get('recall', 0):.3f} | "
            f"F1={c_stats.get('f1-score', 0):.3f} (support={int(c_stats.get('support', 0))})"
        )

    # Export to ONNX bundle
    print("\n[*] Exporting ONNX bundle and verifying numerical parity...")
    exporter = ModelExporter(model=model, class_names=CLASS_ORDER)
    metrics_summary = {
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "val_best_f1": float(ckpt.get("val_f1", 0.0)),
        "melanoma_recall": float(test_report.get("mel", {}).get("recall", 0.0)),
        "test_report": test_report,
    }
    manifest = exporter.create_artifact_bundle(
        bundle_dir=save_dir,
        profile_name="smartphone_clinical",
        metrics=metrics_summary,
        temperature=calibrated_temp,
        tau_abstain=0.40,
        tau_referral_melanoma=0.20,
    )

    print(f"[+] ONNX Export Verified! Parity Max Diff: {manifest['onnx_parity_max_diff']:.8f}")
    print(f"[+] Model SHA-256: {manifest['sha256']}")
    print(f"[+] Artifact Bundle ready at: {save_dir}")

    # Save training history
    with open(save_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Generate Model Card
    card_path = Path("docs/MODEL_CARD_SMARTPHONE_CLINICAL.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("# Model Card: SkinAI Smartphone Clinical Profile\n\n")
        f.write("## 1. Model Details\n")
        f.write("- **Model Name**: SkinAI EfficientNet-B0 (Smartphone Clinical Profile)\n")
        f.write("- **Backbone**: `efficientnet_b0` (timm ImageNet pre-trained)\n")
        f.write("- **Task**: 6-class skin lesion image classification from smartphone photos\n")
        f.write("- **Input Resolution**: 224x224x3\n")
        f.write(f"- **Calibrated Temperature**: {calibrated_temp:.3f}\n")
        f.write(f"- **ONNX Model SHA-256**: `{manifest['sha256']}`\n")
        f.write("- **Training Hardware**: NVIDIA GeForce RTX 3060 12GB (PyTorch AMP)\n\n")

        f.write("## 2. Intended Use & Safety Scope\n")
        f.write("- **Target Profile**: Consumer smartphone close-up skin lesion photos (PAD-UFES-20 baseline)\n")
        f.write("- **License Scope**: `commercial_ok` (Derived 100% from CC BY 4.0 PAD-UFES-20 data)\n")
        f.write("- **Clinical Disclaimer**: Experimental research model. NOT a diagnostic medical device.\n")
        f.write("- **Abstention Policy**: Model rejects input if max calibrated probability < 0.40 or image fails quality filters.\n\n")

        f.write("## 3. Evaluation Metrics (Un-touched Test Set, 346 images, 205 patients)\n\n")
        f.write(f"- **Accuracy**: {test_acc*100:.2f}%\n")
        f.write(f"- **Macro F1-Score**: {test_f1:.4f}\n")
        f.write(f"- **Melanoma Sensitivity (Recall)**: {test_report.get('mel', {}).get('recall', 0)*100:.1f}%\n\n")

        f.write("### Detailed Per-Class Performance\n\n")
        f.write("| Класс | Precision | Recall | F1-Score | Тестовая выборка (Support) |\n")
        f.write("|---|---|---|---|---|\n")
        for c in CLASS_ORDER:
            c_s = test_report.get(c, {})
            f.write(
                f"| `{c}` | {c_s.get('precision', 0):.3f} | {c_s.get('recall', 0):.3f} | "
                f"{c_s.get('f1-score', 0):.3f} | {int(c_s.get('support', 0))} |\n"
            )

    print(f"[+] Model Card generated at: {card_path}")


if __name__ == "__main__":
    main()
