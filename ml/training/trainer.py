"""Model training and evaluation loop with mixed precision and metrics tracking."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.models.classifier import SkinClassifier


class SkinTrainer:
    """Trainer executing GPU-accelerated training, evaluation, and calibration."""

    def __init__(
        self,
        model: SkinClassifier,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        class_names: Optional[List[str]] = None,
    ):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.class_names = class_names or [str(i) for i in range(model.num_classes)]

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Run one training epoch. Returns (avg_loss, accuracy, macro_f1)."""
        self.model.train()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        for images, targets, _ in dataloader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp and self.scaler:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        acc = float(accuracy_score(all_targets, all_preds))
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        return avg_loss, acc, macro_f1

    @torch.no_grad()
    def evaluate(
        self, dataloader: DataLoader
    ) -> Tuple[float, float, float, Dict[str, Dict[str, float]], torch.Tensor, torch.Tensor]:
        """Evaluate model on validation or test set."""
        self.model.eval()
        total_loss = 0.0
        all_logits: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        for images, targets, _ in dataloader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            total_loss += loss.item() * images.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)

        preds = torch.argmax(cat_logits, dim=-1).numpy()
        targets_np = cat_targets.numpy()

        avg_loss = total_loss / len(dataloader.dataset)
        acc = float(accuracy_score(targets_np, preds))
        macro_f1 = float(f1_score(targets_np, preds, average="macro", zero_division=0))
        report = classification_report(
            targets_np,
            preds,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        return avg_loss, acc, macro_f1, report, cat_logits, cat_targets

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        save_dir: Optional[Path] = None,
        checkpoint_name: str = "best_model.pt",
    ) -> Dict[str, List[float]]:
        """Fit model for N epochs, saving best checkpoint by validation Macro F1."""
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "mel_recall": [],
        }

        best_f1 = -1.0
        best_checkpoint_path = None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            best_checkpoint_path = save_dir / checkpoint_name

        print(f"[*] Training on {self.device} (AMP: {self.use_amp}) for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, tr_f1 = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1, report, val_logits, val_targets = self.evaluate(val_loader)

            if self.scheduler:
                self.scheduler.step()

            mel_rec = report.get("mel", {}).get("recall", 0.0)
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["train_f1"].append(tr_f1)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
            history["mel_recall"].append(mel_rec)

            elapsed = time.time() - t0
            saved_mark = ""
            if val_f1 > best_f1:
                best_f1 = val_f1
                if best_checkpoint_path:
                    # Calibrate temperature on this best validation set
                    temp = self.model.calibrate_temperature(val_logits, val_targets)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "val_f1": val_f1,
                            "val_acc": val_acc,
                            "mel_recall": mel_rec,
                            "temperature": temp,
                            "class_names": self.class_names,
                        },
                        best_checkpoint_path,
                    )
                    saved_mark = f" [Saved best F1: {val_f1:.4f}, Temp: {temp:.2f}]"

            print(
                f"Epoch {epoch:>2}/{epochs} ({elapsed:.1f}s) | "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} F1: {tr_f1:.3f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} (Mel Rec: {mel_rec:.2f}){saved_mark}"
            )

        print(f"[+] Training complete! Best Val Macro F1: {best_f1:.4f}")
        return history
