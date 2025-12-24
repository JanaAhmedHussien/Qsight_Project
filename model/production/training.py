"""Training utilities derived from the QML_Qiskit notebook."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ClassicalDRModel

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    num_epochs: int = 25
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path | None = None


class EarlyStopping:
    def __init__(self, patience: int = 7, delta: float = 0.0, verbose: bool = False) -> None:
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss: float) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                LOGGER.info("EarlyStopping counter: %s/%s", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total if total else 0.0


class Phase1Trainer:
    def __init__(self, model: ClassicalDRModel, config: TrainingConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc_a": [],
            "val_acc_a": [],
            "train_acc_b": [],
            "val_acc_b": [],
            "train_acc_c": [],
            "val_acc_c": [],
            "train_acc_ensemble": [],
            "val_acc_ensemble": [],
            "ensemble_weights": [],
        }

    def _train_one_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int) -> Tuple[float, float, float, float, float]:
        self.model.train()
        total_loss = 0.0
        acc_a = acc_b = acc_c = acc_ensemble = 0.0
        num_batches = 0
        epoch_weights = []
        for images, targets, _ in tqdm(loader, desc=f"Epoch {epoch + 1} [Train]"):
            images, targets = images.to(self.config.device), targets.to(self.config.device)
            outputs = self.model(images, return_all=True)
            losses = self.model.compute_losses(outputs, targets)
            optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            acc_a += compute_accuracy(outputs["output_a"], targets)
            acc_b += compute_accuracy(outputs["output_b"], targets)
            acc_c += compute_accuracy(outputs["output_c"], targets)
            acc_ensemble += compute_accuracy(outputs["final_output"], targets)
            total_loss += losses["total_loss"].item()
            epoch_weights.append(outputs["ensemble_weights"].detach().cpu().numpy())
            num_batches += 1
        if epoch_weights:
            self.history["ensemble_weights"].append(np.mean(epoch_weights, axis=0))
        return (
            total_loss / num_batches,
            acc_a / num_batches,
            acc_b / num_batches,
            acc_c / num_batches,
            acc_ensemble / num_batches,
        )

    def _validate(self, loader: DataLoader, epoch: int) -> Tuple[float, float, float, float, float, list[int], list[int]]:
        self.model.eval()
        total_loss = 0.0
        acc_a = acc_b = acc_c = acc_ensemble = 0.0
        num_batches = 0
        preds: list[int] = []
        targets_all: list[int] = []
        with torch.no_grad():
            for images, targets, _ in tqdm(loader, desc=f"Epoch {epoch + 1} [Val]"):
                images, targets = images.to(self.config.device), targets.to(self.config.device)
                outputs = self.model(images, return_all=True)
                losses = self.model.compute_losses(outputs, targets)
                acc_a += compute_accuracy(outputs["output_a"], targets)
                acc_b += compute_accuracy(outputs["output_b"], targets)
                acc_c += compute_accuracy(outputs["output_c"], targets)
                acc_ensemble += compute_accuracy(outputs["final_output"], targets)
                total_loss += losses["total_loss"].item()
                _, batch_preds = torch.max(outputs["final_output"], 1)
                preds.extend(batch_preds.cpu().numpy())
                targets_all.extend(targets.cpu().numpy())
                num_batches += 1
        return (
            total_loss / num_batches,
            acc_a / num_batches,
            acc_b / num_batches,
            acc_c / num_batches,
            acc_ensemble / num_batches,
            preds,
            targets_all,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, list]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        for epoch in range(self.config.num_epochs):
            train_metrics = self._train_one_epoch(train_loader, optimizer, epoch)
            val_metrics = self._validate(val_loader, epoch)
            scheduler.step(val_metrics[0])
            early_stopping(val_metrics[0])
            if early_stopping.early_stop:
                LOGGER.info("Early stopping triggered at epoch %s", epoch + 1)
                break
            keys = [
                "train_loss",
                "train_acc_a",
                "train_acc_b",
                "train_acc_c",
                "train_acc_ensemble",
                "val_loss",
                "val_acc_a",
                "val_acc_b",
                "val_acc_c",
                "val_acc_ensemble",
            ]
            values = [*train_metrics, *val_metrics[:5]]
            for key, value in zip(keys, values, strict=True):
                self.history.setdefault(key, []).append(value)
            LOGGER.info(
                "Epoch %d: train_loss %.4f val_loss %.4f val_acc %.2f%%",
                epoch + 1,
                train_metrics[0],
                val_metrics[0],
                val_metrics[4] * 100.0,
            )
            if self.config.checkpoint_dir:
                self._save_checkpoint(epoch)
        report = classification_report(
            val_metrics[-1], val_metrics[-2], output_dict=False, zero_division=0
        )
        LOGGER.info("\n%s", report)
        return self.history

    def _save_checkpoint(self, epoch: int) -> None:
        if not self.config.checkpoint_dir:
            return
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.config.checkpoint_dir / f"phase1_epoch{epoch + 1}.pth"
        torch.save({"model_state_dict": self.model.state_dict(), "history": self.history}, checkpoint_path)
        LOGGER.debug("Saved checkpoint to %s", checkpoint_path)


def save_model_and_features(
    model: ClassicalDRModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    classes: Iterable[str],
    device: str,
    history: Dict[str, list] | None = None,
    output_dir: Path | str = "trained_model",
) -> Dict[str, np.ndarray]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    train_features: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    val_features: list[np.ndarray] = []
    val_labels: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels, _ in tqdm(train_loader, desc="Extract train features"):
            images = images.to(device)
            features = model.extract_compressed_features(images).cpu().numpy()
            train_features.append(features)
            train_labels.append(labels.numpy())
        for images, labels, _ in tqdm(val_loader, desc="Extract val features"):
            images = images.to(device)
            features = model.extract_compressed_features(images).cpu().numpy()
            val_features.append(features)
            val_labels.append(labels.numpy())
    train_features_np = np.concatenate(train_features, axis=0)
    train_labels_np = np.concatenate(train_labels, axis=0)
    val_features_np = np.concatenate(val_features, axis=0)
    val_labels_np = np.concatenate(val_labels, axis=0)
    model_info = {
        "compressed_dim": model.compressed_dim,
        "num_classes": model.num_classes,
        "classes": list(classes),
        "encoder_type": getattr(model, "encoder_type", "vit"),
        "pretrained": getattr(model, "pretrained", True),
    }
    torch.save(model.state_dict(), output_dir / "phase1_classical_model.pth")
    (output_dir / "metadata.json").write_text(json.dumps(model_info, indent=2))
    if history:
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    np.savez_compressed(
        output_dir / "quantum_training_data.npz",
        train_features=train_features_np,
        train_labels=train_labels_np,
        val_features=val_features_np,
        val_labels=val_labels_np,
    )
    return {
        "train_features": train_features_np,
        "train_labels": train_labels_np,
        "val_features": val_features_np,
        "val_labels": val_labels_np,
    }


def train_classical_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    classes: Iterable[str],
    config: TrainingConfig | None = None,
    output_dir: Path | str = "trained_model",
    encoder_type: str = "vit",
    pretrained: bool = True,
) -> tuple[ClassicalDRModel, Dict[str, list]]:
    config = config or TrainingConfig()
    classes = list(classes)
    model = ClassicalDRModel(num_classes=len(classes), encoder_type=encoder_type, pretrained=pretrained)
    trainer = Phase1Trainer(model, config)
    history = trainer.fit(train_loader, val_loader)
    save_model_and_features(
        trainer.model,
        train_loader,
        val_loader,
        classes,
        config.device,
        history=history,
        output_dir=output_dir,
    )
    return trainer.model, history
