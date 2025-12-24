"""Dataset utilities for diabetic retinopathy classification."""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms

LOGGER = logging.getLogger(__name__)


def _default_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _default_val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@dataclass
class DatasetConfig:
    root: Path
    dataset_type: str = "tanlikesmath"
    image_size: int = 224
    train_ratio: float = 0.8
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 42
    transform: Optional[Callable] = None
    val_transform: Optional[Callable] = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        if self.transform is None:
            self.transform = _default_transform(self.image_size)
        if self.val_transform is None:
            self.val_transform = _default_val_transform(self.image_size)


class DiabeticRetinopathyDataset(Dataset):
    """Dataset supporting Sovitrath (preprocessed) and Tanlikesmath (resized) variants."""

    def __init__(
        self,
        config: DatasetConfig,
        mode: str = "train",
        transform: Optional[Callable] = None,
        image_paths: Optional[Sequence[Path]] = None,
        labels: Optional[Sequence[int]] = None,
        classes: Optional[Sequence[str]] = None,
        class_to_idx: Optional[dict[str, int]] = None,
    ) -> None:
        self.config = config
        self.mode = mode
        self.transform = transform or config.transform
        if image_paths is not None and labels is not None and classes is not None:
            self.image_paths = list(map(Path, image_paths))
            self.labels = list(labels)
            self.classes = list(classes)
            self.class_to_idx = class_to_idx or {cls: idx for idx, cls in enumerate(classes)}
        else:
            self.image_paths: list[Path] = []
            self.labels: list[int] = []
            self.classes: Sequence[str] = []
            self.class_to_idx: dict[str, int] = {}
            loader = self._loaders().get(config.dataset_type.lower())
            if loader is None:
                raise ValueError(f"Unsupported dataset type: {config.dataset_type}")
            loader()

        LOGGER.info("Loaded %d samples for %s (%s)", len(self.image_paths), config.dataset_type, mode)

    # Loader selection ---------------------------------------------------
    def _loaders(self) -> dict[str, Callable[[], None]]:
        return {
            "sovitrath": self._load_sovitrath,
            "tanlikesmath": self._load_tanlikesmath,
        }

    # Dataset loaders ----------------------------------------------------
    def _load_sovitrath(self) -> None:
        base_path = self.config.root / "gaussian_filtered_images" / "gaussian_filtered_images"
        classes = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
        self._load_from_directory(base_path, classes)

    def _load_tanlikesmath(self) -> None:
        base_path = self.config.root / "resized_train" / "resized_train"
        labels_path = self.config.root / "resized_train" / "trainLabels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels CSV: {labels_path}")
        labels_df = pd.read_csv(labels_path)
        self.classes = ["0", "1", "2", "3", "4"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        for _, row in labels_df.iterrows():
            img_name = f"{row['image']}.jpeg"
            img_path = base_path / img_name
            if img_path.exists():
                self.image_paths.append(img_path)
                self.labels.append(int(row["level"]))

    def _load_from_directory(self, base_path: Path, classes: Sequence[str]) -> None:
        self.classes = list(classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        if not base_path.exists():
            raise FileNotFoundError(f"Image directory not found: {base_path}")
        for class_name in self.classes:
            class_path = base_path / class_name
            if not class_path.exists():
                LOGGER.warning("Skipping missing class directory: %s", class_path)
                continue
            for image_name in class_path.glob("*.jp*g"):
                self.image_paths.append(image_name)
                self.labels.append(self.class_to_idx[class_name])

    # Dataset interface --------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, str]:
        img_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(img_path)


def create_dataloaders(config: DatasetConfig) -> Tuple[DataLoader, DataLoader, Sequence[str]]:
    base_dataset = DiabeticRetinopathyDataset(config, mode="full", transform=None)
    indices = list(range(len(base_dataset)))
    labels = [base_dataset.labels[i] for i in indices]
    train_indices, val_indices = train_test_split(
        indices,
        train_size=config.train_ratio,
        stratify=labels,
        random_state=config.seed,
    )

    def subset(indices_subset: Sequence[int], transform: Callable) -> DiabeticRetinopathyDataset:
        image_paths = [base_dataset.image_paths[i] for i in indices_subset]
        subset_labels = [base_dataset.labels[i] for i in indices_subset]
        return DiabeticRetinopathyDataset(
            config,
            mode="subset",
            transform=transform,
            image_paths=image_paths,
            labels=subset_labels,
            classes=base_dataset.classes,
            class_to_idx=base_dataset.class_to_idx,
        )

    train_dataset = subset(train_indices, config.transform)
    val_dataset = subset(val_indices, config.val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, base_dataset.classes


if __name__ == "__main__":
    config = DatasetConfig(root="~/data/diabetic_retinopathy", dataset_type="tanlikesmath")
    train_loader, val_loader, classes = create_dataloaders(config)
    LOGGER.info("Number of classes: %d", len(classes))
    LOGGER.info("Classes: %s", classes)
    LOGGER.info("Training samples: %d", len(train_loader.dataset))
    LOGGER.info("Validation samples: %d", len(val_loader.dataset))