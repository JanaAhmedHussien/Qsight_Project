"""Command-line entry point for classical training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import DatasetConfig, create_dataloaders
from training import TrainingConfig, train_classical_model

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical DR model")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to dataset root directory")
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="tanlikesmath",
        choices=["tanlikesmath", "sovitrath"],
        help="Dataset variant",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("trained_model"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights")
    parser.add_argument("--encoder-type", type=str, default="vit", choices=["vit", "resnet"], help="Backbone encoder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    dataset_config = DatasetConfig(
        root=args.dataset_root,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
    )
    train_loader, val_loader, classes = create_dataloaders(dataset_config)
    training_config = TrainingConfig(num_epochs=args.epochs, learning_rate=args.learning_rate)
    train_classical_model(
        train_loader,
        val_loader,
        classes,
        config=training_config,
        output_dir=args.output_dir,
        encoder_type=args.encoder_type,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
