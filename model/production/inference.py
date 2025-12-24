"""Inference helpers for running the trained classical model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from torchvision import transforms

from .models import ClassicalDRModel

LOGGER = logging.getLogger(__name__)


DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model(checkpoint_dir: Path | str) -> Tuple[ClassicalDRModel, Dict[str, str]]:
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / "phase1_classical_model.pth"
    metadata_path = checkpoint_dir / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    num_classes = int(metadata.get("num_classes", 5))
    encoder_type = metadata.get("encoder_type", "vit")
    pretrained = bool(metadata.get("pretrained", True))
    model = ClassicalDRModel(num_classes=num_classes, encoder_type=encoder_type, pretrained=pretrained)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    class_list = metadata.get("classes")
    if class_list:
        model.classes = class_list
    return model, metadata


def predict_image(
    model: ClassicalDRModel,
    image_path: Path | str,
    device: str = "cpu",
    transform=DEFAULT_TRANSFORM,
    class_names: Dict[int, str] | None = None,
) -> Dict[str, float]:
    model = model.to(device)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor, return_all=False)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    if class_names is None:
        class_labels = getattr(model, "classes", None)
        if class_labels is None:
            class_names = {idx: str(idx) for idx in range(len(probabilities))}
        else:
            class_names = {idx: name for idx, name in enumerate(class_labels)}
    return {class_names[idx]: float(prob) for idx, prob in enumerate(probabilities)}
