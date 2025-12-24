"""Model components extracted from the QML_Qiskit notebook."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VisionEncoder(nn.Module):
    """Backbone encoder supporting ViT or ResNet features."""

    def __init__(self, encoder_type: str = "vit", pretrained: bool = True) -> None:
        super().__init__()
        self.encoder_type = encoder_type.lower()
        if self.encoder_type == "vit":
            vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
            vit.heads = nn.Identity()
            self.encoder = vit
            self.projection = nn.Linear(768, 2048)
        else:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.encoder = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if self.encoder_type == "vit":
            features = self.projection(features)
        return features


class CompressionModule(nn.Module):
    def __init__(self, input_dim: int = 2048, compressed_dim: int = 30) -> None:
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, compressed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compressor(x)


class ClassicalHeadA(nn.Module):
    def __init__(self, input_dim: int = 2048, num_classes: int = 5) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ClassicalHeadB(nn.Module):
    def __init__(self, input_dim: int = 30, num_classes: int = 5) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class QuantumSimulatedHead(nn.Module):
    """Placeholder for quantum head; implemented as trainable NN layer."""

    def __init__(self, input_dim: int = 30, num_classes: int = 5, quantum_layers: int = 3) -> None:
        super().__init__()
        layers = []
        current_dim = input_dim
        for layer_idx in range(quantum_layers):
            next_dim = 64 if layer_idx < quantum_layers - 1 else 32
            layers.extend(
                [
                    nn.Linear(current_dim, next_dim),
                    nn.BatchNorm1d(next_dim),
                    nn.Tanh() if layer_idx < quantum_layers - 1 else nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            current_dim = next_dim
        layers.append(nn.Linear(32, num_classes))
        self.quantum_sim = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantum_sim(x)


class DynamicEnsemble(nn.Module):
    def __init__(self, num_heads: int = 3, init_temp: float = 1.0) -> None:
        super().__init__()
        self.base_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        self.temperature = nn.Parameter(torch.tensor(init_temp))
        self.uncertainty_scales = nn.Parameter(torch.ones(num_heads))

    def forward(self, head_outputs: list[torch.Tensor], uncertainties: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.base_weights / self.temperature, dim=0)
        if uncertainties is not None and self.training:
            if uncertainties.dim() == 1:
                uncertainties = uncertainties.unsqueeze(0)
            scaled = uncertainties * self.uncertainty_scales.unsqueeze(0)
            confidence = 1.0 / (scaled + 1e-8)
            batch_confidence = confidence.mean(dim=0)
            confidence_weights = F.softmax(batch_confidence, dim=0)
            with torch.no_grad():
                predictions = torch.stack([torch.argmax(out, dim=1) for out in head_outputs], dim=1)
                agreement = (predictions.max(dim=1).values == predictions.min(dim=1).values).float().mean()
            uncertainty_weight = 0.7 * (1 - agreement) + 0.3
            weights = (1 - uncertainty_weight) * weights + uncertainty_weight * confidence_weights
        weights = weights / weights.sum()
        if not self.training:
            weights = weights.detach()
        combined = sum(w * out for w, out in zip(weights, head_outputs))
        return combined, weights


class ClassicalDRModel(nn.Module):
    def __init__(
        self,
        encoder_type: str = "vit",
        num_classes: int = 5,
        compressed_dim: int = 30,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.pretrained = pretrained
        self.vision_encoder = VisionEncoder(encoder_type=encoder_type, pretrained=pretrained)
        self.compression = CompressionModule(input_dim=2048, compressed_dim=compressed_dim)
        self.classical_head_a = ClassicalHeadA(input_dim=2048, num_classes=num_classes)
        self.classical_head_b = ClassicalHeadB(input_dim=compressed_dim, num_classes=num_classes)
        self.quantum_head = QuantumSimulatedHead(input_dim=compressed_dim, num_classes=num_classes)
        self.ensemble = DynamicEnsemble(num_heads=3)
        self.num_classes = num_classes
        self.compressed_dim = compressed_dim

    def forward(self, x: torch.Tensor, return_all: bool = True) -> dict[str, torch.Tensor] | torch.Tensor:
        latent = self.vision_encoder(x)
        compressed = self.compression(latent)
        out_a = self.classical_head_a(latent)
        out_b = self.classical_head_b(compressed)
        out_c = self.quantum_head(compressed)
        with torch.no_grad():
            prob_a = F.softmax(out_a, dim=1)
            prob_b = F.softmax(out_b, dim=1)
            prob_c = F.softmax(out_c, dim=1)
            uncertainties = torch.tensor(
                [1.0 - prob_a.max(dim=1).values.mean(), 1.0 - prob_b.max(dim=1).values.mean(), 1.0 - prob_c.max(dim=1).values.mean()],
                device=x.device,
            )
        final_output, ensemble_weights = self.ensemble([out_a, out_b, out_c], uncertainties)
        if return_all:
            return {
                "latent_features": latent,
                "compressed_features": compressed,
                "output_a": out_a,
                "output_b": out_b,
                "output_c": out_c,
                "final_output": final_output,
                "ensemble_weights": ensemble_weights,
                "uncertainties": uncertainties,
            }
        return final_output

    def compute_losses(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor) -> dict[str, torch.Tensor]:
        loss_a = F.cross_entropy(outputs["output_a"], targets)
        loss_b = F.cross_entropy(outputs["output_b"], targets)
        loss_c = F.cross_entropy(outputs["output_c"], targets)
        ensemble_loss = F.cross_entropy(outputs["final_output"], targets)
        total_loss = loss_a + loss_b + loss_c + 0.5 * ensemble_loss
        return {
            "total_loss": total_loss,
            "loss_a": loss_a,
            "loss_b": loss_b,
            "loss_c": loss_c,
            "ensemble_loss": ensemble_loss,
        }

    def extract_compressed_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent = self.vision_encoder(x)
            compressed = self.compression(latent)
        return compressed
