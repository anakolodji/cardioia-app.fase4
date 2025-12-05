from typing import Dict

import torch
import torch.nn as nn
from torchvision import models


class SimpleECGCNN(nn.Module):
    """CNN 2D simples para classificação de ECG em imagem RGB.

    Estrutura aproximada:
    - 3 blocos conv + BN + ReLU + maxpool
    - 1 ou 2 camadas fully-connected no final
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112

            # Bloco 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56

            # Bloco 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28
        )

        # Para input 224x224, após 3 pools 2x2 -> 28x28
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Cria um modelo resnet18 para classificação.

    - Se `pretrained=True`, usa pesos pré-treinados em ImageNet.
    - Substitui a última camada fully-connected para `num_classes`.
    """

    # A API de pesos mudou em versões mais novas; tentamos usar a interface moderna
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except AttributeError:
        # Fallback para versões mais antigas
        model = models.resnet18(pretrained=pretrained)

    in_features = model.fc.in_features  # type: ignore[assignment]
    model.fc = nn.Linear(in_features, num_classes)  # type: ignore[assignment]
    return model


def create_model_from_config(config: Dict) -> nn.Module:
    """Cria um modelo a partir de um dicionário de configuração.

    Espera algo como:

        model:
          name: "simple_cnn"   # ou "resnet18"
          num_classes: 2
          pretrained: true      # apenas relevante para resnet18

    Parâmetros:
        config: dicionário completo carregado do YAML.
    """

    model_cfg = config.get("model", {})
    name = str(model_cfg.get("name", "simple_cnn")).lower()
    num_classes = int(model_cfg.get("num_classes", 2))
    pretrained = bool(model_cfg.get("pretrained", True))

    if num_classes <= 0:
        raise ValueError("model.num_classes deve ser > 0")

    if name in {"simple", "simple_cnn", "cnn"}:
        return SimpleECGCNN(num_classes=num_classes)

    if name in {"resnet18", "resnet_18"}:
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Modelo desconhecido em config.model.name: {name}")

