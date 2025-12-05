from typing import Dict

import torch
import torch.nn as nn
from torchvision import models


def _init_kaiming(module: nn.Module) -> None:
    """Inicialização Kaiming para camadas Conv2d e Linear."""

    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _print_trainable_params(model: nn.Module, name: str) -> None:
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] parâmetros treináveis: {n_params}")


class SimpleECGCNN(nn.Module):
    """CNN 2D simples para classificação de ECG em imagem RGB.

    Mantida para compatibilidade; preferir `SimpleCNN`.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        _init_kaiming(self)
        _print_trainable_params(self, "SimpleECGCNN")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleCNN(nn.Module):
    """CNN 2D genérica com 3 blocos Conv-BN-ReLU-Pool e classifier linear.

    - Inicialização Kaiming para convs/linears
    - Dropout 0.3 no classifier
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

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        _init_kaiming(self)
        _print_trainable_params(self, "SimpleCNN")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.classifier(self.features(x))


def make_resnet18(
    num_classes: int,
    pretrained: bool = True,
    unfreeze_last_n: int = 0,
) -> nn.Module:
    """Cria um modelo resnet18 para classificação com controle de congelamento.

    - Se `pretrained=True`, usa pesos pré-treinados em ImageNet.
    - Substitui a última camada fully-connected para `num_classes` com dropout 0.3.
    - Congela todas as camadas por padrão e reativa treino das últimas `unfreeze_last_n`.
      A ordem considerada é: [layer1, layer2, layer3, layer4, fc].
    """

    # A API de pesos mudou em versões mais novas; tentamos usar a interface moderna
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except AttributeError:
        # Fallback para versões mais antigas
        model = models.resnet18(pretrained=pretrained)

    in_features = model.fc.in_features  # type: ignore[assignment]
    model.fc = nn.Sequential(  # type: ignore[assignment]
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    _init_kaiming(model.fc)

    # Congela tudo
    for p in model.parameters():
        p.requires_grad = False

    # Define ordem de camadas "superiores" para eventual unfreeze
    layers_in_order = [model.layer1, model.layer2, model.layer3, model.layer4, model.fc]
    unfreeze_last_n = max(0, min(unfreeze_last_n, len(layers_in_order)))

    for layer in layers_in_order[-unfreeze_last_n:]:
        for p in layer.parameters():
            p.requires_grad = True

    _print_trainable_params(model, "ResNet18")
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
    unfreeze_last_n = int(model_cfg.get("unfreeze_last_n", 0))

    if num_classes <= 0:
        raise ValueError("model.num_classes deve ser > 0")

    if name in {"simple", "simple_cnn", "cnn"}:
        return SimpleCNN(num_classes=num_classes)

    if name in {"simple_ecg", "ecg_cnn"}:
        return SimpleECGCNN(num_classes=num_classes)

    if name in {"resnet18", "resnet_18"}:
        return make_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
            unfreeze_last_n=unfreeze_last_n,
        )

    raise ValueError(f"Modelo desconhecido em config.model.name: {name}")

