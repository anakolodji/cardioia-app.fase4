import argparse
import os
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from dataset import create_dataloaders_from_config
from models import create_model_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino de modelo para ECG / imagens médicas")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Caminho para o arquivo de configuração YAML (ex.: configs/ecg.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Diretório para salvar checkpoints e logs básicos",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Arquivo de config não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config YAML deve ser um mapeamento (dict). Arquivo: {path}")

    return cfg


def get_device(cfg: Dict[str, Any]) -> torch.device:
    train_cfg = cfg.get("train", {})
    device_str = train_cfg.get("device")
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        _, preds = torch.max(outputs, dim=1)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (preds == targets).sum().item()
        total += batch_size

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": acc}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device(config)

    # Data
    loaders = create_dataloaders_from_config(config)
    train_loader = loaders.get("train")
    if train_loader is None:
        raise RuntimeError("Nenhum DataLoader de treino encontrado (split 'train')")

    val_loader = loaders.get("val")

    # Modelo
    model = create_model_from_config(config)
    model.to(device)

    # Hiperparâmetros de treino
    train_cfg = config.get("train", {})
    num_epochs = int(train_cfg.get("num_epochs", 10))
    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_model_path = os.path.join(args.output_dir, "best.pt")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        msg = f"Epoch {epoch}/{num_epochs} - train_loss: {train_loss:.4f}"

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            msg += f" | val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"

            # Salva melhor modelo com base na acurácia de validação
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({"model_state_dict": model.state_dict(), "config": config}, best_model_path)
        else:
            # Se não houver validação, sempre sobrescreve
            torch.save({"model_state_dict": model.state_dict(), "config": config}, best_model_path)

        print(msg)

    print(f"Treino finalizado. Melhor modelo salvo em: {best_model_path}")


if __name__ == "__main__":
    main()

