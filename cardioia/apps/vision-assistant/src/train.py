import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import yaml

from dataset import build_dataloaders
from models import create_model_from_config
from utils import set_global_seed, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino de modelos (SimpleCNN / ResNet18) para ECG")
    parser.add_argument("--config", type=str, required=True, help="Caminho para o YAML (ex.: configs/ecg.yaml)")
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "resnet18"], help="Arquitetura do modelo")
    parser.add_argument("--epochs", type=int, default=15, help="Número máximo de épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do batch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate inicial")
    parser.add_argument("--unfreeze-last-n", type=int, default=2, help="Número de blocos finais a descongelar na ResNet18")
    parser.add_argument("--seed", type=int, default=42, help="Seed global para reprodutibilidade")
    parser.add_argument("--patience", type=int, default=5, help="Patience para early stopping baseado em val_f1")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Diretório para salvar checkpoints e métricas")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Arquivo de config não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config YAML deve ser um mapeamento (dict). Arquivo: {path}")

    return cfg


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
def evaluate_on_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    running_loss = 0.0
    total = 0

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_proba: List[np.ndarray] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_proba.append(probs.cpu().numpy())

    avg_loss = running_loss / max(total, 1)
    all_proba_arr = np.concatenate(all_proba, axis=0) if all_proba else np.empty((0, 0))
    f1 = f1_score(all_targets, all_preds, average="macro") if all_targets else 0.0

    return {
        "loss": avg_loss,
        "f1": float(f1),
        "y_true": all_targets,
        "y_pred": all_preds,
        "y_proba": all_proba_arr,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_global_seed(args.seed)
    device = get_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Caminho do manifest.csv relativo ao projeto
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "data" / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.csv não encontrado em {manifest_path}")

    # Dataloaders
    loaders = build_dataloaders(
        cfg=cfg,
        manifest_csv=str(manifest_path),
        batch_size=args.batch_size,
        num_workers=4,
    )
    train_loader = loaders.get("train")
    if train_loader is None:
        raise RuntimeError("Nenhum DataLoader 'train' disponível no manifest.csv")

    val_loader = loaders.get("val")

    # Modelo via create_model_from_config
    classes = cfg.get("classes")
    if not classes:
        raise ValueError("Config YAML deve definir 'classes'")

    cfg["model"] = {
        "name": args.model,
        "num_classes": len(classes),
        # Desabilita pesos pretrained por padrão para evitar downloads em ambientes sem internet
        "pretrained": False,
        "unfreeze_last_n": args.unfreeze_last_n,
    }

    model = create_model_from_config(cfg)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Otimizador AdamW + CosineAnnealingLR
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    best_ckpt_path = output_dir / f"best_{args.model}.pt"
    metrics_path = output_dir / f"metrics_{args.model}.json"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        log: Dict[str, Any] = {"epoch": epoch, "train_loss": float(train_loss)}
        msg = f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f}"

        if val_loader is not None:
            val_metrics = evaluate_on_loader(model, val_loader, criterion, device)
            val_loss = val_metrics["loss"]
            val_f1 = val_metrics["f1"]

            log.update({"val_loss": float(val_loss), "val_f1": float(val_f1)})
            msg += f" | val_loss: {val_loss:.4f}, val_f1: {val_f1:.4f}"

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0

                save_checkpoint(
                    path=str(best_ckpt_path),
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    extra={"config": cfg, "model_name": args.model},
                )
            else:
                patience_counter += 1
        else:
            # Sem validação, sempre sobrescreve o melhor
            save_checkpoint(
                path=str(best_ckpt_path),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra={"config": cfg, "model_name": args.model},
            )

        history.append(log)
        print(msg)

        scheduler.step()

        if val_loader is not None and patience_counter >= args.patience:
            print(f"Early stopping ativado (patience={args.patience}) na epoch {epoch}")
            break

    summary = {
        "best_val_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
        "history": history,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Treino finalizado. Melhor epoch: {best_epoch}, best_val_f1: {best_val_f1:.4f}")
    print(f"Checkpoint salvo em: {best_ckpt_path}")
    print(f"Métricas salvas em:   {metrics_path}")


if __name__ == "__main__":
    main()

