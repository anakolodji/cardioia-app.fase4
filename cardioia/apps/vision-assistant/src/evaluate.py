import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import yaml

from dataset import build_dataloaders
from models import create_model_from_config
from utils import (
    compute_classification_metrics,
    load_checkpoint,
    plot_confusion_matrix,
    plot_roc_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliação de modelos no split de teste")
    parser.add_argument("--config", type=str, required=True, help="Caminho para o YAML (ex.: configs/ecg.yaml)")
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "resnet18"], help="Arquitetura do modelo")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size para avaliação")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Diretório com checkpoints e para salvar figuras/relatórios")
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


@torch.no_grad()
def collect_outputs(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_proba: List[np.ndarray] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_proba.append(probs.cpu().numpy())

    all_proba_arr = np.concatenate(all_proba, axis=0) if all_proba else np.empty((0, 0))
    return {"y_true": all_targets, "y_pred": all_preds, "y_proba": all_proba_arr}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Caminho do manifest.csv relativo ao projeto
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "data" / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.csv não encontrado em {manifest_path}")

    # Dataloaders (test ou val como fallback)
    loaders = build_dataloaders(
        cfg=cfg,
        manifest_csv=str(manifest_path),
        batch_size=args.batch_size,
        num_workers=0,
    )

    test_loader = loaders.get("test") or loaders.get("val")
    split_name = "test" if "test" in loaders else "val"
    if test_loader is None:
        raise RuntimeError("Nem split 'test' nem 'val' disponíveis no manifest.csv")

    classes = cfg.get("classes")
    if not classes:
        raise ValueError("Config YAML deve definir 'classes'")

    # Reconstrói modelo e carrega checkpoint
    cfg["model"] = {
        "name": args.model,
        "num_classes": len(classes),
        "pretrained": False,
        "unfreeze_last_n": 0,
    }

    model = create_model_from_config(cfg)
    model.to(device)

    ckpt_path = output_dir / f"best_{args.model}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    load_checkpoint(str(ckpt_path), model, optimizer=None, map_location=device)

    # Coleta outputs
    outputs = collect_outputs(model, test_loader, device)
    y_true = outputs["y_true"]
    y_pred = outputs["y_pred"]
    y_proba = outputs["y_proba"]

    metrics = compute_classification_metrics(y_true, y_pred, y_proba, class_names=classes)

    print("=== Métricas de teste (split: %s) ===" % split_name)
    print("Accuracy:", metrics["accuracy"])
    if "roc_auc" in metrics:
        print("ROC AUC:", metrics["roc_auc"])
    print("\nClassification report:\n")
    print(metrics["classification_report"])

    # Salvar artefatos
    confusion_path = output_dir / f"confusion_{args.model}.png"
    report_path = output_dir / f"report_{args.model}.txt"
    roc_path = output_dir / f"roc_{args.model}.png"

    # Matriz de confusão (normalizada)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names=classes, normalize=True)
    import matplotlib.pyplot as plt

    plt.savefig(confusion_path, bbox_inches="tight")
    plt.close()

    # Relatório em texto
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Métricas de teste (split: %s) ===\n" % split_name)
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        if "roc_auc" in metrics:
            f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
        f.write("\nClassification report:\n\n")
        f.write(metrics["classification_report"])

    # Curvas ROC se tivermos probabilidades adequadas
    if y_proba.size != 0 and y_proba.shape[1] == len(classes):
        plot_roc_curves(np.array(y_true), y_proba, class_names=classes)
        plt.savefig(roc_path, bbox_inches="tight")
        plt.close()

    print(f"Arquivos salvos em: {output_dir}")
    print(f"- Matriz de confusão: {confusion_path}")
    print(f"- Relatório:          {report_path}")
    if y_proba.size != 0:
        print(f"- Curvas ROC:         {roc_path}")


if __name__ == "__main__":
    main()

