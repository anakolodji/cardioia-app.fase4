import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt


def set_global_seed(seed: int) -> None:
    """Fixa seeds para reprodutibilidade básica (Python, NumPy, PyTorch)."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Salva checkpoint de modelo/otimizador em um arquivo .pt/.pth."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if extra is not None:
        payload.update(extra)

    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Carrega checkpoint e restaura modelo/otimizador.

    Retorna o dicionário completo carregado do arquivo para acesso a metadados
    (por exemplo, epoch, config etc.).
    """

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calcula métricas de classificação usando sklearn.

    Retorna um dicionário com:
      - accuracy
      - classification_report (string)
      - confusion_matrix (np.ndarray)
      - roc_auc (se y_proba fornecido e problema binário)
    """

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    report = classification_report(y_true_arr, y_pred_arr, target_names=class_names)
    cm = confusion_matrix(y_true_arr, y_pred_arr)

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    # ROC AUC apenas se probabilidades e problema binário
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 2:
        pos_proba = y_proba[:, 1]
        try:
            roc_auc = roc_auc_score(y_true_arr, pos_proba)
            metrics["roc_auc"] = roc_auc
        except ValueError:
            pass

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    figsize: Tuple[int, int] = (6, 6),
    cmap: str = "Blues",
) -> None:
    """Plota matriz de confusão com matplotlib."""

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.show()


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """Plota curvas ROC multi-classe (one-vs-rest) se houver probabilidades."""

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    n_classes = y_proba.shape[1]
    assert n_classes == len(class_names)

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n_classes):
        # binariza labels para classe i
        y_true_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_proba[:, i])
        try:
            auc = roc_auc_score(y_true_bin, y_proba[:, i])
            label = f"{class_names[i]} (AUC = {auc:.3f})"
        except ValueError:
            label = class_names[i]
        ax.plot(fpr, tpr, label=label)

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.show()

