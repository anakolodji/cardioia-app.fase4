import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import yaml
from flask import Flask, render_template, request, redirect, url_for


# Ajusta caminho para importar módulos de src/
BASE_DIR = Path(__file__).resolve().parents[1]  # .../apps/vision-assistant
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import create_model_from_config  # type: ignore  # noqa: E402


app = Flask(__name__, static_folder="static", template_folder="templates")


def load_config() -> Dict[str, Any]:
    config_path = BASE_DIR / "configs" / "ecg.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    return cfg


def load_model_and_cfg() -> Tuple[nn.Module, Dict[str, Any]]:
    cfg = load_config()
    classes = cfg.get("classes")
    if not classes:
        raise ValueError("Config YAML deve definir 'classes'")

    # Tenta carregar config salva no checkpoint, se existir
    ckpt_path = BASE_DIR / "checkpoints" / "best_resnet18.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint best_resnet18.pt não encontrado em {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")

    model_cfg = checkpoint.get("config", {}).get("model")
    if not model_cfg:
        # Fallback: constrói config de modelo manualmente
        cfg["model"] = {
            "name": "resnet18",
            "num_classes": len(classes),
            "pretrained": False,
            "unfreeze_last_n": 0,
        }
    else:
        cfg["model"] = model_cfg

    model = create_model_from_config(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


MODEL, CFG = load_model_and_cfg()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL.to(DEVICE)


def preprocess_image(img: Image.Image, cfg: Dict[str, Any]) -> torch.Tensor:
    image_size = int(cfg.get("image_size", 224))
    mean = cfg.get("normalize_mean", [0.485, 0.456, 0.406])
    std = cfg.get("normalize_std", [0.229, 0.224, 0.225])

    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - np.array(mean).reshape(1, 1, 3)) / np.array(std).reshape(1, 1, 3)
    img_np = np.transpose(img_np, (2, 0, 1))  # C,H,W
    tensor = torch.from_numpy(img_np).unsqueeze(0)  # 1,C,H,W
    return tensor


class GradCAM:
    """Implementação simples de Grad-CAM para o último bloco convolucional da ResNet18."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):  # type: ignore[no-redef]
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):  # type: ignore[no-redef]
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM: activations ou gradients não disponíveis")

        gradients = self.gradients  # [C,H,W]
        activations = self.activations  # [C,H,W]

        # Global average pooling dos gradientes
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C,1,1]
        cam = (weights * activations).sum(dim=0)  # [H,W]

        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def generate_gradcam_overlay(
    img_pil: Image.Image,
    cam: np.ndarray,
    out_path: Path,
) -> None:
    """Gera overlay de Grad-CAM sobre a imagem original e salva em out_path (PNG)."""

    img = np.array(img_pil.convert("RGB"))
    h, w, _ = img.shape
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.4 * heatmap + 0.6 * img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path)


def get_target_layer_for_resnet18(model: nn.Module) -> nn.Module:
    # Usa o último bloco convolucional (layer4) por padrão
    return model.layer4[-1]  # type: ignore[return-value]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    heatmap_url = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Nenhuma imagem enviada."
        else:
            try:
                img = Image.open(file.stream)

                # Pré-processamento
                tensor = preprocess_image(img, CFG).to(DEVICE)

                # Forward e previsão
                with torch.no_grad():
                    logits = MODEL(tensor)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                classes = CFG.get("classes", [])
                pred_idx = int(np.argmax(probs))
                pred_class = classes[pred_idx] if classes else str(pred_idx)
                pred_prob = float(probs[pred_idx])

                # Grad-CAM
                target_layer = get_target_layer_for_resnet18(MODEL)
                gradcam = GradCAM(MODEL, target_layer)
                cam = gradcam(tensor, class_idx=pred_idx)

                static_dir = Path(app.static_folder or "static")
                static_dir.mkdir(parents=True, exist_ok=True)
                out_path = static_dir / "gradcam_overlay.png"
                generate_gradcam_overlay(img, cam, out_path)

                prediction = pred_class
                probability = f"{pred_prob:.3f}"
                heatmap_url = url_for("static", filename="gradcam_overlay.png")
            except Exception as exc:  # pragma: no cover - log simples
                error = f"Erro ao processar imagem: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        heatmap_url=heatmap_url,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

