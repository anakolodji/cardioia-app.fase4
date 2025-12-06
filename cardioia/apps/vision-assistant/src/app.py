import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from flask import Flask, render_template_string, request
from PIL import Image
import torch
import yaml

from dataset import build_transforms
from models import create_model_from_config
from utils import load_checkpoint


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


def prepare_model(config_path: str, model_name: str, checkpoints_dir: str) -> tuple[torch.nn.Module, list[str]]:
    cfg = load_config(config_path)
    classes = cfg.get("classes")
    if not classes:
        raise ValueError("Config YAML deve definir 'classes'")

    cfg["model"] = {
        "name": model_name,
        "num_classes": len(classes),
        "pretrained": False,
        "unfreeze_last_n": 0,
    }

    device = get_device()
    model = create_model_from_config(cfg)
    model.to(device)
    model.eval()

    ckpt_path = Path(checkpoints_dir) / f"best_{model_name}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    load_checkpoint(str(ckpt_path), model, optimizer=None, map_location=device)

    return model, list(classes)


def preprocess_image(img: Image.Image, cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    img_np = np.array(img.convert("RGB"))
    transform = build_transforms(cfg, split="test")
    transformed = transform(image=img_np)
    tensor = transformed["image"].unsqueeze(0).to(device)
    return tensor


def create_app() -> Flask:
    app = Flask(__name__)

    # Configurações básicas (ajuste o caminho para o seu YAML se necessário)
    project_root = Path(__file__).resolve().parents[3]
    default_config_path = project_root / "apps" / "vision-assistant" / "configs" / "ecg.yaml"
    default_model_name = os.environ.get("VISION_MODEL_NAME", "simple")
    checkpoints_dir = project_root / "checkpoints"

    cfg = load_config(str(default_config_path))
    model, class_names = prepare_model(str(default_config_path), default_model_name, str(checkpoints_dir))
    device = get_device()

    TEMPLATE = """
    <!doctype html>
    <html lang="pt-br">
    <head>
        <meta charset="utf-8">
        <title>Classificação de ECG - CNN</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .preview { margin-top: 20px; }
            img { max-width: 400px; height: auto; border: 1px solid #ccc; }
            table { border-collapse: collapse; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px 12px; }
            th { background-color: #f4f4f4; }
        </style>
    </head>
    <body>
    <div class="container">
        <h1>Classificação de ECG - Modelo {{ model_name }}</h1>
        <form method="post" enctype="multipart/form-data">
            <label>Selecione uma imagem ECG (.png/.jpg):</label><br>
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Enviar</button>
        </form>

        {% if error %}
            <p style="color: red;">{{ error }}</p>
        {% endif %}

        {% if image_url %}
        <div class="preview">
            <h2>Imagem enviada</h2>
            <img src="{{ image_url }}" alt="ECG enviado">
        </div>
        {% endif %}

        {% if prediction %}
        <div class="results">
            <h2>Resultado</h2>
            <p><strong>Classe prevista:</strong> {{ prediction }}</p>

            <h3>Probabilidades por classe</h3>
            <table>
                <tr><th>Classe</th><th>Probabilidade</th></tr>
                {% for cls, prob in probs %}
                    <tr>
                        <td>{{ cls }}</td>
                        <td>{{ "%.4f"|format(prob) }}</td>
                    </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
    </div>
    </body>
    </html>
    """

    @app.route("/", methods=["GET", "POST"])
    def index():
        error = None
        image_url = None
        prediction = None
        probs_table = None

        if request.method == "POST":
            file = request.files.get("image")
            if not file or file.filename == "":
                error = "Nenhum arquivo enviado."
            else:
                try:
                    img = Image.open(file.stream).convert("RGB")

                    with torch.no_grad():
                        tensor = preprocess_image(img, cfg, device)
                        outputs = model(tensor)
                        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

                    pred_idx = int(probabilities.argmax())
                    prediction = class_names[pred_idx]
                    probs_table = list(zip(class_names, probabilities.tolist()))

                except Exception as e:  # noqa: BLE001
                    error = f"Erro ao processar imagem: {e}"

        return render_template_string(
            TEMPLATE,
            model_name=default_model_name,
            error=error,
            image_url=image_url,
            prediction=prediction,
            probs=probs_table,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_RUN_PORT", "5000"))
    app.run(host=host, port=port, debug=True)
