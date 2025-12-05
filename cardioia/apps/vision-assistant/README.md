# Vision Assistant – ECG (CardioIA)

Este módulo implementa um assistente de visão computacional para análise de **ECG em imagens** (PNG/JPG) usando **PyTorch** e uma arquitetura simples de CNN/Transfer Learning.

A estrutura do projeto segue a árvore:

```text
cardioia/
  apps/
    vision-assistant/
      data/                     # links/README sobre datasets
      notebooks/
        01_preprocess.ipynb
        02_cnn_training.ipynb
      src/
        dataset.py              # DataLoader PyTorch para ECG em imagem
        models.py               # CNN simples + Transfer Learning
        train.py                # Script de treino (CLI)
        evaluate.py             # Métricas + matriz de confusão
        gradcam.py              # Explicabilidade (Grad-CAM)
        utils.py
      app/
        flask_app.py            # Protótipo de interface Flask
        templates/
          index.html
      configs/
        ecg.yaml                # Config para ECG (DATA_DIR, hiperparâmetros etc.)
        chestxray.yaml          # (reserva para outro dataset)
      requirements.txt
      README.md
```

---

## Objetivos

- **Treinar modelos de classificação** de ECG em imagem (ex.: normal vs patológico) usando CNN 2D e/ou modelos pré-treinados.
- **Avaliar o desempenho** com métricas padrão (accuracy, F1, matriz de confusão).
- **Fornecer explicabilidade** via Grad-CAM para destacar regiões relevantes do sinal/imagem.
- **Disponibilizar um protótipo de interface web** (Flask) para upload de imagens de ECG e visualização da predição + mapa de calor (Grad-CAM).

---

## Requisitos

As principais dependências estão em `requirements.txt`:

```text
torch>=2.3; sys_platform!="darwin" or platform_machine!="arm64"
torchvision>=0.18
torchaudio>=2.3
# tensorflow>=2.16; python_version>="3.10" ; extra == "tf-option"  # deixe comentado se optar só por PyTorch
scikit-learn>=1.5
opencv-python>=4.10
matplotlib>=3.9
seaborn>=0.13
pandas>=2.2
pillow>=10.4
albumentations>=1.4
grad-cam>=1.5
Flask>=3.0
pyyaml>=6.0
```

> Observação: a linha de TensorFlow está comentada por padrão. Use apenas se desejar experimentar uma versão em TF.

### Instalação

Dentro da pasta `apps/vision-assistant`, execute:

```bash
pip install -r requirements.txt
```

Recomenda-se usar um ambiente virtual (por exemplo, `python -m venv .venv` e depois `source .venv/bin/activate` em Unix/macOS, ou `.venv\\Scripts\\activate` no Windows).

---

## Estrutura esperada dos dados (DATA_DIR)

Os dados de ECG em imagem são esperados no formato de **pastas por classe**, separados em `train/`, `val/` e `test/` (os últimos dois opcionais):

```text
DATA_DIR/
  train/
    classe_0/  *.png, *.jpg, ...
    classe_1/  *.png, *.jpg, ...
    ...
  val/        (opcional)
    classe_0/  *.png, *.jpg, ...
    classe_1/  *.png, *.jpg, ...
  test/       (opcional)
    classe_0/  *.png, *.jpg, ...
    classe_1/  *.png, *.jpg, ...
```

No código, esse diretório raiz é referenciado como `DATA_DIR` através do arquivo de configuração YAML (`configs/ecg.yaml`).

---

## Configuração via YAML (ecg.yaml)

Um exemplo mínimo de `configs/ecg.yaml` é:

```yaml
data_dir: "/CAMINHO/ABSOLUTO/PARA/SEU_DATASET_ECG"  # diretório com subpastas por classe

image_size: 224

normalize_mean: [0.485, 0.456, 0.406]
normalize_std:  [0.229, 0.224, 0.225]

train_split: 0.7
val_split:   0.15
test_split:  0.15

classes: ["normal", "abnormal"]
```

O campo crítico para apontar o **DATA_DIR** é `data_dir`. Basta editar esse caminho para a pasta onde você descompactou/baixou o dataset de ECG.

---

## Uso básico (conceitual)

### 1. Pré-processamento e manifest.csv

Use o notebook `notebooks/01_preprocess.ipynb` para:

- Ler `configs/ecg.yaml`.
- Indexar as imagens em `data_dir`.
- Criar splits estratificados (`train`/`val`/`test`).
- Gerar o arquivo `data/manifest.csv` com colunas `filepath`, `label`, `split`.

Esse `manifest.csv` é usado pelos scripts de treino e avaliação.

### 2. Treino

Um script típico de treino em `src/train.py` pode ser executado assim (exemplos):

```bash
cd cardioia/apps/vision-assistant

# Treinar SimpleCNN
python -m src.train \
  --config configs/ecg.yaml \
  --model simple \
  --epochs 15 \
  --batch-size 32 \
  --lr 3e-4

# Treinar ResNet18 com fine-tuning (descongelando últimos blocos)
python -m src.train \
  --config configs/ecg.yaml \
  --model resnet18 \
  --epochs 15 \
  --batch-size 32 \
  --lr 3e-4 \
  --unfreeze-last-n 2
```

O script irá:

- Ler o YAML (`ecg.yaml`) e o `manifest.csv`.
- Criar DataLoaders a partir do manifest usando `build_dataloaders` (em `src/dataset.py`).
- Instanciar o modelo (`SimpleCNN` ou `ResNet18`) conforme `--model`.
- Treinar usando AdamW + CosineAnnealingLR, com early stopping baseado em F1 macro de validação.
- Salvar o melhor modelo em `checkpoints/best_{model}.pt` e as métricas em `checkpoints/metrics_{model}.json`.

### 3. Avaliação

Após o treino, o script `src/evaluate.py` permite avaliar o modelo no split de teste e gerar artefatos de relatório:

```bash
cd cardioia/apps/vision-assistant

python -m src.evaluate \
  --config configs/ecg.yaml \
  --model resnet18
```

O script irá:

- Carregar o checkpoint `checkpoints/best_{model}.pt`.
- Rodar inferência no split `test` (ou `val` como fallback) usando o `manifest.csv`.
- Calcular métricas com `scikit-learn` (accuracy, F1, ROC AUC quando aplicável).
- Salvar:
  - `confusion_{model}.png` – matriz de confusão normalizada.
  - `report_{model}.txt` – relatório de classificação textual.
  - `roc_{model}.png` – curvas ROC (quando probabilidades disponíveis).

### 4. Interface Flask

Após treinar e ter o checkpoint `checkpoints/best_resnet18.pt`, é possível rodar uma interface web simples para inferência e Grad-CAM:

```bash
cd cardioia/apps/vision-assistant/app
python flask_app.py
```

Isso irá:

- Subir um servidor Flask em `http://localhost:5000/`.
- Carregar automaticamente o modelo `best_resnet18.pt`.
- Permitir upload de uma imagem de ECG (PNG/JPEG) via formulário.
- Aplicar o mesmo pré-processamento usado no treino (`image_size`, `normalize_mean/std`).
- Exibir:
  - Classe prevista (`normal`/`abnormal`).
  - Probabilidade estimada.
  - Imagem de saída com overlay de Grad-CAM salva em `app/static/`.

---

## Próximos Passos

1. Ajustar `configs/ecg.yaml` para apontar corretamente para o `DATA_DIR` local.
2. Rodar o notebook de pré-processamento para gerar o `manifest.csv`.
3. Treinar modelos (`simple` e/ou `resnet18`) com `src/train.py` e comparar as métricas.
4. Avaliar no split de teste com `src/evaluate.py`, analisando matriz de confusão e curvas ROC.
5. Integrar Grad-CAM de forma mais avançada em `src/gradcam.py` e na interface Flask.
6. Explorar uso em outros datasets (ex.: `chestxray.yaml`).
