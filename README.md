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
numpy<2.0
torch>=2.2.0
torchvision>=0.17.0
torchaudio>=2.2.0
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

Na raiz do projeto (onde está a pasta `cardioia/`), recomenda-se criar um ambiente virtual e instalar as dependências do módulo:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Unix/macOS

pip install --upgrade pip
pip install -r cardioia/apps/vision-assistant/requirements.txt
```

No Windows, a ativação típica é:

```bash
.venv\Scripts\activate
```

---

## Estrutura dos dados e manifest.csv

No uso atual, o dataset real de ECG é organizado em uma pasta bruta `ecg_raw/` (com pastas originais do autor do dataset) e um arquivo `manifest.csv` com os caminhos e rótulos finais em português.

Estrutura típica em `apps/vision-assistant/data/`:

```text
data/
  ecg_raw/
    ECG Images of Myocardial Infarction Patients/
    ECG Images of Patient that have abnormal heartbeat/
    ECG Images of Patient that have History of MI/
    Normal Person ECG Images (284x12=3408)/

  manifest.csv   # gerado a partir do notebook 01_preprocess ou do script make_manifest_ecg.py
  README.md
```

O `manifest.csv` contém colunas:

- `filepath`: caminho absoluto da imagem
- `label`: rótulo em português (`infarto_mi`, `batimento_anormal`, `historico_infarto`, `normal`)
- `split`: `train`, `val` ou `test`

Esse arquivo é consumido pelos DataLoaders em `src/dataset.py` e pelos scripts de treino/avaliação.

---

## Configuração via YAML (ecg.yaml)

Um exemplo mínimo de `configs/ecg.yaml` (já alinhado com o dataset real e o manifest) é:

```yaml
"""Configuração base para treino em ECG (imagens PNG/JPG em pastas por classe).

Edite principalmente:
- data_dir: caminho para o diretório raiz do dataset de ECG
- classes: lista de nomes de classes na ordem desejada
"""

data_dir: "cardioia/apps/vision-assistant/data"

image_size: 224

normalize_mean: [0.485, 0.456, 0.406]
normalize_std:  [0.229, 0.224, 0.225]

train_split: 0.7
val_split:   0.15
test_split:  0.15

classes:
  - infarto_mi
  - batimento_anormal
  - historico_infarto
  - normal
```

O campo crítico para apontar o **DATA_DIR** é `data_dir`. O `manifest.csv` é esperado em `data/manifest.csv` dentro desse diretório.

---

## Uso básico (conceitual)

### 1. Pré-processamento e manifest.csv (Notebook 01)

Use o notebook `notebooks/01_preprocess.ipynb` (ou o script `scripts/make_manifest_ecg.py`) para:

- Ler as pastas brutas em `data/ecg_raw/`.
- Mapear nomes de pastas originais → rótulos em português.
- Criar splits estratificados (`train`/`val`/`test`).
- Gerar o arquivo `data/manifest.csv` com colunas `filepath`, `label`, `split`.

Esse `manifest.csv` é usado pelos scripts de treino e avaliação.

### 2. Treino (Notebook 02 ou terminal)

O notebook `notebooks/02_cnn_training.ipynb` mostra exemplos de como disparar o treino e a avaliação a partir do Python, além de visualizar o histórico e as figuras geradas.

Diretamente no terminal (a partir da raiz `cardioia-app.fase4`), o treino pode ser feito assim:

```bash
source .venv/bin/activate  # se ainda não estiver ativo

python3 cardioia/apps/vision-assistant/src/train.py \
  --config cardioia/apps/vision-assistant/configs/ecg.yaml \
  --model resnet18 \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-4 \
  --unfreeze-last-n 2
```

O script irá:

- Ler o YAML (`ecg.yaml`) e o `manifest.csv` em `data/`.
- Criar DataLoaders a partir do manifest usando `build_dataloaders` (em `src/dataset.py`).
- Instanciar o modelo (`SimpleCNN` ou `ResNet18`) conforme `--model`.
- Treinar usando AdamW + CosineAnnealingLR, com early stopping baseado em F1 macro de validação.
- Salvar o melhor modelo em `checkpoints/best_{model}.pt` e as métricas em `checkpoints/metrics_{model}.json`.

### 3. Avaliação

Após o treino, o script `src/evaluate.py` permite avaliar o modelo no split de teste e gerar artefatos de relatório:

```bash
source .venv/bin/activate  # se necessário

python3 cardioia/apps/vision-assistant/src/evaluate.py \
  --config cardioia/apps/vision-assistant/configs/ecg.yaml \
  --model resnet18
```

O script irá:

- Carregar o checkpoint `checkpoints/best_{model}.pt`.
- Rodar inferência no split `test` (ou `val` como fallback) usando o `manifest.csv`.
- Calcular métricas com `scikit-learn` (accuracy, F1, ROC AUC quando aplicável).
- Salvar:
  - `confusion_matrix_{model}.png` – matriz de confusão normalizada.
  - `classification_report_{model}.txt` – relatório de classificação textual.
  - `roc_curves_{model}.png` – curvas ROC (quando probabilidades disponíveis).

### 4. Interface Flask

Após treinar e ter o checkpoint `checkpoints/best_resnet18.pt`, é possível rodar uma interface web simples para inferência e Grad-CAM:

```bash
source .venv/bin/activate  # se necessário

export FLASK_APP=cardioia.apps.vision-assistant.app.flask_app
python3 -m flask --app cardioia.apps.vision-assistant.app.flask_app run
```

Isso irá:

- Subir um servidor Flask em `http://localhost:5000/`.
- Carregar automaticamente o modelo `best_resnet18.pt`.
- Permitir upload de uma imagem de ECG (PNG/JPEG) via formulário.
- Aplicar o mesmo pré-processamento usado no treino (`image_size`, `normalize_mean/std`).
- Exibir:
  - Classe prevista (`infarto_mi`, `batimento_anormal`, `historico_infarto`, `normal`).
  - Probabilidade estimada.
  - Imagem de saída com overlay de Grad-CAM salva em `app/static/`.

---

## Próximos Passos

1. Ajustar `configs/ecg.yaml` para apontar corretamente para o `data_dir` local e classes em português.
2. Rodar o notebook de pré-processamento (`01_preprocess.ipynb`) ou o script `scripts/make_manifest_ecg.py` para gerar o `manifest.csv`.
3. Treinar modelos (`simple` e/ou `resnet18`) com `src/train.py` (via notebook 02 ou terminal) e comparar as métricas.
4. Avaliar no split de teste com `src/evaluate.py`, analisando matriz de confusão e curvas ROC.
5. Integrar Grad-CAM de forma mais avançada em `src/gradcam.py` e na interface Flask.
6. Explorar uso em outros datasets (ex.: `chestxray.yaml`).
