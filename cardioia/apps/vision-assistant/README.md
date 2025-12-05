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
data:
  root: "/CAMINHO/ABSOLUTO/PARA/DATA_DIR"  # aponte aqui para o diretório raiz do dataset de ECG
  image_size: 224
  batch_size: 32
  num_workers: 4
  pin_memory: true

train:
  num_epochs: 20
  learning_rate: 1e-4
  weight_decay: 1e-4
  device: "cuda"  # ou "cpu" se não houver GPU

model:
  name: "simple_cnn"   # ou "resnet18" (por exemplo, para transfer learning)
  num_classes: 2       # ajustar de acordo com o número de classes do seu dataset
```

O campo crítico para apontar o **DATA_DIR** é `data.root`. Basta editar esse caminho para a pasta onde você descompactou/baixou o dataset de ECG.

---

## Uso básico (conceitual)

### 1. Treino

Um script típico de treino em `src/train.py` poderá ser executado assim:

```bash
cd cardioia/apps/vision-assistant
python -m src.train --config configs/ecg.yaml
```

O script deve:

- Ler o YAML (`ecg.yaml`).
- Criar DataLoaders a partir de `data.root` usando `ECGImageDataset` (em `src/dataset.py`).
- Instanciar o modelo definido em `model.name`.
- Rodar o loop de treino/validação pelos `num_epochs`.
- Salvar o melhor modelo em um diretório (por exemplo, `checkpoints/`).

### 2. Avaliação

O script `src/evaluate.py` deverá carregar o modelo salvo e calcular métricas em `val/` ou `test/`, gerando gráficos (ex.: matriz de confusão) em uma pasta de resultados (ex.: `outputs/`).

### 3. Grad-CAM

O módulo `src/gradcam.py` fornecerá funções para:

- Gerar mapas Grad-CAM para imagens individuais.
- Salvar figuras com a sobreposição do mapa de calor sobre o ECG.

### 4. Interface Flask

O app Flask (`app/flask_app.py`) utilizará o modelo treinado para:

- Receber uploads de imagens de ECG pelo `index.html`.
- Rodar inferência usando o modelo carregado.
- Exibir a predição (classe / probabilidade) e, opcionalmente, o Grad-CAM correspondente.

---

## Próximos Passos

1. Ajustar `configs/ecg.yaml` para apontar corretamente para o `DATA_DIR` local.
2. Implementar/ajustar `train.py`, `evaluate.py`, `models.py` e `gradcam.py` conforme as necessidades do experimento.
3. Rodar o treino e avaliar as métricas.
4. Integrar o modelo na interface Flask para uso interativo.
