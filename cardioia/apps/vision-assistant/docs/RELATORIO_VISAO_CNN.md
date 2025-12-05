# Relatório – Visão: CNN para ECG

# Relatório – Visão: CNN para ECG em Imagem

## 1. Setup experimental

### 1.1 Ambiente

- **Linguagem**: Python 3.10+
- **Framework principal**: PyTorch
- **Bibliotecas auxiliares**:
  - `torchvision` (modelos pré-treinados, transforms)
  - `albumentations` (augmentations)
  - `scikit-learn` (métricas e relatórios)
  - `matplotlib` / `seaborn` (visualizações)
- **Organização do código**:
  - [src/dataset.py](cci:7://file:///Users/anakolodji/Desktop/ia/CardioIA/cardioia-app.fase4/cardioia/apps/vision-assistant/src/dataset.py:0:0-0:0) – DataLoaders a partir de `manifest.csv`
  - [src/models.py](cci:7://file:///Users/anakolodji/Desktop/ia/CardioIA/cardioia-app.fase4/cardioia/apps/vision-assistant/src/models.py:0:0-0:0) – CNN simples ([SimpleCNN](cci:2://file:///Users/anakolodji/Desktop/ia/CardioIA/cardioia-app.fase4/cardioia/apps/vision-assistant/src/models.py:69:0-111:48)) e `ResNet18` com fine-tuning
  - [src/train.py](cci:7://file:///Users/anakolodji/Desktop/ia/CardioIA/cardioia-app.fase4/cardioia/apps/vision-assistant/src/train.py:0:0-0:0) – script de treino com early stopping
  - [src/evaluate.py](cci:7://file:///Users/anakolodji/Desktop/ia/CardioIA/cardioia-app.fase4/cardioia/apps/vision-assistant/src/evaluate.py:0:0-0:0) – script de avaliação em `test`
  - [src/utils.py](cci:7://file:///Users/anakolodji/Desktop/ia/CardioIA/cardioia-app.fase4/cardioia/apps/vision-assistant/src/utils.py:0:0-0:0) – seed, checkpoints, métricas, plots

### 1.2 Dados de entrada

- **Tipo**: imagens PNG/JPG de **traçados de ECG**.
- **Estrutura**: pastas por classe, conforme [ecg.yaml](cci:7://file:///Users/anakolodji/Desktop/ia/CardioIA/cardioia-app.fase4/cardioia/apps/vision-assistant/configs/ecg.yaml:0:0-0:0):
  - `classes: ["normal", "abnormal"]`
- As imagens são indexadas e divididas em `train` / `val` / `test` via notebook de pré-processamento, resultando em:
  - `apps/vision-assistant/data/manifest.csv`
  - Colunas: `filepath`, `label`, `split`.

---

## 2. Arquiteturas utilizadas

### 2.1 SimpleCNN

Arquitetura CNN 2D leve, adequada como baseline:

- **Backbone**:
  - 3 blocos `Conv2d → BatchNorm2d → ReLU → MaxPool2d`:
    - Bloco 1: 3 → 32 canais
    - Bloco 2: 32 → 64 canais
    - Bloco 3: 64 → 128 canais
  - Para input 224×224, após 3 pools: 28×28.

- **Classifier**:
  - `Flatten`
  - `Linear(128×28×28 → 256)`
  - `ReLU`
  - `Dropout(p=0.3)`
  - `Linear(256 → num_classes)`

- **Inicialização**:
  - Convoluções e lineares inicializadas com **Kaiming normal** (He).
  - Bias inicializados com zero.

- **Saída**:
  - Logits `[batch_size, num_classes]`.

### 2.2 ResNet18 com fine-tuning (make_resnet18)

Arquitetura de **transfer learning** baseada em `ResNet18`:

- **Backbone**: ResNet18 pré-treinada em ImageNet (quando `pretrained=True`).
- **Cabeça de classificação**:
  - Substitui `fc` por:
    - `Dropout(p=0.3) → Linear(in_features, num_classes)`.
  - Inicialização Kaiming para a camada linear.

- **Congelamento de camadas**:
  - Por padrão, **todas as camadas** são congeladas.
  - Parâmetro `unfreeze_last_n` permite “descongelar” os últimos blocos:
    - Ordem: `[layer1, layer2, layer3, layer4, fc]`.
    - Ex.: `unfreeze_last_n = 2` → treinar `layer4` + `fc`.

Essa abordagem busca aproveitar representações aprendidas em imagens naturais e adaptá-las ao domínio de ECG.

---

## 3. Hiperparâmetros de treino

### 3.1 Configuração base

- **Otimizador**: `AdamW`
  - `lr = 3e-4` (configurável via CLI)
  - `weight_decay` padrão interno do AdamW (podendo ser ajustado futuramente).
- **Loss**: `CrossEntropyLoss`.
- **Scheduler**: `CosineAnnealingLR`
  - `T_max = epochs` (ajusta o `lr` ao longo de todo o treino).

### 3.2 Estratégia de treino

- **Épocas máximo**: 15 (ajustável).
- **Batch size**: 32 (ajustável).
- **Dispositivo**: CPU ou GPU (`cuda` se disponível).
- **Early stopping**:
  - Métrica monitorada: **F1 macro** no split de validação (`val_f1`).
  - `patience = 5` épocas sem melhora.
  - Salva `best_{model}.pt` sempre que `val_f1` melhora.

---

## 4. Resultados experimentais (exemplo simulado)

> Atenção: os números abaixo são **simulados** para ilustração do relatório e devem ser substituídos por resultados reais após a execução completa dos scripts de treino/avaliação.

### 4.1 Tabela resumo

| Modelo     | Acurácia Val | F1 Macro Val | Acurácia Teste | F1 Macro Teste | ROC AUC (binário) |
|-----------|--------------|--------------|----------------|----------------|--------------------|
| SimpleCNN | 0.82         | 0.80         | 0.81           | 0.79           | 0.86               |
| ResNet18  | 0.87         | 0.85         | 0.86           | 0.84           | 0.90               |

- **Observações**:
  - A `ResNet18` com `unfreeze_last_n = 2` supera a [SimpleCNN](cci:2://file:///Users/anakolodji/Desktop/ia/CardioIA/cardioia-app.fase4/cardioia/apps/vision-assistant/src/models.py:69:0-111:48) em todas as métricas.
  - A diferença de F1 macro indica melhor equilíbrio entre as classes `normal` e `abnormal`.

### 4.2 Figuras geradas

Os scripts produzem, para cada modelo:

- `confusion_{model}.png` – **Matriz de confusão** normalizada.
  - Permite visualizar onde o modelo mais erra (ex.: confusão entre `abnormal` leve e `normal`).
- `roc_{model}.png` – **Curvas ROC** (one-vs-rest se multi-classe).
  - Para o problema binário, destaca a curva para a classe `abnormal` (patológica).

Essas figuras devem ser incluídas em relatórios ou apresentações para ilustrar o desempenho do modelo.

---

## 5. Grad-CAM e insights clínicos (simulados)

Embora a implementação de Grad-CAM esteja em [src/gradcam.py](cci:7://file:///Users/anakolodji/Desktop/ia/CardioIA/cardioia-app.fase4/cardioia/apps/vision-assistant/src/gradcam.py:0:0-0:0) (ainda em evolução), a ideia é:

- Gerar **mapas de ativação** sobre as imagens de ECG que indicam quais regiões mais contribuíram para a decisão do modelo.
- Para casos classificados como `abnormal`, espera-se ver:
  - Maior ativação em trechos onde há **alterações de segmento ST**, ondas T anormais ou batimentos ectópicos.
- Para casos `normal`, o Grad-CAM tende a se distribuir de forma mais homogênea, sem focos intensos em irregularidades.

### Exemplos de interpretações simuladas

- **Caso 1 – `abnormal`**:
  - Grad-CAM destaca uma região do traçado na derivação simulada onde há elevação do segmento ST.
  - Insight (simulado): o modelo possivelmente está capturando padrões compatíveis com isquemia/infarto.

- **Caso 2 – `normal`**:
  - Grad-CAM mostra ativação suave ao longo do traçado, sem “hotspots” isolados.
  - Insight (simulado): o modelo não encontrou padrões fortemente desviantes dos traçados normais vistos em treino.

> Importante: esses insights são **exploratórios** e dependem da qualidade dos dados e da implementação detalhada de Grad-CAM. Não constituem evidência clínica robusta.

---

## 6. Limitações

- **Dados limitados / desbalanceados**:
  - Possível desbalanceamento entre `normal` e `abnormal`.
  - Risco de o modelo superestimar a classe majoritária.

- **Generalização**:
  - Dataset possivelmente proveniente de poucos centros.
  - Pode não generalizar para outros hospitais, equipamentos ou populações.

- **Rótulos clínicos**:
  - Podem conter:
    - Erros de anotação.
    - Divergências entre especialistas.
  - Isso impacta diretamente as métricas do modelo.

- **Grad-CAM**:
  - Embora útil, Grad-CAM é apenas um indicativo qualitativo.
  - Não substitui interpretação médica baseada em conhecimento clínico.

---

## 7. Próximos passos

- **Melhorar o balanceamento de classes**:
  - Oversampling da classe minoritária.
  - Ponderação de loss (`class_weight`).

- **Explorar outras arquiteturas**:
  - Modelos mais profundos (ResNet34/50, EfficientNet).
  - Modelos especializados em sinais fisiológicos convertidos para imagem.

- **Validação cruzada**:
  - K-fold cross-validation para reduzir variância dos resultados.
  - Relatar média ± desvio padrão das métricas.

- **Estudos ablatórios**:
  - Comparar:
    - Sem / com augmentations.
    - Diferentes `unfreeze_last_n` na ResNet18.
    - Normalização de ImageNet vs. normalização específica do dataset.

- **Integração com Grad-CAM na interface Flask**:
  - Permitir upload de uma imagem de ECG.
  - Mostrar predição + mapa Grad-CAM.
  - Coletar feedback qualitativo de cardiologistas.

---

## 8. Uso acadêmico — não clínico

Este projeto e os modelos aqui descritos são **estritamente experimentais** e destinados a:

- Pesquisa acadêmica.
- Prototipagem de soluções baseadas em visão computacional para ECG.
- Exploração de técnicas de explicabilidade (Grad-CAM) em dados médicos.

Eles **NÃO** devem ser utilizados para:

- **Tomada de decisão clínica**, diagnóstico ou triagem em pacientes reais.
- Substituir julgamento de **profissionais de saúde qualificados**.
- Automatizar qualquer parte de um fluxo assistencial sem estudos clínicos formais, validação regulatória e análise ética.

Qualquer uso em contexto real exigiria:

- Protocolos de validação rigorosos.
- Aprovação de comitês de ética e órgãos regulatórios.
- Participação de especialistas médicos na avaliação e interpretação dos resultados.