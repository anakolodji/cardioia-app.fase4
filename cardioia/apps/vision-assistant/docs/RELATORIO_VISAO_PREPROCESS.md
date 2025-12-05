# Relatório – Visão: Pré-processamento de ECG

## Dataset escolhido e estrutura de pastas

### Dataset

Para este experimento estamos usando um dataset de **ECG em imagens** (PNG/JPG) obtido a partir de traçados de eletrocardiograma. Cada imagem representa um exame (ou janela de um exame) e é rotulada em categorias clínicas, por exemplo:

- **normal**
- **abnormal** (ou outra nomenclatura específica, como tipos de arritmia)

O dataset foi organizado localmente em um diretório referenciado no YAML por:

```yaml
data_dir: "/CAMINHO/ABSOLUTO/PARA/SEU_DATASET_ECG"
classes: ["normal", "abnormal"]
```

### Estrutura de pastas

O dataset foi organizado em uma estrutura de pastas por classe, conforme o exemplo abaixo:

```text
DATA_DIR/
  train/
    normal/ *.png
    abnormal/ *.png
  val/        (opcional)
  test/       (opcional)
```

### Pré-processamento

No notebook de pré-processamento, esse diretório é varrido recursivamente e cada imagem é indexada com:

filepath: caminho absoluto da imagem.
label: nome da classe (por exemplo, normal ou abnormal).
Em seguida fazemos um split estratificado para gerar o manifest.csv com colunas:

filepath
label
split (train, val, test)
O arquivo é salvo em:

text
apps/vision-assistant/data/manifest.csv
Esse manifest passa a ser a “fonte de verdade” para treino/validação/teste.

Tamanho da imagem e rationale
Tamanho (image_size)
Utilizamos um parâmetro configurável image_size definido no YAML:

yaml
image_size: 224
Escolhas principais:

224×224:
Padrão usado em CNNs 2D clássicas (ResNet, VGG, etc.).
Compatível com modelos pré-treinados em ImageNet (por exemplo, resnet18), que esperam esse tamanho ou algo próximo.
Balanceia:
Detalhe suficiente para padrões finos do traçado ECG.
Custo computacional razoável para treino em GPU ou até CPU.
O redimensionamento é feito on-the-fly no loader (via Albumentations / torchvision), sem sobrescrever os arquivos originais, preservando o dataset bruto.

Normalização e rationale
Parâmetros de normalização
No YAML:

yaml
normalize_mean: [0.485, 0.456, 0.406]
normalize_std:  [0.229, 0.224, 0.225]
Esses valores são as médias e desvios padrão por canal do ImageNet (dataset padrão de imagens naturais RGB).

Por que usar estatísticas do ImageNet?
Compatibilidade com modelos pré-treinados:
Modelos como resnet18 foram treinados com essas estatísticas.
Manter a mesma normalização ajuda a reutilizar representações aprendidas, mesmo em um domínio diferente (ECG em imagem).
Prática comum:
Em muitos problemas de visão onde só se dispõe de imagens RGB “convencionais”, é comum herdar essa normalização ao usar transfer learning.
Limitações dessa escolha
As estatísticas de intensidade de ECG podem diferir bastante das do ImageNet.
Uma alternativa (mais específica ao domínio) seria:
Estimar mean/std diretamente no dataset de ECG.
Atualizar o YAML com esses novos valores.
Entretanto, para uma primeira versão focada em transfer learning, a normalização de ImageNet é aceitável e simples de replicar.
Importante: a normalização é aplicada apenas no pipeline de carregamento (transforms); as imagens em disco permanecem inalteradas.

Augmentations
Objetivo
Augmentations servem para:

Aumentar a robustez do modelo a pequenas variações de aquisição (iluminação, contraste, ruído).
Reduzir overfitting, especialmente quando há poucas amostras por classe.
Pipeline (apenas no split train)
Configurado via Albumentations, aproximadamente:

Resize para image_size × image_size.
HorizontalFlip (p≈0.5):
Simula pequenas variações na orientação da imagem.
No contexto de ECG, é preciso cuidado para não introduzir transformações que “invertam o tempo” de forma clinicamente sem sentido, mas um flip horizontal suave em imagens padronizadas de traçado costuma ser tolerável, desde que se valide clinicamente.
RandomBrightnessContrast (p≈0.5):
Pequenas variações de brilho e contraste (±0.1).
Representa diferenças de digitalização, monitores, impressões, etc.
GaussNoise (p≈0.3):
Ruído gaussiano leve para robustez a ruído de aquisição.
Normalize (sempre):
Aplica normalize_mean e normalize_std definidos no YAML.
Splits val e test
Para val e test, usamos um pipeline mais simples:

Resize(image_size, image_size)
Normalize(mean, std)
Sem flips ou perturbações de brilho/ruído, garantindo que avaliação e teste reflitam melhor o comportamento do modelo em dados não manipulados.

Riscos, vieses e qualidade dos dados
Riscos de viés
Origem única do dataset:
Se os dados vêm de um único hospital ou coorte, o modelo pode capturar padrões específicos de equipamento (marca do ECG, configuração de papel/ruído) em vez de padrões fisiológicos generalizáveis.
Distribuição demográfica:
Idade, sexo, comorbidades e outras variáveis clínicas podem estar desbalanceadas entre classes, levando o modelo a aprender atalhos espúrios.
Protocolo de aquisição:
Diferentes frequências de amostragem, filtros ou layouts de impressão podem tornar o modelo sensível a detalhes não clínicos.
Qualidade dos dados
Artefatos de aquisição:
Cabos soltos, interferência elétrica, cortes de sinal, etc., podem aparecer nas imagens.
Se esses artefatos se correlacionarem com certas labels (por exemplo, exames “difíceis” marcados como anormais), o modelo pode capturar esses artefatos como sinal de doença.
Rotulagem imperfeita:
Erros de anotação (por exemplo, laudos ambíguos ou divergentes entre especialistas) introduzem ruído nas labels.
Desbalanceamento de classes
É comum em datasets médicos que a classe normal seja muito mais frequente que abnormal ou vice-versa (por exemplo, em conjuntos enriquecidos com casos patológicos).
Desbalanceamento pode levar a:
Alta acurácia aparente favorecendo sempre a classe majoritária.
Baixa sensibilidade para a classe minoritária (pior caso: a classe clinicamente mais relevante).
Mitigações possíveis (para fases futuras):

Reamostragem (oversampling da minoritária, undersampling da majoritária).
Ponderação na loss (por exemplo, CrossEntropyLoss(weight=...)).
Métricas adicionais (F1, AUC, sensitividade/especificidade por classe).
Contagens por classe e split
No notebook de pré-processamento, após criar o manifest.csv, fazemos verificações de sanidade.

Código usado
python
print("=== Checklist de qualidade ===")
print("Total de imagens:", len(manifest))
print()

print("Por split:")
print(manifest["split"].value_counts())
print()

print("Por classe:")
print(manifest["label"].value_counts())
print()

print("Por split e classe:")
print(manifest.groupby(["split", "label"]).size())
Interpretação esperada (exemplo ilustrativo)
Suponha que, após rodar o notebook, a saída seja algo como:

text
=== Checklist de qualidade ===
Total de imagens: 5000

Por split:
train    3500
val       750
test      750
Name: split, dtype: int64

Por classe:
normal      3000
abnormal    2000
Name: label, dtype: int64

Por split e classe:
split  label
test   abnormal     300
       normal       450
train  abnormal    1400
       normal      2100
val    abnormal     300
       normal       450
dtype: int64
Esse tipo de tabela permite verificar que:

Os splits respeitam aproximadamente as proporções configuradas (70/15/15).
A proporção de classes é preservada em cada split (estratificação correta).
Não há splits vazios nem classes ausentes em val ou test.
Se os números reais divergirem muito (por exemplo, alguma classe com pouquíssimos exemplos em test), isso deve ser documentado no relatório e considerado na interpretação das métricas.

Resumo
O dataset de ECG em imagem é organizado por subpastas de classe, e um manifest CSV centraliza filepath, label e split.
As imagens são redimensionadas para 224×224 e normalizadas com estatísticas ImageNet, visando compatibilidade com modelos pré-treinados.
Augmentations sutis (flip, brilho/contraste, ruído) são aplicadas somente no treino, preservando a integridade dos dados de validação e teste.
Há riscos importantes de viés (origem única, demografia, protocolo de aquisição) e de desbalanceamento de classes, que devem ser avaliados e mitigados nas próximas fases.
As contagens por classe e split (impressas a partir do manifest) são uma checagem fundamental de sanidade antes de confiar nos resultados de treino e avaliação.