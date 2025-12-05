# Dados de ECG

Esta pasta deve conter (ou apontar para) os datasets utilizados pelo Vision Assistant.

- Para ECG em imagem (PNG/JPG) usar estrutura por classe:

```text
DATA_DIR/
  train/
    classe_0/ *.png
    classe_1/ *.png
  val/        (opcional)
  test/       (opcional)
```

Consulte o arquivo principal `README.md` em `apps/vision-assistant` para detalhes de configuração via `configs/ecg.yaml`.
