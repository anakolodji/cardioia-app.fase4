import os
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split


def build_manifest(
    base_dir: str,
    raw_subdir: str = "ecg_raw",
    output_csv: str = "manifest.csv",
    test_size: float = 0.3,
    val_size: float = 0.5,
    random_state: int = 42,
) -> str:
    """Gera um manifest.csv a partir da estrutura em ecg_raw.

    Espera uma pasta:

        base_dir/
          ecg_raw/
            <folder_1>/*.jpg
            <folder_2>/*.jpg
            ...

    onde os nomes das pastas originais são mapeados para labels em português
    pelo dicionário FOLDER_TO_LABEL abaixo.
    """

    data_dir = base_dir
    raw_dir = os.path.join(data_dir, raw_subdir)

    folder_to_label: Dict[str, str] = {
        "ECG Images of Myocardial Infarction Patients": "infarto_mi",
        "ECG Images of Patient that have abnormal heartbeat": "batimento_anormal",
        "ECG Images of Patient that have History of MI": "historico_infarto",
        "Normal Person ECG Images (284x12=3408)": "normal",
    }

    exts = {".jpg", ".jpeg", ".png"}

    rows = []
    for folder, label in folder_to_label.items():
        folder_path = os.path.join(raw_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"[AVISO] Pasta não encontrada: {folder_path}")
            continue

        for fname in os.listdir(folder_path):
            _, ext = os.path.splitext(fname)
            if ext.lower() not in exts:
                continue
            fpath = os.path.join(folder_path, fname)
            rows.append({"filepath": fpath, "label": label})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Nenhuma imagem encontrada em {raw_dir}.")

    # Split estratificado: primeiro separa train e temp (val+test).
    train_df, temp_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state
    )

    # Depois separa temp em val e test.
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_size,
        stratify=temp_df["label"],
        random_state=random_state,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    out_path = os.path.join(data_dir, output_csv)
    full_df.to_csv(out_path, index=False)

    print(f"Manifest salvo em: {out_path}")
    print("Contagem por split:")
    print(full_df["split"].value_counts())
    print("\nContagem por label:")
    print(full_df["label"].value_counts())

    return out_path


def main() -> None:
    # base_dir deve ser .../apps/vision-assistant/data
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),  # .../apps/vision-assistant
        "data",
    )

    build_manifest(base_dir=base_dir)


if __name__ == "__main__":
    main()
