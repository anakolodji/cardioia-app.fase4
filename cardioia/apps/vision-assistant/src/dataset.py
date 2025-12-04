import os
from typing import Callable, Dict, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ECGImageDataset(Dataset):
    """Dataset para ECG em formato de imagem (PNG/JPG) organizado em pastas por classe.

    Estrutura esperada:
        root/
          train/
            classe_0/ *.png
            classe_1/ *.png
          val/   (opcional)
          test/  (opcional)

    Este dataset implementa funcionalidade similar ao ImageFolder, mas de forma explícita
    para facilitar controle e logging das classes em outros módulos.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        if split not in {"train", "val", "test"}:
            raise ValueError(f"split deve ser 'train', 'val' ou 'test', recebido: {split}")

        self.root = os.path.join(root, split)
        self.transform = transform

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Diretório de split não encontrado: {self.root}")

        # Descobre classes pelas subpastas em ordem alfabética estável
        classes = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        if not classes:
            raise RuntimeError(f"Nenhuma subpasta de classe encontrada em {self.root}")

        classes.sort()
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Indexa todos os arquivos de imagem
        samples = []
        extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        for class_name in self.classes:
            class_dir = os.path.join(self.root, class_name)
            for fname in os.listdir(class_dir):
                _, ext = os.path.splitext(fname)
                if ext.lower() not in extensions:
                    continue
                path = os.path.join(class_dir, fname)
                samples.append((path, self.class_to_idx[class_name]))

        if not samples:
            raise RuntimeError(f"Nenhuma imagem encontrada em {self.root} com extensões {extensions}")

        self.samples = samples

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:  # type: ignore[override]
        path, target = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def _build_transforms(image_size: int = 224, is_train: bool = True) -> Callable:
    """Cria transforms padrão para ECG em imagem.

    Para ECG em PNG, tratamos como imagem RGB normal. Augmentations leves no treino.
    """

    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def create_dataloaders_from_config(config: Dict) -> Dict[str, DataLoader]:
    """Cria DataLoaders (train/val/test) a partir de um dicionário de config.

    Exemplo de config esperado (ecg.yaml):

        data:
          root: "/caminho/para/ecg_dataset"
          image_size: 224
          batch_size: 32
          num_workers: 4
          pin_memory: true

    Retorna um dicionário com possíveis chaves "train", "val", "test".
    Só cria loaders para splits cujo diretório existir.
    """

    data_cfg = config.get("data", {})
    root = data_cfg.get("root")
    if root is None:
        raise ValueError("Config de dados deve conter a chave 'data.root'")

    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue

        is_train = split == "train"
        transform = _build_transforms(image_size=image_size, is_train=is_train)
        dataset = ECGImageDataset(root=root, split=split, transform=transform)

        shuffle = is_train
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        loaders[split] = loader

    if not loaders:
        raise RuntimeError(
            f"Nenhum DataLoader foi criado. Verifique se há pastas train/val/test dentro de {root}."
        )

    return loaders

