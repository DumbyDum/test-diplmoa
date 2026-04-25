from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

import config as c


def to_rgb(image: Image.Image) -> Image.Image:
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


@dataclass(slots=True)
class DatasetPaths:
    train_root: Path
    val_root: Path
    train_glob: str = "*.png"
    val_glob: str = "*.png"


class HinetDataset(Dataset):
    def __init__(self, file_paths: list[Path], transform=None):
        self.file_paths = file_paths
        self.transform = transform or T.ToTensor()

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int):
        path = self.file_paths[index]
        with Image.open(path) as handle:
            image = to_rgb(handle)
        return self.transform(image)


DEFAULT_TRAIN_TRANSFORM = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomCrop(c.cropsize),
        T.ToTensor(),
    ]
)

DEFAULT_VAL_TRANSFORM = T.Compose([T.ToTensor()])


def discover_dataset_paths() -> DatasetPaths:
    return DatasetPaths(
        train_root=Path(c.TRAIN_PATH),
        val_root=Path(c.VAL_PATH),
        train_glob=f"*.{c.format_train}",
        val_glob=f"*.{c.format_val}",
    )


def collect_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.glob(pattern) if path.is_file())


def build_dataloaders(
    train_transform=DEFAULT_TRAIN_TRANSFORM,
    val_transform=DEFAULT_VAL_TRANSFORM,
    batch_size: int = c.batch_size,
    val_batch_size: int = c.batchsize_val,
) -> tuple[DataLoader | None, DataLoader | None]:
    paths = discover_dataset_paths()
    train_files = collect_files(paths.train_root, paths.train_glob)
    val_files = collect_files(paths.val_root, paths.val_glob)

    train_loader = None
    if train_files:
        train_loader = DataLoader(
            HinetDataset(train_files, transform=train_transform),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            drop_last=True,
        )

    val_loader = None
    if val_files:
        val_loader = DataLoader(
            HinetDataset(val_files, transform=val_transform),
            batch_size=val_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            drop_last=False,
        )

    return train_loader, val_loader


trainloader, testloader = build_dataloaders()
