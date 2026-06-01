from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms


class DFNDataset(Dataset):
    """Load single-channel DFN PNG images and normalize them to [-1, 1]."""

    def __init__(
        self,
        image_dir: str | Path,
        image_size: int = 128,
        transform: Callable | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.paths = sorted(self.image_dir.glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"No PNG images found in {self.image_dir}")

        self.transform = transform or transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image)


def create_train_val_split(
    image_dir: str | Path,
    image_size: int = 128,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    dataset = DFNDataset(image_dir=image_dir, image_size=image_size)
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)
