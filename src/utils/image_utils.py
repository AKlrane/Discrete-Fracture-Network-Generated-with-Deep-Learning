from pathlib import Path

import torch
from torchvision.utils import save_image


def _to_probability(images: torch.Tensor) -> torch.Tensor:
    return ((images.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)


def save_image_grid(
    images: torch.Tensor,
    out_path: str | Path,
    nrow: int = 8,
    threshold: float = 0.0,
) -> tuple[Path, Path]:
    """Save probability and thresholded binary grids from tanh-range images."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    probability = _to_probability(images)
    binary = (images.detach().cpu() > threshold).float()

    probability_path = out_path.with_name(f"{out_path.stem}_prob{out_path.suffix}")
    binary_path = out_path.with_name(f"{out_path.stem}_binary{out_path.suffix}")
    save_image(probability, probability_path, nrow=nrow, normalize=False)
    save_image(binary, binary_path, nrow=nrow, normalize=False)
    return probability_path, binary_path
