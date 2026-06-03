from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.models.wgan_gp import Generator
from src.utils.device import select_device

from .common import resolve_path


@dataclass
class GeneratedDFN:
    z: np.ndarray
    tanh: np.ndarray
    probability: np.ndarray
    binary: np.ndarray


class WGANLatentPrior:
    """Load a trained WGAN-GP generator and evaluate DFN images from latent z."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        latent_dim: int,
        base_channels: int = 64,
        image_size: int = 128,
        threshold: float = 0.0,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = resolve_path(checkpoint_path)
        self.latent_dim = int(latent_dim)
        self.base_channels = int(base_channels)
        self.image_size = int(image_size)
        self.threshold = float(threshold)
        self.device = select_device(device)

        self.generator = Generator(
            latent_dim=self.latent_dim,
            base_channels=self.base_channels,
        ).to(self.device)
        self._load_checkpoint()
        self.generator.eval()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WGANLatentPrior":
        return cls(
            checkpoint_path=config["checkpoint"],
            latent_dim=int(config["latent_dim"]),
            base_channels=int(config.get("base_channels", 64)),
            image_size=int(config.get("image_size", 128)),
            threshold=float(config.get("threshold", 0.0)),
            device=str(config.get("device", "auto")),
        )

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"WGAN-GP checkpoint not found: {self.checkpoint_path}. "
                "Train the 16D conditional prior first."
            )
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "generator" not in checkpoint:
            raise KeyError(f"Expected checkpoint with a 'generator' state_dict: {self.checkpoint_path}")
        self.generator.load_state_dict(checkpoint["generator"])

    def _as_latent_tensor(self, z: np.ndarray | torch.Tensor) -> torch.Tensor:
        z_tensor = torch.as_tensor(z, dtype=torch.float32)
        if z_tensor.ndim == 1:
            z_tensor = z_tensor.unsqueeze(0)
        if z_tensor.ndim != 2 or z_tensor.size(1) != self.latent_dim:
            raise ValueError(f"Expected latent shape (N, {self.latent_dim}), got {tuple(z_tensor.shape)}")
        return z_tensor.to(self.device)

    def generate_tanh(self, z: np.ndarray | torch.Tensor) -> torch.Tensor:
        z_tensor = self._as_latent_tensor(z)
        with torch.no_grad():
            return self.generator(z_tensor).detach().cpu()

    def generate(self, z: np.ndarray | torch.Tensor) -> GeneratedDFN:
        images = self.generate_tanh(z)
        tanh = images.numpy()
        probability = np.clip((tanh + 1.0) / 2.0, 0.0, 1.0)
        binary = (tanh > self.threshold).astype(np.uint8)
        z_array = self._as_latent_tensor(z).detach().cpu().numpy()
        return GeneratedDFN(
            z=z_array,
            tanh=tanh[:, 0],
            probability=probability[:, 0],
            binary=binary[:, 0],
        )

    def generate_one(self, z: np.ndarray | torch.Tensor) -> GeneratedDFN:
        generated = self.generate(z)
        return GeneratedDFN(
            z=generated.z[0],
            tanh=generated.tanh[0],
            probability=generated.probability[0],
            binary=generated.binary[0],
        )


def save_probability_image(array: np.ndarray, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((np.clip(array, 0.0, 1.0) * 255).astype(np.uint8))
    image.save(out_path)
    return out_path


def save_binary_image(array: np.ndarray, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(((array > 0).astype(np.uint8) * 255))
    image.save(out_path)
    return out_path
