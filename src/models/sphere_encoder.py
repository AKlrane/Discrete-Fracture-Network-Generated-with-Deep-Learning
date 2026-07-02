import math

import torch
from torch import nn


def spherify(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Project each latent vector onto the RMS sphere with radius sqrt(latent_dim)."""
    if z.ndim != 2:
        raise ValueError("z must have shape [batch_size, latent_dim]")
    rms = torch.sqrt(z.square().mean(dim=1, keepdim=True) + eps)
    return z / rms


class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        image_channels: int = 1,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, base_channels, 4, 2, 1),  # 64x64
            nn.LeakyReLU(0.2, inplace=True),
            self._block(base_channels, base_channels * 2),  # 32x32
            self._block(base_channels * 2, base_channels * 4),  # 16x16
            self._block(base_channels * 4, base_channels * 8),  # 8x8
            self._block(base_channels * 8, base_channels * 8),  # 4x4
        )
        self.head = nn.Linear(base_channels * 8 * 4 * 4, latent_dim)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = x.view(images.size(0), -1)
        return self.head(x)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        image_channels: int = 1,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.project = nn.Sequential(
            nn.Linear(latent_dim, base_channels * 8 * 4 * 4),
            nn.BatchNorm1d(base_channels * 8 * 4 * 4),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            self._block(base_channels * 8, base_channels * 8),  # 8x8
            self._block(base_channels * 8, base_channels * 4),  # 16x16
            self._block(base_channels * 4, base_channels * 2),  # 32x32
            self._block(base_channels * 2, base_channels),  # 64x64
            nn.ConvTranspose2d(base_channels, image_channels, 4, 2, 1),  # 128x128
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward_logits(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z)
        x = x.view(z.size(0), -1, 4, 4)
        return self.net(x)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(z))


class SphereEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        image_channels: int = 1,
        base_channels: int = 64,
        noise_angle_degrees: float = 80.0,
        spherify_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if not 0.0 <= noise_angle_degrees < 90.0:
            raise ValueError("noise_angle_degrees must be in [0, 90)")

        self.latent_dim = latent_dim
        self.noise_angle_degrees = float(noise_angle_degrees)
        self.spherify_eps = float(spherify_eps)
        self.encoder = Encoder(
            latent_dim=latent_dim,
            image_channels=image_channels,
            base_channels=base_channels,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            image_channels=image_channels,
            base_channels=base_channels,
        )

    @property
    def sigma_max(self) -> float:
        return math.tan(math.radians(self.noise_angle_degrees))

    def spherify(self, z: torch.Tensor) -> torch.Tensor:
        return spherify(z, eps=self.spherify_eps)

    def encode_raw(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def encode_sphere(self, images: torch.Tensor) -> torch.Tensor:
        return self.spherify(self.encode_raw(images))

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder.forward_logits(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @staticmethod
    def stratified_unit_radii(
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if batch_size == 1:
            return torch.rand(1, 1, device=device, dtype=dtype)
        values = (torch.arange(batch_size, device=device, dtype=torch.float32) + torch.rand(batch_size, device=device)) / batch_size
        values = values[torch.randperm(batch_size, device=device)]
        return values.reshape(batch_size, 1).to(dtype=dtype)

    def noisy_latents(
        self,
        clean_latent: torch.Tensor,
        sub_noise_max_scale: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sub_noise_max_scale < 0.0:
            raise ValueError("sub_noise_max_scale must be non-negative")
        noise = torch.randn_like(clean_latent)
        radius = self.stratified_unit_radii(
            clean_latent.size(0),
            device=clean_latent.device,
            dtype=clean_latent.dtype,
        )
        sub_scale = torch.rand_like(radius).mul_(sub_noise_max_scale)
        sigma = clean_latent.new_tensor(self.sigma_max)
        small_noise_latent = self.spherify(clean_latent + sub_scale * radius * sigma * noise)
        large_noise_latent = self.spherify(clean_latent + radius * sigma * noise)
        return small_noise_latent, large_noise_latent

    def forward(
        self,
        images: torch.Tensor,
        sub_noise_max_scale: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        clean_latent = self.encode_sphere(images)
        small_noise_latent, large_noise_latent = self.noisy_latents(
            clean_latent,
            sub_noise_max_scale=sub_noise_max_scale,
        )
        reconstruction_logits = self.decode_logits(small_noise_latent)
        large_noise_logits = self.decode_logits(large_noise_latent)
        return {
            "clean_latent": clean_latent,
            "small_noise_latent": small_noise_latent,
            "large_noise_latent": large_noise_latent,
            "reconstruction_logits": reconstruction_logits,
            "reconstruction_probability": torch.sigmoid(reconstruction_logits),
            "large_noise_logits": large_noise_logits,
            "large_noise_probability": torch.sigmoid(large_noise_logits),
        }

    @torch.no_grad()
    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode_sphere(images))

    @torch.no_grad()
    def generate(
        self,
        initial_noise: torch.Tensor,
        steps: int = 1,
    ) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        z = self.spherify(initial_noise)
        probability = self.decode(z)
        for _ in range(steps - 1):
            tanh_range = probability.mul(2.0).sub(1.0)
            z = self.encode_sphere(tanh_range)
            noise = torch.randn_like(z)
            z = self.spherify(z + self.sigma_max * noise)
            probability = self.decode(z)
        return probability


def weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        nn.init.xavier_uniform_(module.weight.data)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
