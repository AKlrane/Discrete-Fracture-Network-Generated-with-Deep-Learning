import torch
from torch import nn


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
        flattened_dim = base_channels * 8 * 4 * 4
        self.mu = nn.Linear(flattened_dim, latent_dim)
        self.logvar = nn.Linear(flattened_dim, latent_dim)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(images)
        x = x.view(images.size(0), -1)
        return self.mu(x), self.logvar(x)


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


class BetaVAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        image_channels: int = 1,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
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

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(images)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder.forward_logits(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(images)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        probability = torch.sigmoid(logits)
        return probability, logits, mu, logvar, z


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
