import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
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
            nn.Tanh(),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z)
        x = x.view(z.size(0), -1, 4, 4)
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, image_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, base_channels, 4, 2, 1),  # 64x64
            nn.LeakyReLU(0.2, inplace=True),
            self._block(base_channels, base_channels * 2),  # 32x32
            self._block(base_channels * 2, base_channels * 4),  # 16x16
            self._block(base_channels * 4, base_channels * 8),  # 8x8
            self._block(base_channels * 8, base_channels * 8),  # 4x4
        )
        self.head = nn.Linear(base_channels * 8 * 4 * 4, 1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        # No BatchNorm in the critic; this keeps the GP objective well-defined.
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = x.view(images.size(0), -1)
        return self.head(x).view(-1)


def weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
