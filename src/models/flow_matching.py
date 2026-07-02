import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _num_groups(channels: int, requested_groups: int) -> int:
    groups = min(requested_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            return timesteps[:, None]

        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=timesteps.dtype,
        )
        exponent = exponent / max(half_dim - 1, 1)
        frequencies = torch.exp(exponent)
        embedding = timesteps[:, None] * frequencies[None, :]
        embedding = torch.cat([embedding.sin(), embedding.cos()], dim=1)
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels, groups), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels),
        )
        self.norm2 = nn.GroupNorm(_num_groups(out_channels, groups), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        time_bias = self.time_projection(time_embedding)[:, :, None, None]
        x = x + time_bias
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class LegacyTimeConditionedUNet(nn.Module):
    """Legacy UNet velocity field for pixel-space Flow Matching."""

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 32,
        channel_multipliers: Sequence[int] = (1, 2, 4, 8),
        time_embedding_dim: int | None = None,
        groups: int = 8,
    ) -> None:
        super().__init__()
        if not channel_multipliers:
            raise ValueError("channel_multipliers must contain at least one value")

        self.image_channels = image_channels
        self.base_channels = base_channels
        self.channel_multipliers = tuple(int(multiplier) for multiplier in channel_multipliers)
        channels = [base_channels * multiplier for multiplier in self.channel_multipliers]
        time_dim = time_embedding_dim or base_channels * 4

        self.input_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_channels = base_channels
        for index, out_channels in enumerate(channels):
            self.down_blocks.append(
                ResidualBlock(in_channels, out_channels, time_dim, groups=groups)
            )
            in_channels = out_channels
            if index != len(channels) - 1:
                self.downsamples.append(Downsample(in_channels))

        self.middle_blocks = nn.ModuleList(
            [
                ResidualBlock(in_channels, in_channels, time_dim, groups=groups),
                ResidualBlock(in_channels, in_channels, time_dim, groups=groups),
            ]
        )

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for index, skip_channels in enumerate(reversed(channels)):
            self.up_blocks.append(
                ResidualBlock(
                    in_channels + skip_channels,
                    skip_channels,
                    time_dim,
                    groups=groups,
                )
            )
            in_channels = skip_channels
            if index != len(channels) - 1:
                self.upsamples.append(Upsample(in_channels))

        self.output_norm = nn.GroupNorm(_num_groups(in_channels, groups), in_channels)
        self.output_conv = nn.Conv2d(in_channels, image_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape [batch_size]")
        if timesteps.size(0) != x.size(0):
            raise ValueError("timesteps batch size must match x batch size")

        time_embedding = self.time_embedding(timesteps)
        x = self.input_conv(x)
        skips: list[torch.Tensor] = []

        for index, block in enumerate(self.down_blocks):
            x = block(x, time_embedding)
            skips.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)

        for block in self.middle_blocks:
            x = block(x, time_embedding)

        for index, block in enumerate(self.up_blocks):
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_embedding)
            if index < len(self.upsamples):
                x = self.upsamples[index](x)

        return self.output_conv(F.silu(self.output_norm(x)))


# Backward-compatible name for existing imports and checkpoints.
TimeConditionedUNet = LegacyTimeConditionedUNet


def weights_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
