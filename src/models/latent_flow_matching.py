import math

import torch
import torch.nn.functional as F
from torch import nn


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


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        time_embedding_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x)
        hidden = hidden + self.time_projection(time_embedding)
        return x + self.net(hidden)


class LatentFlowMLP(nn.Module):
    """Velocity field for Rectified Flow in a compact vector latent space."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        time_embedding_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.time_embedding_dim = time_embedding_dim

        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    hidden_dim=hidden_dim,
                    time_embedding_dim=time_embedding_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError("z must have shape [batch_size, latent_dim]")
        if z.size(1) != self.latent_dim:
            raise ValueError("z latent dimension does not match model latent_dim")
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape [batch_size]")
        if timesteps.size(0) != z.size(0):
            raise ValueError("timesteps batch size must match z batch size")

        time_embedding = self.time_embedding(timesteps)
        hidden = self.input_projection(z)
        for block in self.blocks:
            hidden = block(hidden, time_embedding)
        return self.output_projection(hidden)


def weights_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
