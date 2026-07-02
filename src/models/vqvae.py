import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_residual_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, hidden_channels // 2, 4, 2, 1),  # 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, 2, 1),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),  # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            *[ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)],
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, embedding_dim, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class Decoder(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_residual_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels, 3, padding=1),
            *[ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)],
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1),  # 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1),  # 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 2, image_channels, 4, 2, 1),  # 128x128
        )

    def forward_logits(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.net(quantized)

    def forward(self, quantized: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.forward_logits(quantized))


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_channels_last = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_channels_last.view(-1, self.embedding_dim)

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        encoding_indices = distances.argmin(dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_z.dtype)
        quantized_flat = encodings @ self.embedding.weight
        quantized = quantized_flat.view_as(z_channels_last)

        codebook_loss = F.mse_loss(quantized, z_channels_last.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z_channels_last)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized = z_channels_last + (quantized - z_channels_last).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        average_probs = encodings.mean(dim=0)
        perplexity = torch.exp(
            -torch.sum(average_probs * torch.log(average_probs + 1e-10))
        )
        indices = encoding_indices.view(z.size(0), z.size(2), z.size(3))
        return quantized, vq_loss, perplexity, indices


class VQVAE(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        num_residual_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.encoder = Encoder(
            image_channels=image_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            num_residual_blocks=num_residual_blocks,
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = Decoder(
            image_channels=image_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            num_residual_blocks=num_residual_blocks,
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(images)
        _, _, _, indices = self.quantizer(z)
        return z, indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.embedding(indices)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized)

    def forward(
        self,
        images: torch.Tensor,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        z = self.encoder(images)
        quantized, vq_loss, perplexity, indices = self.quantizer(z)
        decoder_logits = self.decoder.forward_logits(quantized)
        reconstructed = torch.tanh(decoder_logits)
        if return_logits:
            return reconstructed, decoder_logits, vq_loss, perplexity, indices
        return reconstructed, vq_loss, perplexity, indices


def weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
