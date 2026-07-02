import argparse
import csv
import math
import sys
from itertools import chain
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dfn_dataset import DFNDataset
from src.models.wae import Decoder, Encoder, LatentDiscriminator, weights_init
from src.utils.device import select_device
from src.utils.image_utils import save_image_grid
from src.utils.seed import set_seed


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def append_log(log_path: Path, row: dict[str, float | int | str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x**2).sum(dim=1, keepdim=True)
    y_norm = (y**2).sum(dim=1, keepdim=True).t()
    distance = x_norm + y_norm - 2.0 * x @ y.t()
    return distance.clamp_min(0.0)


def imq_kernel(x: torch.Tensor, y: torch.Tensor, scales: list[float]) -> torch.Tensor:
    distance = squared_distance(x, y)
    latent_dim = x.size(1)
    kernel = torch.zeros_like(distance)
    for scale in scales:
        constant = 2.0 * latent_dim * float(scale)
        kernel = kernel + constant / (constant + distance)
    return kernel


def mmd_imq(
    encoded: torch.Tensor,
    prior: torch.Tensor,
    scales: list[float],
    estimator: str = "biased",
) -> torch.Tensor:
    estimator = estimator.lower()
    if estimator in {"biased", "v_statistic", "v-statistic"}:
        k_encoded = imq_kernel(encoded, encoded, scales)
        k_prior = imq_kernel(prior, prior, scales)
        k_cross = imq_kernel(encoded, prior, scales)
        return k_encoded.mean() + k_prior.mean() - 2.0 * k_cross.mean()
    if estimator not in {"unbiased", "u_statistic", "u-statistic"}:
        raise ValueError("regularizer.mmd_estimator must be either 'biased' or 'unbiased'")

    batch_size = encoded.size(0)
    k_encoded = imq_kernel(encoded, encoded, scales)
    k_prior = imq_kernel(prior, prior, scales)
    k_cross = imq_kernel(encoded, prior, scales)

    if batch_size > 1:
        normalizer = batch_size * (batch_size - 1)
        encoded_term = (k_encoded.sum() - k_encoded.diag().sum()) / normalizer
        prior_term = (k_prior.sum() - k_prior.diag().sum()) / normalizer
    else:
        encoded_term = k_encoded.mean()
        prior_term = k_prior.mean()
    return encoded_term + prior_term - 2.0 * k_cross.mean()


def regularization_weight(
    regularizer_cfg: dict[str, Any],
    weight_key: str,
    warmup_key: str,
    step: int,
    default: float,
) -> float:
    max_weight = float(regularizer_cfg.get(weight_key, default))
    warmup_steps = int(regularizer_cfg.get(warmup_key, 0))
    if warmup_steps <= 0:
        return max_weight
    progress = min(max(step, 0) / warmup_steps, 1.0)
    return max_weight * progress


def mmd_regularization(
    encoded: torch.Tensor,
    prior: torch.Tensor,
    scales: list[float],
    regularizer_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    estimator = str(regularizer_cfg.get("mmd_estimator", "biased")).lower()
    penalty_type = str(regularizer_cfg.get("mmd_penalty", "raw")).lower()
    raw_mmd = mmd_imq(encoded, prior, scales, estimator=estimator)
    baseline = torch.tensor(0.0, device=encoded.device, dtype=encoded.dtype)

    if penalty_type == "raw":
        return raw_mmd, raw_mmd, baseline
    if penalty_type in {"clamp", "clamped", "clamp_min_zero"}:
        return raw_mmd, raw_mmd.clamp_min(0.0), baseline
    if penalty_type in {"prior_baseline_corrected", "baseline_corrected", "corrected_biased"}:
        num_baseline_batches = int(regularizer_cfg.get("mmd_baseline_batches", 1))
        if num_baseline_batches < 1:
            raise ValueError("regularizer.mmd_baseline_batches must be >= 1")
        baseline_values = []
        for _ in range(num_baseline_batches):
            prior_reference = torch.randn_like(prior)
            baseline_values.append(
                mmd_imq(prior, prior_reference, scales, estimator=estimator).detach()
            )
        baseline = torch.stack(baseline_values).mean()
        return raw_mmd, (raw_mmd - baseline).clamp_min(0.0), baseline
    raise ValueError(
        "regularizer.mmd_penalty must be one of: raw, clamp_min_zero, "
        "prior_baseline_corrected"
    )


def _tanh_range_to_probability(images: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return ((images + 1.0) / 2.0).clamp(eps, 1.0 - eps)


def dice_loss(
    predicted_probability: torch.Tensor,
    target_probability: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    predicted_flat = predicted_probability.flatten(start_dim=1)
    target_flat = target_probability.flatten(start_dim=1)
    intersection = (predicted_flat * target_flat).sum(dim=1)
    denominator = predicted_flat.sum(dim=1) + target_flat.sum(dim=1)
    score = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - score.mean()


def sobel_edge_magnitude(images: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    sobel_x = images.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(1) / 4.0
    sobel_y = images.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    ).unsqueeze(1) / 4.0
    channels = images.size(1)
    sobel_x = sobel_x.expand(channels, 1, 3, 3)
    sobel_y = sobel_y.expand(channels, 1, 3, 3)
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    gradient_x = F.conv2d(padded, sobel_x, groups=channels)
    gradient_y = F.conv2d(padded, sobel_y, groups=channels)
    return torch.sqrt(gradient_x.square() + gradient_y.square() + eps)


def edge_alignment_loss(
    predicted_probability: torch.Tensor,
    target_probability: torch.Tensor,
) -> torch.Tensor:
    predicted_edges = sobel_edge_magnitude(predicted_probability)
    target_edges = sobel_edge_magnitude(target_probability)
    return F.l1_loss(predicted_edges, target_edges)


def multiscale_dice_loss(
    predicted_probability: torch.Tensor,
    target_probability: torch.Tensor,
    scales: list[int],
    smooth: float = 1.0,
) -> torch.Tensor:
    losses = []
    height, width = predicted_probability.shape[-2:]
    for scale in scales:
        scale = int(scale)
        if scale <= 1 or height < scale or width < scale:
            continue
        predicted_downsampled = F.avg_pool2d(
            predicted_probability,
            kernel_size=scale,
            stride=scale,
        )
        target_downsampled = F.avg_pool2d(
            target_probability,
            kernel_size=scale,
            stride=scale,
        )
        losses.append(
            dice_loss(
                predicted_downsampled,
                target_downsampled,
                smooth=smooth,
            )
        )
    if not losses:
        return predicted_probability.new_tensor(0.0)
    return torch.stack(losses).mean()


def foreground_ratio_loss(
    predicted_probability: torch.Tensor,
    target_probability: torch.Tensor,
    *,
    mode: str = "l1",
    margin: float = 0.0,
) -> torch.Tensor:
    predicted_ratio = predicted_probability.mean(dim=(1, 2, 3))
    target_ratio = target_probability.mean(dim=(1, 2, 3))
    difference = predicted_ratio - target_ratio

    if margin < 0.0:
        raise ValueError("regularizer.foreground_ratio_margin must be non-negative")
    if mode == "l1":
        return F.l1_loss(predicted_ratio, target_ratio)
    if mode == "over_density":
        return F.relu(difference - margin).mean()
    if mode == "under_density":
        return F.relu(-difference - margin).mean()
    raise ValueError(
        "regularizer.foreground_ratio_mode must be one of: "
        "l1, over_density, under_density"
    )


def wae_reconstruction_loss_and_metrics(
    reconstructed: torch.Tensor,
    real_images: torch.Tensor,
    regularizer_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_type = str(regularizer_cfg.get("reconstruction_loss", "l1")).lower()
    if loss_type == "l1":
        l1 = F.l1_loss(reconstructed, real_images)
        nan = l1.new_tensor(float("nan"))
        return l1, {
            "base_reconstruction_loss": l1,
            "bce_loss": nan,
            "dice_loss": nan,
            "edge_loss": nan,
            "weighted_edge_loss": nan,
            "multiscale_dice_loss": nan,
            "weighted_multiscale_dice_loss": nan,
            "foreground_ratio_loss": nan,
            "weighted_foreground_ratio_loss": nan,
        }
    if loss_type not in {"bce_dice", "bce_dice_edge", "bce_dice_structural"}:
        raise ValueError(
            "regularizer.reconstruction_loss must be one of: l1, bce_dice, "
            "bce_dice_edge, bce_dice_structural"
        )

    bce_weight = float(regularizer_cfg.get("bce_weight", 0.5))
    if not 0.0 <= bce_weight <= 1.0:
        raise ValueError("regularizer.bce_weight must be in [0, 1]")
    edge_weight = float(regularizer_cfg.get("edge_weight", 0.0))
    if edge_weight < 0.0:
        raise ValueError("regularizer.edge_weight must be non-negative")
    multiscale_dice_weight = float(regularizer_cfg.get("multiscale_dice_weight", 0.0))
    if multiscale_dice_weight < 0.0:
        raise ValueError("regularizer.multiscale_dice_weight must be non-negative")
    foreground_ratio_weight = float(regularizer_cfg.get("foreground_ratio_weight", 0.0))
    if foreground_ratio_weight < 0.0:
        raise ValueError("regularizer.foreground_ratio_weight must be non-negative")
    foreground_ratio_mode = str(regularizer_cfg.get("foreground_ratio_mode", "l1")).lower()
    foreground_ratio_margin = float(regularizer_cfg.get("foreground_ratio_margin", 0.0))

    eps = float(regularizer_cfg.get("probability_eps", 1e-6))
    smooth = float(regularizer_cfg.get("dice_smooth", 1.0))
    predicted_probability = _tanh_range_to_probability(reconstructed, eps=eps)
    target_probability = _tanh_range_to_probability(real_images, eps=eps)
    bce = F.binary_cross_entropy(predicted_probability, target_probability)
    dice = dice_loss(predicted_probability, target_probability, smooth=smooth)
    base_loss = bce_weight * bce + (1.0 - bce_weight) * dice

    if loss_type in {"bce_dice_edge", "bce_dice_structural"} or edge_weight > 0.0:
        edge = edge_alignment_loss(predicted_probability, target_probability)
    else:
        edge = predicted_probability.new_tensor(0.0)

    if loss_type == "bce_dice_structural" or multiscale_dice_weight > 0.0:
        scales = [
            int(scale)
            for scale in regularizer_cfg.get("multiscale_dice_scales", [2, 4])
        ]
        multiscale_dice = multiscale_dice_loss(
            predicted_probability,
            target_probability,
            scales=scales,
            smooth=smooth,
        )
    else:
        multiscale_dice = predicted_probability.new_tensor(0.0)

    weighted_edge = edge_weight * edge
    weighted_multiscale_dice = multiscale_dice_weight * multiscale_dice
    foreground_ratio = foreground_ratio_loss(
        predicted_probability,
        target_probability,
        mode=foreground_ratio_mode,
        margin=foreground_ratio_margin,
    )
    weighted_foreground_ratio = foreground_ratio_weight * foreground_ratio
    total = (
        base_loss
        + weighted_edge
        + weighted_multiscale_dice
        + weighted_foreground_ratio
    )
    return total, {
        "base_reconstruction_loss": base_loss,
        "bce_loss": bce,
        "dice_loss": dice,
        "edge_loss": edge,
        "weighted_edge_loss": weighted_edge,
        "multiscale_dice_loss": multiscale_dice,
        "weighted_multiscale_dice_loss": weighted_multiscale_dice,
        "foreground_ratio_loss": foreground_ratio,
        "weighted_foreground_ratio_loss": weighted_foreground_ratio,
    }


def wae_reconstruction_loss(
    reconstructed: torch.Tensor,
    real_images: torch.Tensor,
    regularizer_cfg: dict[str, Any],
) -> torch.Tensor:
    return wae_reconstruction_loss_and_metrics(
        reconstructed,
        real_images,
        regularizer_cfg,
    )[0]


def save_wae_checkpoint(
    path: str | Path,
    encoder: Encoder,
    decoder: Decoder,
    optimizer_autoencoder: torch.optim.Optimizer,
    epoch: int,
    step: int,
    config: dict[str, Any],
    latent_discriminator: LatentDiscriminator | None = None,
    optimizer_discriminator: torch.optim.Optimizer | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer_autoencoder": optimizer_autoencoder.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
    }
    if latent_discriminator is not None:
        checkpoint["latent_discriminator"] = latent_discriminator.state_dict()
    if optimizer_discriminator is not None:
        checkpoint["optimizer_discriminator"] = optimizer_discriminator.state_dict()
    torch.save(checkpoint, path)


def load_wae_checkpoint(
    path: str | Path,
    encoder: Encoder,
    decoder: Decoder,
    optimizer_autoencoder: torch.optim.Optimizer,
    latent_discriminator: LatentDiscriminator | None = None,
    optimizer_discriminator: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, int]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    optimizer_autoencoder.load_state_dict(checkpoint["optimizer_autoencoder"])

    if latent_discriminator is not None:
        latent_discriminator.load_state_dict(checkpoint["latent_discriminator"])
    if optimizer_discriminator is not None and "optimizer_discriminator" in checkpoint:
        optimizer_discriminator.load_state_dict(checkpoint["optimizer_discriminator"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0))


def sample_decoder(
    encoder: Encoder,
    decoder: Decoder,
    fixed_noise: torch.Tensor,
    fixed_images: torch.Tensor,
    sample_dir: Path,
    global_step: int,
    num_sample_images: int,
) -> None:
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        samples = decoder(fixed_noise)
        reconstructed = decoder(encoder(fixed_images))
    nrow = int(math.sqrt(num_sample_images))
    save_image_grid(samples, sample_dir / f"step_{global_step:07d}.png", nrow=nrow)
    save_image_grid(
        reconstructed,
        sample_dir / f"step_{global_step:07d}_recon.png",
        nrow=nrow,
    )
    encoder.train()
    decoder.train()


def train(config: dict[str, Any], resume: str | Path | None = None, max_batches: int | None = None) -> None:
    training_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    regularizer_cfg = config["regularizer"]
    outputs_cfg = config["outputs"]

    regularizer_type = str(regularizer_cfg["type"]).lower()
    if regularizer_type not in {"mmd", "gan"}:
        raise ValueError("regularizer.type must be either 'mmd' or 'gan'")
    if max_batches is not None and max_batches < 1:
        raise ValueError("--max_batches must be a positive integer")

    set_seed(int(training_cfg["seed"]))
    device = select_device(str(training_cfg.get("device", "cuda")))

    dataset = DFNDataset(
        image_dir=resolve_path(data_cfg["image_dir"]),
        image_size=int(data_cfg["image_size"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    latent_dim = int(model_cfg["latent_dim"])
    base_channels = int(model_cfg["base_channels"])
    encoder = Encoder(latent_dim=latent_dim, base_channels=base_channels).to(device)
    decoder = Decoder(latent_dim=latent_dim, base_channels=base_channels).to(device)
    encoder.apply(weights_init)
    decoder.apply(weights_init)

    latent_discriminator: LatentDiscriminator | None = None
    optimizer_discriminator: torch.optim.Optimizer | None = None
    if regularizer_type == "gan":
        hidden_dim = int(regularizer_cfg.get("discriminator_hidden_dim", 256))
        latent_discriminator = LatentDiscriminator(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
        latent_discriminator.apply(weights_init)

    betas = (float(training_cfg["beta1"]), float(training_cfg["beta2"]))
    optimizer_autoencoder = torch.optim.Adam(
        chain(encoder.parameters(), decoder.parameters()),
        lr=float(training_cfg["lr"]),
        betas=betas,
    )
    if latent_discriminator is not None:
        optimizer_discriminator = torch.optim.Adam(
            latent_discriminator.parameters(),
            lr=float(regularizer_cfg.get("discriminator_lr", training_cfg["lr"])),
            betas=betas,
        )

    start_epoch = 0
    global_step = 0
    if resume is not None:
        start_epoch, global_step = load_wae_checkpoint(
            resume,
            encoder,
            decoder,
            optimizer_autoencoder,
            latent_discriminator=latent_discriminator,
            optimizer_discriminator=optimizer_discriminator,
            map_location=device,
        )
        start_epoch += 1

    sample_dir = resolve_path(outputs_cfg["sample_dir"])
    checkpoint_dir = resolve_path(outputs_cfg["checkpoint_dir"])
    log_dir = resolve_path(outputs_cfg["log_dir"])
    fixed_noise = torch.randn(int(training_cfg["num_sample_images"]), latent_dim, device=device)
    fixed_images = torch.stack(
        [dataset[index] for index in range(min(int(training_cfg["num_sample_images"]), len(dataset)))]
    ).to(device)
    sample_interval = int(training_cfg["sample_interval"])
    checkpoint_interval = int(training_cfg["checkpoint_interval"])
    lambda_recon = float(regularizer_cfg.get("lambda_recon", 1.0))
    lambda_adv = float(regularizer_cfg.get("lambda_adv", 1.0))
    imq_scales = [float(scale) for scale in regularizer_cfg.get("imq_scales", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])]
    discriminator_steps = int(regularizer_cfg.get("discriminator_steps", 1))
    batches_this_run = 0
    should_stop = False
    saved_current_step_sample = False
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, int(training_cfg["num_epochs"])):
        last_epoch = epoch
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{training_cfg['num_epochs']}")
        for real_images in progress:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            discriminator_loss = torch.tensor(float("nan"), device=device)
            encoded_score_mean = torch.tensor(float("nan"), device=device)
            prior_score_mean = torch.tensor(float("nan"), device=device)

            if regularizer_type == "gan":
                assert latent_discriminator is not None
                assert optimizer_discriminator is not None
                latent_discriminator.train()
                for _ in range(discriminator_steps):
                    with torch.no_grad():
                        encoded_detached = encoder(real_images).detach()
                    prior_z = torch.randn(batch_size, latent_dim, device=device)
                    encoded_logits = latent_discriminator(encoded_detached)
                    prior_logits = latent_discriminator(prior_z)
                    discriminator_loss = 0.5 * (
                        F.binary_cross_entropy_with_logits(prior_logits, torch.ones_like(prior_logits))
                        + F.binary_cross_entropy_with_logits(encoded_logits, torch.zeros_like(encoded_logits))
                    )
                    optimizer_discriminator.zero_grad(set_to_none=True)
                    discriminator_loss.backward()
                    optimizer_discriminator.step()
                    encoded_score_mean = encoded_logits.detach().mean()
                    prior_score_mean = prior_logits.detach().mean()

            encoded = encoder(real_images)
            reconstructed = decoder(encoded)
            reconstruction_loss, reconstruction_metrics = wae_reconstruction_loss_and_metrics(
                reconstructed,
                real_images,
                regularizer_cfg,
            )

            if regularizer_type == "mmd":
                prior_z = torch.randn_like(encoded)
                mmd_loss, mmd_penalty, mmd_baseline = mmd_regularization(
                    encoded,
                    prior_z,
                    imq_scales,
                    regularizer_cfg,
                )
                current_lambda_mmd = regularization_weight(
                    regularizer_cfg,
                    "lambda_mmd",
                    "lambda_mmd_warmup_steps",
                    global_step,
                    default=10.0,
                )
                weighted_mmd_loss = current_lambda_mmd * mmd_penalty
                adversarial_loss = torch.tensor(float("nan"), device=device)
                total_loss = lambda_recon * reconstruction_loss + weighted_mmd_loss
            else:
                assert latent_discriminator is not None
                mmd_loss = torch.tensor(float("nan"), device=device)
                mmd_penalty = torch.tensor(float("nan"), device=device)
                mmd_baseline = torch.tensor(float("nan"), device=device)
                current_lambda_mmd = float("nan")
                weighted_mmd_loss = torch.tensor(float("nan"), device=device)
                encoded_logits_for_autoencoder = latent_discriminator(encoded)
                adversarial_loss = F.binary_cross_entropy_with_logits(
                    encoded_logits_for_autoencoder,
                    torch.ones_like(encoded_logits_for_autoencoder),
                )
                total_loss = lambda_recon * reconstruction_loss + lambda_adv * adversarial_loss

            optimizer_autoencoder.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer_autoencoder.step()

            global_step += 1
            batches_this_run += 1
            saved_current_step_sample = False
            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "regularizer": regularizer_type,
                "total_loss": float(total_loss.detach().cpu()),
                "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
                "base_reconstruction_loss": float(
                    reconstruction_metrics["base_reconstruction_loss"].detach().cpu()
                ),
                "bce_loss": float(reconstruction_metrics["bce_loss"].detach().cpu()),
                "dice_loss": float(reconstruction_metrics["dice_loss"].detach().cpu()),
                "edge_loss": float(reconstruction_metrics["edge_loss"].detach().cpu()),
                "weighted_edge_loss": float(
                    reconstruction_metrics["weighted_edge_loss"].detach().cpu()
                ),
                "multiscale_dice_loss": float(
                    reconstruction_metrics["multiscale_dice_loss"].detach().cpu()
                ),
                "weighted_multiscale_dice_loss": float(
                    reconstruction_metrics["weighted_multiscale_dice_loss"].detach().cpu()
                ),
                "foreground_ratio_loss": float(
                    reconstruction_metrics["foreground_ratio_loss"].detach().cpu()
                ),
                "weighted_foreground_ratio_loss": float(
                    reconstruction_metrics["weighted_foreground_ratio_loss"].detach().cpu()
                ),
                "mmd_loss": float(mmd_loss.detach().cpu()),
                "mmd_penalty": float(mmd_penalty.detach().cpu()),
                "mmd_baseline": float(mmd_baseline.detach().cpu()),
                "mmd_weight": float(current_lambda_mmd),
                "weighted_mmd_loss": float(weighted_mmd_loss.detach().cpu()),
                "adversarial_loss": float(adversarial_loss.detach().cpu()),
                "discriminator_loss": float(discriminator_loss.detach().cpu()),
                "encoded_score_mean": float(encoded_score_mean.cpu()),
                "prior_score_mean": float(prior_score_mean.cpu()),
            }
            append_log(log_dir / "train_log.csv", row)
            progress.set_postfix(
                loss=f"{row['total_loss']:.3f}",
                recon=f"{row['reconstruction_loss']:.3f}",
                reg=f"{row['mmd_loss'] if regularizer_type == 'mmd' else row['adversarial_loss']:.3f}",
                mmd_w=f"{row['mmd_weight']:.2f}" if regularizer_type == "mmd" else "nan",
            )

            if global_step % sample_interval == 0:
                sample_decoder(
                    encoder,
                    decoder,
                    fixed_noise,
                    fixed_images,
                    sample_dir,
                    global_step,
                    fixed_images.size(0),
                )
                saved_current_step_sample = True

            if max_batches is not None and batches_this_run >= max_batches:
                should_stop = True
                break

        if (epoch + 1) % checkpoint_interval == 0 or should_stop:
            save_wae_checkpoint(
                checkpoint_dir / f"wae_{regularizer_type}_epoch_{epoch + 1:04d}.pt",
                encoder,
                decoder,
                optimizer_autoencoder,
                epoch,
                global_step,
                config,
                latent_discriminator=latent_discriminator,
                optimizer_discriminator=optimizer_discriminator,
            )

        if should_stop:
            break

    if global_step > 0 and not saved_current_step_sample:
        sample_decoder(
            encoder,
            decoder,
            fixed_noise,
            fixed_images,
            sample_dir,
            global_step,
            fixed_images.size(0),
        )

    save_wae_checkpoint(
        checkpoint_dir / f"wae_{regularizer_type}_latest.pt",
        encoder,
        decoder,
        optimizer_autoencoder,
        max(0, last_epoch),
        global_step,
        config,
        latent_discriminator=latent_discriminator,
        optimizer_discriminator=optimizer_discriminator,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WAE on 2D DFN binary images.")
    parser.add_argument("--config", type=Path, default=Path("configs/wae_mmd_128.yaml"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_config(resolve_path(args.config)), resume=args.resume, max_batches=args.max_batches)
