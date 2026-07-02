import argparse
import csv
import math
import sys
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
from src.models.sphere_encoder import SphereEncoder, weights_init
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


def create_model(config: dict[str, Any]) -> SphereEncoder:
    model_cfg = config["model"]
    return SphereEncoder(
        latent_dim=int(model_cfg.get("latent_dim", 16)),
        image_channels=int(model_cfg.get("image_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        noise_angle_degrees=float(model_cfg.get("noise_angle_degrees", 80.0)),
        spherify_eps=float(model_cfg.get("spherify_eps", 1e-6)),
    )


def create_optimizer(
    model: torch.nn.Module,
    training_cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    betas = (
        float(training_cfg.get("beta1", 0.0)),
        float(training_cfg.get("beta2", 0.9)),
    )
    return torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        betas=betas,
    )


def tanh_range_to_probability(images: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return ((images + 1.0) / 2.0).clamp(eps, 1.0 - eps)


def probability_to_tanh_range(images: torch.Tensor) -> torch.Tensor:
    return images.clamp(0.0, 1.0) * 2.0 - 1.0


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
        raise ValueError("loss.foreground_ratio_margin must be non-negative")
    if mode == "l1":
        return F.l1_loss(predicted_ratio, target_ratio)
    if mode == "over_density":
        return F.relu(difference - margin).mean()
    if mode == "under_density":
        return F.relu(-difference - margin).mean()
    raise ValueError(
        "loss.foreground_ratio_mode must be one of: "
        "l1, over_density, under_density"
    )


def structural_reconstruction_loss(
    decoder_logits: torch.Tensor,
    reconstructed_probability: torch.Tensor,
    real_images: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eps = float(loss_cfg.get("probability_eps", 1e-6))
    target_probability = tanh_range_to_probability(real_images, eps=eps)

    bce_weight = float(loss_cfg.get("bce_weight", 0.4))
    if not 0.0 <= bce_weight <= 1.0:
        raise ValueError("loss.bce_weight must be in [0, 1]")
    edge_weight = float(loss_cfg.get("edge_weight", 0.0))
    multiscale_dice_weight = float(loss_cfg.get("multiscale_dice_weight", 0.0))
    foreground_ratio_weight = float(loss_cfg.get("foreground_ratio_weight", 0.0))
    if min(edge_weight, multiscale_dice_weight, foreground_ratio_weight) < 0.0:
        raise ValueError("loss structural weights must be non-negative")

    smooth = float(loss_cfg.get("dice_smooth", 1.0))
    predicted_probability = reconstructed_probability.clamp(eps, 1.0 - eps)
    bce = F.binary_cross_entropy_with_logits(decoder_logits, target_probability)
    dice = dice_loss(predicted_probability, target_probability, smooth=smooth)
    base_loss = bce_weight * bce + (1.0 - bce_weight) * dice

    edge = edge_alignment_loss(predicted_probability, target_probability)
    scales = [int(scale) for scale in loss_cfg.get("multiscale_dice_scales", [2, 4])]
    multiscale_dice = multiscale_dice_loss(
        predicted_probability,
        target_probability,
        scales=scales,
        smooth=smooth,
    )
    foreground_ratio = foreground_ratio_loss(
        predicted_probability,
        target_probability,
        mode=str(loss_cfg.get("foreground_ratio_mode", "l1")).lower(),
        margin=float(loss_cfg.get("foreground_ratio_margin", 0.0)),
    )

    weighted_edge = edge_weight * edge
    weighted_multiscale_dice = multiscale_dice_weight * multiscale_dice
    weighted_foreground_ratio = foreground_ratio_weight * foreground_ratio
    total = base_loss + weighted_edge + weighted_multiscale_dice + weighted_foreground_ratio
    return total, {
        "base_reconstruction_loss": base_loss.detach(),
        "bce_loss": bce.detach(),
        "dice_loss": dice.detach(),
        "edge_loss": edge.detach(),
        "weighted_edge_loss": weighted_edge.detach(),
        "multiscale_dice_loss": multiscale_dice.detach(),
        "weighted_multiscale_dice_loss": weighted_multiscale_dice.detach(),
        "foreground_ratio_loss": foreground_ratio.detach(),
        "weighted_foreground_ratio_loss": weighted_foreground_ratio.detach(),
    }


def paper_reconstruction_loss(
    reconstructed_probability: torch.Tensor,
    real_images: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eps = float(loss_cfg.get("probability_eps", 1e-6))
    target_probability = tanh_range_to_probability(real_images, eps=eps)
    beta = float(loss_cfg.get("smooth_l1_beta", 1.0))
    loss = F.smooth_l1_loss(
        reconstructed_probability.clamp(eps, 1.0 - eps),
        target_probability,
        beta=beta,
    )
    nan = loss.new_tensor(float("nan"))
    return loss, {
        "base_reconstruction_loss": loss.detach(),
        "bce_loss": nan,
        "dice_loss": nan,
        "edge_loss": nan,
        "weighted_edge_loss": nan,
        "multiscale_dice_loss": nan,
        "weighted_multiscale_dice_loss": nan,
        "foreground_ratio_loss": nan,
        "weighted_foreground_ratio_loss": nan,
    }


def reconstruction_loss(
    decoder_logits: torch.Tensor,
    reconstructed_probability: torch.Tensor,
    real_images: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_type = str(loss_cfg.get("reconstruction", "bce_dice_structural")).lower()
    if loss_type in {"bce_dice_structural", "structural", "dfn_structural"}:
        return structural_reconstruction_loss(
            decoder_logits,
            reconstructed_probability,
            real_images,
            loss_cfg,
        )
    if loss_type in {"smooth_l1", "paper", "l1"}:
        return paper_reconstruction_loss(
            reconstructed_probability,
            real_images,
            loss_cfg,
        )
    raise ValueError(
        "loss.reconstruction must be one of: "
        "bce_dice_structural, structural, smooth_l1, paper"
    )


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_flat = x.flatten(start_dim=1)
    y_flat = y.flatten(start_dim=1)
    return 1.0 - F.cosine_similarity(x_flat, y_flat, dim=1).mean()


def sphere_encoder_loss(
    model: SphereEncoder,
    real_images: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs = model(
        real_images,
        sub_noise_max_scale=float(loss_cfg.get("sub_noise_max_scale", 0.5)),
    )
    recon_loss, recon_metrics = reconstruction_loss(
        outputs["reconstruction_logits"],
        outputs["reconstruction_probability"],
        real_images,
        loss_cfg,
    )

    pixel_consistency_raw = F.smooth_l1_loss(
        outputs["large_noise_probability"],
        outputs["reconstruction_probability"].detach(),
        beta=float(loss_cfg.get("smooth_l1_beta", 1.0)),
    )
    large_noise_images = probability_to_tanh_range(outputs["large_noise_probability"])
    encoded_large_noise = model.encode_sphere(large_noise_images)
    latent_consistency_raw = cosine_distance(
        encoded_large_noise,
        outputs["clean_latent"].detach(),
    )

    lambda_recon = float(loss_cfg.get("lambda_recon", 1.0))
    pixel_consistency_weight = float(loss_cfg.get("pixel_consistency_weight", 1.0))
    latent_consistency_weight = float(loss_cfg.get("latent_consistency_weight", 0.1))
    if min(lambda_recon, pixel_consistency_weight, latent_consistency_weight) < 0.0:
        raise ValueError("loss weights must be non-negative")

    weighted_recon = lambda_recon * recon_loss
    weighted_pixel_consistency = pixel_consistency_weight * pixel_consistency_raw
    weighted_latent_consistency = latent_consistency_weight * latent_consistency_raw
    total_loss = (
        weighted_recon
        + weighted_pixel_consistency
        + weighted_latent_consistency
    )

    sphere_norm = outputs["clean_latent"].norm(dim=1)
    metrics = {
        "total_loss": total_loss.detach(),
        "reconstruction_loss": recon_loss.detach(),
        "weighted_reconstruction_loss": weighted_recon.detach(),
        "pixel_consistency_loss": pixel_consistency_raw.detach(),
        "weighted_pixel_consistency_loss": weighted_pixel_consistency.detach(),
        "latent_consistency_loss": latent_consistency_raw.detach(),
        "weighted_latent_consistency_loss": weighted_latent_consistency.detach(),
        "sphere_norm_mean": sphere_norm.detach().mean(),
        "sphere_norm_std": sphere_norm.detach().std(unbiased=False),
        "noise_angle_deg": real_images.new_tensor(model.noise_angle_degrees),
        "sigma_max": real_images.new_tensor(model.sigma_max),
    }
    metrics.update(recon_metrics)
    return total_loss, metrics


@torch.no_grad()
def save_sphere_encoder_samples(
    model: SphereEncoder,
    fixed_noise: torch.Tensor,
    fixed_images: torch.Tensor,
    sample_dir: Path,
    global_step: int,
    num_sample_images: int,
) -> None:
    model.eval()
    reconstructed_probability = model.reconstruct(fixed_images)
    one_step_probability = model.generate(fixed_noise, steps=1)
    two_step_probability = model.generate(fixed_noise, steps=2)
    four_step_probability = model.generate(fixed_noise, steps=4)
    nrow = int(math.sqrt(num_sample_images))

    save_image_grid(
        probability_to_tanh_range(one_step_probability),
        sample_dir / f"step_{global_step:07d}.png",
        nrow=nrow,
    )
    save_image_grid(
        probability_to_tanh_range(two_step_probability),
        sample_dir / f"step_{global_step:07d}_2step.png",
        nrow=nrow,
    )
    save_image_grid(
        probability_to_tanh_range(four_step_probability),
        sample_dir / f"step_{global_step:07d}_4step.png",
        nrow=nrow,
    )
    save_image_grid(
        probability_to_tanh_range(reconstructed_probability),
        sample_dir / f"step_{global_step:07d}_recon.png",
        nrow=nrow,
    )
    model.train()


def save_sphere_encoder_checkpoint(
    path: str | Path,
    model: SphereEncoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config,
        },
        path,
    )


def load_sphere_encoder_checkpoint(
    path: str | Path,
    model: SphereEncoder,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, int]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0))


def create_fixed_images(
    dataset: DFNDataset,
    num_images: int,
    device: torch.device,
) -> torch.Tensor:
    num_images = min(num_images, len(dataset))
    images = torch.stack([dataset[index] for index in range(num_images)])
    return images.to(device)


def metric_to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def train(
    config: dict[str, Any],
    resume: str | Path | None = None,
    max_batches: int | None = None,
) -> None:
    training_cfg = config["training"]
    data_cfg = config["data"]
    loss_cfg = config.get("loss", {})
    outputs_cfg = config["outputs"]

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

    model = create_model(config).to(device)
    model.apply(weights_init)
    optimizer = create_optimizer(model, training_cfg)

    start_epoch = 0
    global_step = 0
    if resume is not None:
        start_epoch, global_step = load_sphere_encoder_checkpoint(
            resume,
            model,
            optimizer,
            map_location=device,
        )
        start_epoch += 1

    sample_dir = resolve_path(outputs_cfg["sample_dir"])
    checkpoint_dir = resolve_path(outputs_cfg["checkpoint_dir"])
    log_dir = resolve_path(outputs_cfg["log_dir"])
    num_sample_images = int(training_cfg["num_sample_images"])
    fixed_images = create_fixed_images(dataset, num_sample_images, device)
    fixed_noise = torch.randn(fixed_images.size(0), model.latent_dim, device=device)
    sample_interval = int(training_cfg["sample_interval"])
    checkpoint_interval = int(training_cfg["checkpoint_interval"])
    batches_this_run = 0
    should_stop = False
    saved_current_step_sample = False
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, int(training_cfg["num_epochs"])):
        last_epoch = epoch
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{training_cfg['num_epochs']}")
        for real_images in progress:
            real_images = real_images.to(device)
            total_loss, metrics = sphere_encoder_loss(model, real_images, loss_cfg)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            global_step += 1
            batches_this_run += 1
            saved_current_step_sample = False
            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "total_loss": metric_to_float(metrics["total_loss"]),
                "reconstruction_loss": metric_to_float(metrics["reconstruction_loss"]),
                "weighted_reconstruction_loss": metric_to_float(
                    metrics["weighted_reconstruction_loss"]
                ),
                "base_reconstruction_loss": metric_to_float(
                    metrics["base_reconstruction_loss"]
                ),
                "bce_loss": metric_to_float(metrics["bce_loss"]),
                "dice_loss": metric_to_float(metrics["dice_loss"]),
                "edge_loss": metric_to_float(metrics["edge_loss"]),
                "weighted_edge_loss": metric_to_float(metrics["weighted_edge_loss"]),
                "multiscale_dice_loss": metric_to_float(
                    metrics["multiscale_dice_loss"]
                ),
                "weighted_multiscale_dice_loss": metric_to_float(
                    metrics["weighted_multiscale_dice_loss"]
                ),
                "foreground_ratio_loss": metric_to_float(
                    metrics["foreground_ratio_loss"]
                ),
                "weighted_foreground_ratio_loss": metric_to_float(
                    metrics["weighted_foreground_ratio_loss"]
                ),
                "pixel_consistency_loss": metric_to_float(
                    metrics["pixel_consistency_loss"]
                ),
                "weighted_pixel_consistency_loss": metric_to_float(
                    metrics["weighted_pixel_consistency_loss"]
                ),
                "latent_consistency_loss": metric_to_float(
                    metrics["latent_consistency_loss"]
                ),
                "weighted_latent_consistency_loss": metric_to_float(
                    metrics["weighted_latent_consistency_loss"]
                ),
                "sphere_norm_mean": metric_to_float(metrics["sphere_norm_mean"]),
                "sphere_norm_std": metric_to_float(metrics["sphere_norm_std"]),
                "noise_angle_deg": metric_to_float(metrics["noise_angle_deg"]),
                "sigma_max": metric_to_float(metrics["sigma_max"]),
            }
            append_log(log_dir / "train_log.csv", row)
            progress.set_postfix(
                loss=f"{row['total_loss']:.4f}",
                recon=f"{row['reconstruction_loss']:.4f}",
                pixcon=f"{row['pixel_consistency_loss']:.4f}",
                latcon=f"{row['latent_consistency_loss']:.4f}",
            )

            if global_step % sample_interval == 0:
                save_sphere_encoder_samples(
                    model,
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
            save_sphere_encoder_checkpoint(
                checkpoint_dir / f"sphere_encoder_epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                epoch,
                global_step,
                config,
            )

        if should_stop:
            break

    if global_step > 0 and not saved_current_step_sample:
        save_sphere_encoder_samples(
            model,
            fixed_noise,
            fixed_images,
            sample_dir,
            global_step,
            fixed_images.size(0),
        )

    save_sphere_encoder_checkpoint(
        checkpoint_dir / "sphere_encoder_latest.pt",
        model,
        optimizer,
        max(0, last_epoch),
        global_step,
        config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sphere Encoder on 2D DFN binary images.")
    parser.add_argument("--config", type=Path, default=Path("configs/sphere_encoder_16_structural.yaml"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_config(resolve_path(args.config)), resume=args.resume, max_batches=args.max_batches)
