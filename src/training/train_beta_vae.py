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
from src.models.beta_vae import BetaVAE, weights_init
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


def create_model(config: dict[str, Any]) -> BetaVAE:
    model_cfg = config["model"]
    return BetaVAE(
        latent_dim=int(model_cfg.get("latent_dim", 16)),
        image_channels=int(model_cfg.get("image_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 64)),
    )


def create_optimizer(model: torch.nn.Module, training_cfg: dict[str, Any]) -> torch.optim.Optimizer:
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


def reconstruction_loss(
    decoder_logits: torch.Tensor,
    reconstructed_probability: torch.Tensor,
    real_images: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_type = str(loss_cfg.get("reconstruction", "bce_dice")).lower()
    eps = float(loss_cfg.get("probability_eps", 1e-6))
    target_probability = tanh_range_to_probability(real_images, eps=eps)
    nan_metric = torch.tensor(float("nan"), device=real_images.device)
    if loss_type == "l1":
        loss = F.l1_loss(
            reconstructed_probability.clamp(eps, 1.0 - eps),
            target_probability,
        )
        return loss, {"bce_loss": nan_metric, "dice_loss": nan_metric}
    if loss_type != "bce_dice":
        raise ValueError("loss.reconstruction must be either 'l1' or 'bce_dice'")

    bce_weight = float(loss_cfg.get("bce_weight", 0.5))
    if not 0.0 <= bce_weight <= 1.0:
        raise ValueError("loss.bce_weight must be in [0, 1]")

    smooth = float(loss_cfg.get("dice_smooth", 1.0))
    predicted_probability = torch.sigmoid(decoder_logits).clamp(eps, 1.0 - eps)
    bce = F.binary_cross_entropy_with_logits(decoder_logits, target_probability)
    dice = dice_loss(predicted_probability, target_probability, smooth=smooth)
    total = bce_weight * bce + (1.0 - bce_weight) * dice
    return total, {"bce_loss": bce.detach(), "dice_loss": dice.detach()}


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    per_sample = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=1)
    return per_sample.mean()


def linear_schedule(start: float, end: float, warmup_steps: int, step: int) -> float:
    if warmup_steps <= 0:
        return end
    progress = min(max(step, 0) / warmup_steps, 1.0)
    return start + (end - start) * progress


def kl_weight(loss_cfg: dict[str, Any], step: int) -> float:
    max_weight = float(loss_cfg.get("beta_kl", 1.0))
    warmup_steps = int(loss_cfg.get("beta_kl_warmup_steps", 0))
    if warmup_steps <= 0:
        return max_weight
    return max_weight * min(max(step, 0) / warmup_steps, 1.0)


def kl_regularization(
    kl: torch.Tensor,
    loss_cfg: dict[str, Any],
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    objective = str(loss_cfg.get("kl_objective", "beta")).lower()
    if objective in {"beta", "standard", "weighted"}:
        current_kl_weight = kl_weight(loss_cfg, step)
        weighted_kl = current_kl_weight * kl
        nan = kl.new_tensor(float("nan"))
        return weighted_kl, {
            "kl_weight": kl.new_tensor(current_kl_weight),
            "kl_capacity": nan,
            "kl_gamma": nan,
            "kl_capacity_error": nan,
        }

    if objective != "capacity":
        raise ValueError("loss.kl_objective must be either 'beta' or 'capacity'")

    capacity_start = float(loss_cfg.get("capacity_start", 0.0))
    capacity_max = float(loss_cfg.get("capacity_max", 8.0))
    capacity_warmup_steps = int(loss_cfg.get("capacity_warmup_steps", 50000))
    capacity = linear_schedule(
        capacity_start,
        capacity_max,
        capacity_warmup_steps,
        step,
    )
    gamma = float(loss_cfg.get("capacity_gamma", 0.05))
    if gamma < 0.0:
        raise ValueError("loss.capacity_gamma must be non-negative")

    capacity_tensor = kl.new_tensor(capacity)
    capacity_error = kl - capacity_tensor
    capacity_loss = str(loss_cfg.get("capacity_loss", "absolute")).lower()
    if capacity_loss in {"absolute", "l1", "abs"}:
        weighted_kl = gamma * capacity_error.abs()
    elif capacity_loss in {"squared", "l2", "mse"}:
        weighted_kl = gamma * capacity_error.square()
    else:
        raise ValueError("loss.capacity_loss must be either 'absolute' or 'squared'")

    return weighted_kl, {
        "kl_weight": kl.new_tensor(gamma),
        "kl_capacity": capacity_tensor,
        "kl_gamma": kl.new_tensor(gamma),
        "kl_capacity_error": capacity_error.detach(),
    }


def beta_vae_loss(
    model: BetaVAE,
    real_images: torch.Tensor,
    loss_cfg: dict[str, Any],
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstructed_probability, decoder_logits, mu, logvar, _ = model(real_images)
    recon_loss, reconstruction_metrics = reconstruction_loss(
        decoder_logits,
        reconstructed_probability,
        real_images,
        loss_cfg,
    )
    kl = kl_divergence(mu, logvar)
    weighted_kl, kl_metrics = kl_regularization(kl, loss_cfg, step)
    total_loss = float(loss_cfg.get("lambda_recon", 1.0)) * recon_loss + weighted_kl
    posterior_std = torch.exp(0.5 * logvar)
    metrics = {
        "total_loss": total_loss.detach(),
        "reconstruction_loss": recon_loss.detach(),
        "bce_loss": reconstruction_metrics["bce_loss"],
        "dice_loss": reconstruction_metrics["dice_loss"],
        "kl_loss": kl.detach(),
        "kl_per_dim": (kl / mu.size(1)).detach(),
        "kl_weight": kl_metrics["kl_weight"],
        "kl_capacity": kl_metrics["kl_capacity"],
        "kl_gamma": kl_metrics["kl_gamma"],
        "kl_capacity_error": kl_metrics["kl_capacity_error"],
        "weighted_kl_loss": weighted_kl.detach(),
        "mu_mean": mu.detach().mean(),
        "mu_std": mu.detach().std(unbiased=False),
        "logvar_mean": logvar.detach().mean(),
        "posterior_std_mean": posterior_std.detach().mean(),
    }
    return total_loss, metrics


@torch.no_grad()
def save_beta_vae_samples(
    model: BetaVAE,
    fixed_noise: torch.Tensor,
    fixed_images: torch.Tensor,
    sample_dir: Path,
    global_step: int,
    num_sample_images: int,
) -> None:
    model.eval()
    samples_probability = model.decode(fixed_noise)
    mu, _ = model.encode(fixed_images)
    reconstructed_probability = model.decode(mu)
    nrow = int(math.sqrt(num_sample_images))
    save_image_grid(
        probability_to_tanh_range(samples_probability),
        sample_dir / f"step_{global_step:07d}.png",
        nrow=nrow,
    )
    save_image_grid(
        probability_to_tanh_range(reconstructed_probability),
        sample_dir / f"step_{global_step:07d}_recon.png",
        nrow=nrow,
    )
    model.train()


def save_beta_vae_checkpoint(
    path: str | Path,
    model: BetaVAE,
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


def load_beta_vae_checkpoint(
    path: str | Path,
    model: BetaVAE,
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
        start_epoch, global_step = load_beta_vae_checkpoint(
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
            total_loss, metrics = beta_vae_loss(model, real_images, loss_cfg, global_step)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            global_step += 1
            batches_this_run += 1
            saved_current_step_sample = False
            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "total_loss": float(metrics["total_loss"].cpu()),
                "reconstruction_loss": float(metrics["reconstruction_loss"].cpu()),
                "bce_loss": float(metrics["bce_loss"].cpu()),
                "dice_loss": float(metrics["dice_loss"].cpu()),
                "kl_loss": float(metrics["kl_loss"].cpu()),
                "kl_per_dim": float(metrics["kl_per_dim"].cpu()),
                "kl_weight": float(metrics["kl_weight"].cpu()),
                "kl_capacity": float(metrics["kl_capacity"].cpu()),
                "kl_gamma": float(metrics["kl_gamma"].cpu()),
                "kl_capacity_error": float(metrics["kl_capacity_error"].cpu()),
                "weighted_kl_loss": float(metrics["weighted_kl_loss"].cpu()),
                "mu_mean": float(metrics["mu_mean"].cpu()),
                "mu_std": float(metrics["mu_std"].cpu()),
                "logvar_mean": float(metrics["logvar_mean"].cpu()),
                "posterior_std_mean": float(metrics["posterior_std_mean"].cpu()),
            }
            append_log(log_dir / "train_log.csv", row)
            progress.set_postfix(
                loss=f"{row['total_loss']:.4f}",
                recon=f"{row['reconstruction_loss']:.4f}",
                kl=f"{row['kl_loss']:.2f}",
                beta=f"{row['kl_weight']:.4f}",
            )

            if global_step % sample_interval == 0:
                save_beta_vae_samples(
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
            save_beta_vae_checkpoint(
                checkpoint_dir / f"beta_vae_epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                epoch,
                global_step,
                config,
            )

        if should_stop:
            break

    if global_step > 0 and not saved_current_step_sample:
        save_beta_vae_samples(
            model,
            fixed_noise,
            fixed_images,
            sample_dir,
            global_step,
            fixed_images.size(0),
        )

    save_beta_vae_checkpoint(
        checkpoint_dir / "beta_vae_latest.pt",
        model,
        optimizer,
        max(0, last_epoch),
        global_step,
        config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train beta-VAE on 2D DFN binary images.")
    parser.add_argument("--config", type=Path, default=Path("configs/beta_vae_16.yaml"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_config(resolve_path(args.config)), resume=args.resume, max_batches=args.max_batches)
