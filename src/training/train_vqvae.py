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
from src.models.vqvae import VQVAE, weights_init
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


def create_model(config: dict[str, Any]) -> VQVAE:
    model_cfg = config["model"]
    return VQVAE(
        image_channels=int(model_cfg.get("image_channels", 1)),
        hidden_channels=int(model_cfg.get("hidden_channels", 128)),
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
        num_embeddings=int(model_cfg.get("num_embeddings", 512)),
        commitment_cost=float(model_cfg.get("commitment_cost", 0.25)),
        num_residual_blocks=int(model_cfg.get("num_residual_blocks", 2)),
    )


def create_optimizer(model: torch.nn.Module, training_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    betas = (
        float(training_cfg.get("beta1", 0.9)),
        float(training_cfg.get("beta2", 0.999)),
    )
    return torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        betas=betas,
    )


def reconstruction_loss(
    reconstructed: torch.Tensor,
    real_images: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    loss_type = loss_type.lower()
    if loss_type == "l1":
        return F.l1_loss(reconstructed, real_images)
    if loss_type == "mse":
        return F.mse_loss(reconstructed, real_images)
    raise ValueError("loss.reconstruction must be either 'l1' or 'mse'")


def vqvae_loss(
    model: VQVAE,
    real_images: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstructed, vq_loss, perplexity, indices = model(real_images)
    recon_loss = reconstruction_loss(
        reconstructed,
        real_images,
        str(loss_cfg.get("reconstruction", "l1")),
    )
    total_loss = (
        float(loss_cfg.get("lambda_recon", 1.0)) * recon_loss
        + float(loss_cfg.get("lambda_vq", 1.0)) * vq_loss
    )
    metrics = {
        "total_loss": total_loss.detach(),
        "reconstruction_loss": recon_loss.detach(),
        "vq_loss": vq_loss.detach(),
        "perplexity": perplexity.detach(),
        "code_usage": torch.tensor(
            indices.unique().numel() / model.num_embeddings,
            device=real_images.device,
        ),
    }
    return total_loss, metrics


@torch.no_grad()
def sample_random_codes(
    model: VQVAE,
    num_images: int,
    latent_size: int,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.randint(
        low=0,
        high=model.num_embeddings,
        size=(num_images, latent_size, latent_size),
        device=device,
    )
    return model.decode_indices(indices)


@torch.no_grad()
def save_vqvae_samples(
    model: VQVAE,
    fixed_images: torch.Tensor,
    sample_dir: Path,
    global_step: int,
    num_sample_images: int,
    latent_size: int,
    save_random: bool = True,
) -> None:
    model.eval()
    reconstructed, _, _, _ = model(fixed_images)
    nrow = int(math.sqrt(num_sample_images))
    save_image_grid(
        reconstructed,
        sample_dir / f"step_{global_step:07d}_recon.png",
        nrow=nrow,
    )
    if save_random:
        random_samples = sample_random_codes(
            model,
            fixed_images.size(0),
            latent_size,
            fixed_images.device,
        )
        save_image_grid(
            random_samples,
            sample_dir / f"step_{global_step:07d}_random.png",
            nrow=nrow,
        )
    model.train()


def save_vqvae_checkpoint(
    path: str | Path,
    model: VQVAE,
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


def load_vqvae_checkpoint(
    path: str | Path,
    model: VQVAE,
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
    sampling_cfg = config.get("sampling", {})

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
        start_epoch, global_step = load_vqvae_checkpoint(
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
    latent_downsample_factor = int(config["model"].get("latent_downsample_factor", 8))
    latent_size = int(data_cfg["image_size"]) // latent_downsample_factor
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
            total_loss, metrics = vqvae_loss(model, real_images, loss_cfg)

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
                "vq_loss": float(metrics["vq_loss"].cpu()),
                "perplexity": float(metrics["perplexity"].cpu()),
                "code_usage": float(metrics["code_usage"]),
            }
            append_log(log_dir / "train_log.csv", row)
            progress.set_postfix(
                loss=f"{row['total_loss']:.4f}",
                recon=f"{row['reconstruction_loss']:.4f}",
                ppl=f"{row['perplexity']:.1f}",
            )

            if global_step % sample_interval == 0:
                save_vqvae_samples(
                    model,
                    fixed_images,
                    sample_dir,
                    global_step,
                    fixed_images.size(0),
                    latent_size,
                    save_random=bool(sampling_cfg.get("save_random_samples", True)),
                )
                saved_current_step_sample = True

            if max_batches is not None and batches_this_run >= max_batches:
                should_stop = True
                break

        if (epoch + 1) % checkpoint_interval == 0 or should_stop:
            save_vqvae_checkpoint(
                checkpoint_dir / f"vqvae_epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                epoch,
                global_step,
                config,
            )

        if should_stop:
            break

    if global_step > 0 and not saved_current_step_sample:
        save_vqvae_samples(
            model,
            fixed_images,
            sample_dir,
            global_step,
            fixed_images.size(0),
            latent_size,
            save_random=bool(sampling_cfg.get("save_random_samples", True)),
        )

    save_vqvae_checkpoint(
        checkpoint_dir / "vqvae_latest.pt",
        model,
        optimizer,
        max(0, last_epoch),
        global_step,
        config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VQ-VAE on 2D DFN binary images.")
    parser.add_argument("--config", type=Path, default=Path("configs/vqvae_128.yaml"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_config(resolve_path(args.config)), resume=args.resume, max_batches=args.max_batches)
