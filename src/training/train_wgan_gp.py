import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import autograd
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dfn_dataset import DFNDataset
from src.models.wgan_gp import Critic, Generator, weights_init
from src.utils.checkpoint import load_checkpoint, save_checkpoint
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


def gradient_penalty(
    critic: Critic,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real_images.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (epsilon * real_images + (1.0 - epsilon) * fake_images).requires_grad_(True)
    interpolated_score = critic(interpolated)

    gradients = autograd.grad(
        outputs=interpolated_score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


def append_log(log_path: Path, row: dict[str, float | int]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(config: dict[str, Any], resume: str | Path | None = None) -> None:
    training_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    outputs_cfg = config["outputs"]

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
    generator = Generator(latent_dim=latent_dim, base_channels=base_channels).to(device)
    critic = Critic(base_channels=base_channels).to(device)
    generator.apply(weights_init)
    critic.apply(weights_init)

    # WGAN-GP commonly uses Adam betas=(0.0, 0.9); beta1=0.0 avoids extra momentum in the critic.
    betas = (float(training_cfg["beta1"]), float(training_cfg["beta2"]))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(training_cfg["lr"]), betas=betas)
    optimizer_c = torch.optim.Adam(critic.parameters(), lr=float(training_cfg["lr"]), betas=betas)

    start_epoch = 0
    global_step = 0
    if resume is not None:
        start_epoch, global_step = load_checkpoint(
            resume,
            generator,
            critic,
            optimizer_g,
            optimizer_c,
            map_location=device,
        )
        start_epoch += 1

    sample_dir = resolve_path(outputs_cfg["sample_dir"])
    checkpoint_dir = resolve_path(outputs_cfg["checkpoint_dir"])
    log_dir = resolve_path(outputs_cfg["log_dir"])
    fixed_noise = torch.randn(int(training_cfg["num_sample_images"]), latent_dim, device=device)
    last_generator_loss = torch.tensor(float("nan"), device=device)

    for epoch in range(start_epoch, int(training_cfg["num_epochs"])):
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{training_cfg['num_epochs']}")
        for real_images in progress:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            for _ in range(int(training_cfg["critic_steps"])):
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_images = generator(z).detach()
                real_score = critic(real_images)
                fake_score = critic(fake_images)
                gp = gradient_penalty(critic, real_images, fake_images, device)
                critic_loss = (
                    fake_score.mean()
                    - real_score.mean()
                    + float(training_cfg["lambda_gp"]) * gp
                )
                optimizer_c.zero_grad(set_to_none=True)
                critic_loss.backward()
                optimizer_c.step()

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(z)
            fake_score_for_g = critic(fake_images)
            generator_loss = -fake_score_for_g.mean()
            optimizer_g.zero_grad(set_to_none=True)
            generator_loss.backward()
            optimizer_g.step()
            last_generator_loss = generator_loss.detach()

            global_step += 1
            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "critic_loss": float(critic_loss.detach().cpu()),
                "generator_loss": float(last_generator_loss.cpu()),
                "gradient_penalty": float(gp.detach().cpu()),
                "real_score_mean": float(real_score.detach().mean().cpu()),
                "fake_score_mean": float(fake_score.detach().mean().cpu()),
            }
            append_log(log_dir / "train_log.csv", row)
            progress.set_postfix(
                c_loss=f"{row['critic_loss']:.3f}",
                g_loss=f"{row['generator_loss']:.3f}",
                gp=f"{row['gradient_penalty']:.3f}",
            )

            if global_step % int(training_cfg["sample_interval"]) == 0:
                generator.eval()
                with torch.no_grad():
                    samples = generator(fixed_noise)
                nrow = int(math.sqrt(int(training_cfg["num_sample_images"])))
                save_image_grid(samples, sample_dir / f"step_{global_step:07d}.png", nrow=nrow)
                generator.train()

        if (epoch + 1) % int(training_cfg["checkpoint_interval"]) == 0:
            save_checkpoint(
                checkpoint_dir / f"wgan_gp_epoch_{epoch + 1:04d}.pt",
                generator,
                critic,
                optimizer_g,
                optimizer_c,
                epoch,
                global_step,
                config,
            )

    save_checkpoint(
        checkpoint_dir / "wgan_gp_latest.pt",
        generator,
        critic,
        optimizer_g,
        optimizer_c,
        int(training_cfg["num_epochs"]) - 1,
        global_step,
        config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WGAN-GP on 2D DFN binary images.")
    parser.add_argument("--config", type=Path, default=Path("configs/wgan_gp_128.yaml"))
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_config(resolve_path(args.config)), resume=args.resume)
