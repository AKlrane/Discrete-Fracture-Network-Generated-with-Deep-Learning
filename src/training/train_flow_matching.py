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
from src.models.flow_matching import TimeConditionedUNet, weights_init
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


def create_model(config: dict[str, Any]) -> TimeConditionedUNet:
    model_cfg = config["model"]
    return TimeConditionedUNet(
        image_channels=int(model_cfg.get("image_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        channel_multipliers=[
            int(multiplier)
            for multiplier in model_cfg.get("channel_multipliers", [1, 2, 4, 8])
        ],
        time_embedding_dim=(
            int(model_cfg["time_embedding_dim"])
            if "time_embedding_dim" in model_cfg
            else None
        ),
        groups=int(model_cfg.get("groups", 8)),
    )


def create_optimizer(
    model: torch.nn.Module,
    training_cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    betas = (
        float(training_cfg.get("beta1", 0.9)),
        float(training_cfg.get("beta2", 0.999)),
    )
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        betas=betas,
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )


def flow_matching_loss(
    model: TimeConditionedUNet,
    real_images: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = real_images.size(0)
    x0 = torch.randn_like(real_images)
    t = torch.rand(batch_size, device=real_images.device, dtype=real_images.dtype)
    t_view = t.view(batch_size, 1, 1, 1)
    x_t = (1.0 - t_view) * x0 + t_view * real_images
    target_velocity = real_images - x0
    predicted_velocity = model(x_t, t)
    loss = F.mse_loss(predicted_velocity, target_velocity)
    metrics = {
        "t_mean": t.detach().mean(),
        "target_velocity_norm": target_velocity.detach().flatten(1).norm(dim=1).mean(),
        "predicted_velocity_norm": predicted_velocity.detach().flatten(1).norm(dim=1).mean(),
    }
    return loss, metrics


def _validate_sampler(solver: str, num_steps: int) -> str:
    solver = solver.lower()
    if solver not in {"euler", "heun", "midpoint"}:
        raise ValueError("sampler.solver must be one of: euler, heun, midpoint")
    if num_steps < 1:
        raise ValueError("sampler.num_steps must be a positive integer")
    return solver


@torch.no_grad()
def sample_flow(
    model: TimeConditionedUNet,
    initial_noise: torch.Tensor,
    solver: str = "euler",
    num_steps: int = 50,
) -> torch.Tensor:
    solver = _validate_sampler(solver, num_steps)
    x = initial_noise
    dt = 1.0 / num_steps
    batch_size = x.size(0)

    for step in range(num_steps):
        t_value = step / num_steps
        t = torch.full(
            (batch_size,),
            t_value,
            device=x.device,
            dtype=x.dtype,
        )

        if solver == "euler":
            velocity = model(x, t)
            x = x + dt * velocity
        elif solver == "midpoint":
            velocity = model(x, t)
            t_mid = torch.full(
                (batch_size,),
                t_value + 0.5 * dt,
                device=x.device,
                dtype=x.dtype,
            )
            midpoint = x + 0.5 * dt * velocity
            x = x + dt * model(midpoint, t_mid)
        else:
            velocity = model(x, t)
            proposal = x + dt * velocity
            t_next = torch.full(
                (batch_size,),
                min(t_value + dt, 1.0),
                device=x.device,
                dtype=x.dtype,
            )
            next_velocity = model(proposal, t_next)
            x = x + 0.5 * dt * (velocity + next_velocity)

    return x.clamp(-1.0, 1.0)


def save_flow_samples(
    model: TimeConditionedUNet,
    fixed_noise: torch.Tensor,
    sample_dir: Path,
    global_step: int,
    num_sample_images: int,
    sampler_cfg: dict[str, Any],
) -> None:
    model.eval()
    samples = sample_flow(
        model,
        fixed_noise,
        solver=str(sampler_cfg.get("solver", "euler")),
        num_steps=int(sampler_cfg.get("num_steps", 50)),
    )
    nrow = int(math.sqrt(num_sample_images))
    save_image_grid(samples, sample_dir / f"step_{global_step:07d}.png", nrow=nrow)
    model.train()


def save_flow_matching_checkpoint(
    path: str | Path,
    model: TimeConditionedUNet,
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


def load_flow_matching_checkpoint(
    path: str | Path,
    model: TimeConditionedUNet,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, int]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0))


def train(
    config: dict[str, Any],
    resume: str | Path | None = None,
    max_batches: int | None = None,
) -> None:
    training_cfg = config["training"]
    data_cfg = config["data"]
    sampler_cfg = config.get("sampler", {})
    outputs_cfg = config["outputs"]

    if max_batches is not None and max_batches < 1:
        raise ValueError("--max_batches must be a positive integer")
    _validate_sampler(
        str(sampler_cfg.get("solver", "euler")),
        int(sampler_cfg.get("num_steps", 50)),
    )

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
        start_epoch, global_step = load_flow_matching_checkpoint(
            resume,
            model,
            optimizer,
            map_location=device,
        )
        start_epoch += 1

    sample_dir = resolve_path(outputs_cfg["sample_dir"])
    checkpoint_dir = resolve_path(outputs_cfg["checkpoint_dir"])
    log_dir = resolve_path(outputs_cfg["log_dir"])
    image_channels = int(config["model"].get("image_channels", 1))
    image_size = int(data_cfg["image_size"])
    fixed_noise = torch.randn(
        int(training_cfg["num_sample_images"]),
        image_channels,
        image_size,
        image_size,
        device=device,
    )
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
            loss, metrics = flow_matching_loss(model, real_images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            batches_this_run += 1
            saved_current_step_sample = False
            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "loss": float(loss.detach().cpu()),
                "t_mean": float(metrics["t_mean"].cpu()),
                "target_velocity_norm": float(metrics["target_velocity_norm"].cpu()),
                "predicted_velocity_norm": float(metrics["predicted_velocity_norm"].cpu()),
                "sampler_solver": str(sampler_cfg.get("solver", "euler")),
                "sampler_num_steps": int(sampler_cfg.get("num_steps", 50)),
            }
            append_log(log_dir / "train_log.csv", row)
            progress.set_postfix(loss=f"{row['loss']:.4f}", t=f"{row['t_mean']:.3f}")

            if global_step % sample_interval == 0:
                save_flow_samples(
                    model,
                    fixed_noise,
                    sample_dir,
                    global_step,
                    int(training_cfg["num_sample_images"]),
                    sampler_cfg,
                )
                saved_current_step_sample = True

            if max_batches is not None and batches_this_run >= max_batches:
                should_stop = True
                break

        if (epoch + 1) % checkpoint_interval == 0 or should_stop:
            save_flow_matching_checkpoint(
                checkpoint_dir / f"flow_matching_epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                epoch,
                global_step,
                config,
            )

        if should_stop:
            break

    if global_step > 0 and not saved_current_step_sample:
        save_flow_samples(
            model,
            fixed_noise,
            sample_dir,
            global_step,
            int(training_cfg["num_sample_images"]),
            sampler_cfg,
        )

    save_flow_matching_checkpoint(
        checkpoint_dir / "flow_matching_latest.pt",
        model,
        optimizer,
        max(0, last_epoch),
        global_step,
        config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train pixel-space Flow Matching on 2D DFN binary images."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/flow_matching_128.yaml"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_config(resolve_path(args.config)), resume=args.resume, max_batches=args.max_batches)
