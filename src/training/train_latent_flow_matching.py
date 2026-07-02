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
from src.models.latent_flow_matching import LatentFlowMLP, weights_init
from src.models.wae import Decoder, Encoder
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
    if log_path.exists():
        with log_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        fieldnames = existing_fieldnames.copy()
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
        if fieldnames != existing_fieldnames:
            with log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerow(row)
        return

    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dictionary, got {type(checkpoint).__name__}")
    return checkpoint


def prefixed_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    for prefix in prefixes:
        if any(key.startswith(prefix) for key in state_dict):
            return {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
    return {}


def infer_autoencoder_model_config(
    checkpoint: dict[str, Any],
    autoencoder_cfg: dict[str, Any],
    flow_model_cfg: dict[str, Any],
) -> dict[str, int]:
    checkpoint_config = checkpoint.get("config")
    checkpoint_model_cfg = {}
    if isinstance(checkpoint_config, dict):
        checkpoint_model_cfg = checkpoint_config.get("model") or {}

    latent_dim = int(
        autoencoder_cfg.get(
            "latent_dim",
            checkpoint_model_cfg.get("latent_dim", flow_model_cfg.get("latent_dim", 16)),
        )
    )
    base_channels = int(autoencoder_cfg.get("base_channels", checkpoint_model_cfg.get("base_channels", 64)))
    image_channels = int(autoencoder_cfg.get("image_channels", checkpoint_model_cfg.get("image_channels", 1)))
    return {
        "latent_dim": latent_dim,
        "base_channels": base_channels,
        "image_channels": image_channels,
    }


def load_autoencoder(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[Encoder, Decoder, dict[str, int], Path]:
    autoencoder_cfg = config["autoencoder"]
    flow_model_cfg = config["flow_model"]
    checkpoint_path = resolve_path(autoencoder_cfg["checkpoint"])
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model_cfg = infer_autoencoder_model_config(checkpoint, autoencoder_cfg, flow_model_cfg)

    encoder = Encoder(**model_cfg).to(device)
    decoder = Decoder(**model_cfg).to(device)

    if "encoder" in checkpoint and "decoder" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
        encoder_state = prefixed_state_dict(
            state_dict,
            ("encoder.", "module.encoder.", "model.encoder."),
        )
        decoder_state = prefixed_state_dict(
            state_dict,
            ("decoder.", "module.decoder.", "model.decoder."),
        )
        if not encoder_state or not decoder_state:
            raise KeyError("Lightning checkpoint must contain encoder.* and decoder.* weights")
        encoder.load_state_dict(encoder_state)
        decoder.load_state_dict(decoder_state)
    else:
        raise KeyError("Autoencoder checkpoint must contain encoder/decoder weights")

    encoder.eval()
    decoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return encoder, decoder, model_cfg, checkpoint_path


def create_model(config: dict[str, Any]) -> LatentFlowMLP:
    model_cfg = config["flow_model"]
    return LatentFlowMLP(
        latent_dim=int(model_cfg.get("latent_dim", 16)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_blocks=int(model_cfg.get("num_blocks", 4)),
        time_embedding_dim=int(model_cfg.get("time_embedding_dim", 64)),
        dropout=float(model_cfg.get("dropout", 0.0)),
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


def scheduler_config(training_cfg: dict[str, Any]) -> dict[str, Any]:
    scheduler_cfg = training_cfg.get("scheduler")
    if scheduler_cfg is None:
        return {"enabled": False, "type": "none"}
    if isinstance(scheduler_cfg, str):
        return {"enabled": scheduler_cfg.lower() not in {"none", "constant", "off"}, "type": scheduler_cfg}
    if not isinstance(scheduler_cfg, dict):
        raise TypeError("training.scheduler must be a string, mapping, or omitted")
    return scheduler_cfg


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: dict[str, Any],
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    config = scheduler_config(training_cfg)
    scheduler_type = str(config.get("type", config.get("preset", "none"))).lower()
    enabled = bool(config.get("enabled", scheduler_type not in {"none", "constant", "off"}))
    if not enabled or scheduler_type in {"none", "constant", "off"}:
        return None
    if scheduler_type not in {"cosine", "cosine_warmup", "warmup_cosine"}:
        raise ValueError("training.scheduler.type must be one of: none, constant, cosine")

    base_lr = float(training_cfg["lr"])
    min_lr = float(config.get("min_lr", 0.0))
    min_lr_ratio = float(config.get("min_lr_ratio", min_lr / base_lr if base_lr > 0.0 else 0.0))
    warmup_steps = int(config.get("warmup_steps", 0))
    total_steps = int(config.get("total_steps", total_steps))

    if total_steps < 1:
        raise ValueError("scheduler total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("scheduler warmup_steps must be non-negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("scheduler min_lr_ratio must be in [0, 1]")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1.0 / warmup_steps, float(step + 1) / float(warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(total_steps - warmup_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def time_sampling_config(training_cfg: dict[str, Any] | None) -> dict[str, Any]:
    if training_cfg is None:
        return {"distribution": "uniform"}
    time_cfg = training_cfg.get("time_sampling", "uniform")
    if isinstance(time_cfg, str):
        return {"distribution": time_cfg}
    if not isinstance(time_cfg, dict):
        raise TypeError("training.time_sampling must be a string, mapping, or omitted")
    return time_cfg


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    training_cfg: dict[str, Any] | None = None,
) -> torch.Tensor:
    config = time_sampling_config(training_cfg)
    distribution = str(config.get("distribution", config.get("type", "uniform"))).lower()
    if distribution == "uniform":
        return torch.rand(batch_size, device=device, dtype=dtype)
    if distribution == "beta":
        alpha = float(config.get("alpha", config.get("beta_alpha", 2.0)))
        beta = float(config.get("beta", config.get("beta_beta", 1.0)))
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("beta time sampling requires alpha > 0 and beta > 0")
        sample_device = torch.device("cpu") if device.type == "mps" else device
        alpha_tensor = torch.tensor(alpha, device=sample_device, dtype=torch.float32)
        beta_tensor = torch.tensor(beta, device=sample_device, dtype=torch.float32)
        return torch.distributions.Beta(alpha_tensor, beta_tensor).sample((batch_size,)).to(
            device=device,
            dtype=dtype,
        )
    raise ValueError("training.time_sampling.distribution must be one of: uniform, beta")


def latent_flow_matching_loss(
    model: LatentFlowMLP,
    encoded_latents: torch.Tensor,
    training_cfg: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = encoded_latents.size(0)
    z0 = torch.randn_like(encoded_latents)
    t = sample_timesteps(batch_size, encoded_latents.device, encoded_latents.dtype, training_cfg)
    t_view = t.view(batch_size, 1)
    z_t = (1.0 - t_view) * z0 + t_view * encoded_latents
    target_velocity = encoded_latents - z0
    predicted_velocity = model(z_t, t)
    loss = F.mse_loss(predicted_velocity, target_velocity)

    cosine_similarity = F.cosine_similarity(
        predicted_velocity.detach(),
        target_velocity.detach(),
        dim=1,
    ).mean()
    metrics = {
        "t_mean": t.detach().mean(),
        "latent_mean": encoded_latents.detach().mean(),
        "latent_std": encoded_latents.detach().std(unbiased=False),
        "latent_norm": encoded_latents.detach().norm(dim=1).mean(),
        "target_velocity_norm": target_velocity.detach().norm(dim=1).mean(),
        "predicted_velocity_norm": predicted_velocity.detach().norm(dim=1).mean(),
        "velocity_cosine_similarity": cosine_similarity,
    }
    return loss, metrics


def validate_sampler(solver: str, num_steps: int) -> str:
    solver = solver.lower()
    if solver not in {"euler", "heun", "midpoint"}:
        raise ValueError("sampler.solver must be one of: euler, heun, midpoint")
    if num_steps < 1:
        raise ValueError("sampler.num_steps must be a positive integer")
    return solver


@torch.no_grad()
def sample_latent_flow(
    model: LatentFlowMLP,
    initial_noise: torch.Tensor,
    solver: str = "heun",
    num_steps: int = 100,
) -> torch.Tensor:
    solver = validate_sampler(solver, num_steps)
    z = initial_noise
    dt = 1.0 / num_steps
    batch_size = z.size(0)

    for step in range(num_steps):
        t_value = step / num_steps
        t = torch.full((batch_size,), t_value, device=z.device, dtype=z.dtype)

        if solver == "euler":
            velocity = model(z, t)
            z = z + dt * velocity
        elif solver == "midpoint":
            velocity = model(z, t)
            t_mid = torch.full((batch_size,), t_value + 0.5 * dt, device=z.device, dtype=z.dtype)
            midpoint = z + 0.5 * dt * velocity
            z = z + dt * model(midpoint, t_mid)
        else:
            velocity = model(z, t)
            proposal = z + dt * velocity
            t_next = torch.full((batch_size,), min(t_value + dt, 1.0), device=z.device, dtype=z.dtype)
            next_velocity = model(proposal, t_next)
            z = z + 0.5 * dt * (velocity + next_velocity)

    return z


def save_latent_flow_samples(
    model: LatentFlowMLP,
    decoder: Decoder,
    fixed_noise: torch.Tensor,
    sample_dir: Path,
    global_step: int,
    num_sample_images: int,
    sampler_cfg: dict[str, Any],
) -> None:
    model.eval()
    decoder.eval()
    with torch.no_grad():
        latents = sample_latent_flow(
            model,
            fixed_noise,
            solver=str(sampler_cfg.get("solver", "heun")),
            num_steps=int(sampler_cfg.get("num_steps", 100)),
        )
        samples = decoder(latents)
    nrow = int(math.sqrt(num_sample_images))
    save_image_grid(samples, sample_dir / f"step_{global_step:07d}.png", nrow=nrow)
    model.train()


def save_latent_flow_checkpoint(
    path: str | Path,
    model: LatentFlowMLP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    epoch: int,
    step: int,
    config: dict[str, Any],
    autoencoder_checkpoint: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
        "autoencoder_checkpoint": str(autoencoder_checkpoint),
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def align_scheduler_to_step(
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
) -> None:
    if step <= 0:
        return
    scheduler.last_epoch = int(step)
    lrs = [
        base_lr * lr_lambda(scheduler.last_epoch)
        for lr_lambda, base_lr in zip(scheduler.lr_lambdas, scheduler.base_lrs)
    ]
    for param_group, lr in zip(scheduler.optimizer.param_groups, lrs):
        param_group["lr"] = lr
    scheduler._last_lr = lrs


def load_latent_flow_checkpoint(
    path: str | Path,
    model: LatentFlowMLP,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, int]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    step = int(checkpoint.get("step", 0))
    if scheduler is not None and "scheduler" not in checkpoint:
        align_scheduler_to_step(scheduler, step)
    return int(checkpoint.get("epoch", 0)), step


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
    validate_sampler(str(sampler_cfg.get("solver", "heun")), int(sampler_cfg.get("num_steps", 100)))

    set_seed(int(training_cfg["seed"]))
    device = select_device(str(training_cfg.get("device", "cuda")))
    encoder, decoder, autoencoder_model_cfg, autoencoder_checkpoint = load_autoencoder(config, device)

    model = create_model(config).to(device)
    latent_dim = int(config["flow_model"].get("latent_dim", 16))
    if latent_dim != int(autoencoder_model_cfg["latent_dim"]):
        raise ValueError("flow_model.latent_dim must match autoencoder latent_dim")
    model.apply(weights_init)
    optimizer = create_optimizer(model, training_cfg)

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
    total_steps = int(training_cfg["num_epochs"]) * max(1, len(dataloader))
    scheduler = create_lr_scheduler(optimizer, training_cfg, total_steps)

    start_epoch = 0
    global_step = 0
    if resume is not None:
        start_epoch, global_step = load_latent_flow_checkpoint(
            resume,
            model,
            optimizer,
            scheduler,
            map_location=device,
        )
        start_epoch += 1

    sample_dir = resolve_path(outputs_cfg["sample_dir"])
    checkpoint_dir = resolve_path(outputs_cfg["checkpoint_dir"])
    log_dir = resolve_path(outputs_cfg["log_dir"])
    fixed_noise = torch.randn(int(training_cfg["num_sample_images"]), latent_dim, device=device)
    sample_interval = int(training_cfg["sample_interval"])
    checkpoint_interval = int(training_cfg["checkpoint_interval"])
    batches_this_run = 0
    should_stop = False
    saved_current_step_sample = False
    last_epoch = start_epoch - 1

    model.train()
    for epoch in range(start_epoch, int(training_cfg["num_epochs"])):
        last_epoch = epoch
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{training_cfg['num_epochs']}")
        for real_images in progress:
            real_images = real_images.to(device)
            with torch.no_grad():
                encoded = encoder(real_images)
            loss, metrics = latent_flow_matching_loss(model, encoded, training_cfg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            batches_this_run += 1
            saved_current_step_sample = False
            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "loss": float(loss.detach().cpu()),
                "t_mean": float(metrics["t_mean"].cpu()),
                "latent_mean": float(metrics["latent_mean"].cpu()),
                "latent_std": float(metrics["latent_std"].cpu()),
                "latent_norm": float(metrics["latent_norm"].cpu()),
                "target_velocity_norm": float(metrics["target_velocity_norm"].cpu()),
                "predicted_velocity_norm": float(metrics["predicted_velocity_norm"].cpu()),
                "velocity_cosine_similarity": float(metrics["velocity_cosine_similarity"].cpu()),
                "sampler_solver": str(sampler_cfg.get("solver", "heun")),
                "sampler_num_steps": int(sampler_cfg.get("num_steps", 100)),
                "autoencoder_checkpoint": str(autoencoder_checkpoint),
            }
            append_log(log_dir / "train_log.csv", row)
            progress.set_postfix(loss=f"{row['loss']:.4f}", t=f"{row['t_mean']:.3f}")

            if global_step % sample_interval == 0:
                save_latent_flow_samples(
                    model,
                    decoder,
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
            save_latent_flow_checkpoint(
                checkpoint_dir / f"latent_flow_matching_epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
                autoencoder_checkpoint,
            )

        if should_stop:
            break

    if global_step > 0 and not saved_current_step_sample:
        save_latent_flow_samples(
            model,
            decoder,
            fixed_noise,
            sample_dir,
            global_step,
            int(training_cfg["num_sample_images"]),
            sampler_cfg,
        )

    save_latent_flow_checkpoint(
        checkpoint_dir / "latent_flow_matching_latest.pt",
        model,
        optimizer,
        scheduler,
        max(0, last_epoch),
        global_step,
        config,
        autoencoder_checkpoint,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 16D latent-space Flow Matching on frozen DFN autoencoder latents."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/latent_flow_matching_16.yaml"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_config(resolve_path(args.config)), resume=args.resume, max_batches=args.max_batches)
