import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_latent_flow_matching import create_model, load_autoencoder, sample_latent_flow
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


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dictionary, got {type(checkpoint).__name__}")
    return checkpoint


def state_dict_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("Checkpoint must contain either 'model', 'state_dict', or a raw state_dict")


def strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def normalize_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    model_keys = set(model.state_dict().keys())
    candidates = [state_dict]
    for prefix in ("model.", "module.", "_orig_mod.", "latent_flow.", "model._orig_mod."):
        candidates.append(strip_prefix(state_dict, prefix))
    return max(candidates, key=lambda candidate: len(model_keys.intersection(candidate.keys())))


def checkpoint_step(checkpoint: dict[str, Any]) -> int:
    for key in ("step", "global_step"):
        if key in checkpoint:
            return int(checkpoint[key])
    return 0


def default_output_path(
    config: dict[str, Any],
    out_dir: Path | None,
    out_prefix: str | None,
    step: int,
    solver: str,
    num_steps: int,
) -> Path:
    if out_dir is None:
        out_dir = resolve_path(config["outputs"]["sample_dir"]) / "resampled"
    else:
        out_dir = resolve_path(out_dir)
    if out_prefix is None:
        step_label = f"step_{step:07d}" if step > 0 else "step_unknown"
        out_prefix = f"{step_label}_{solver}_{num_steps:04d}"
    return out_dir / f"{out_prefix}.png"


def save_individual_images(
    images: torch.Tensor,
    out_path: str | Path,
    threshold: float = 0.0,
) -> tuple[Path, Path]:
    out_path = Path(out_path)
    probability_dir = out_path.with_name(f"{out_path.stem}_prob")
    binary_dir = out_path.with_name(f"{out_path.stem}_binary")
    probability_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)

    probability = ((images.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    binary = (images.detach().cpu() > threshold).float()
    for index, (probability_image, binary_image) in enumerate(zip(probability, binary)):
        save_image(probability_image, probability_dir / f"sample_{index:06d}.png", normalize=False)
        save_image(binary_image, binary_dir / f"sample_{index:06d}.png", normalize=False)
    return probability_dir, binary_dir


@torch.no_grad()
def sample_in_batches(
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    num_images: int,
    batch_size: int,
    latent_dim: int,
    device: torch.device,
    solver: str,
    num_steps: int,
    seed: int,
) -> torch.Tensor:
    if num_images < 1:
        raise ValueError("--num_images must be a positive integer")
    if batch_size < 1:
        raise ValueError("--batch_size must be a positive integer")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    samples = []
    remaining = num_images
    model.eval()
    decoder.eval()
    while remaining > 0:
        current_batch_size = min(batch_size, remaining)
        noise = torch.randn(
            current_batch_size,
            latent_dim,
            generator=generator,
            device="cpu",
        ).to(device)
        latents = sample_latent_flow(
            model,
            noise,
            solver=solver,
            num_steps=num_steps,
        )
        samples.append(decoder(latents).cpu())
        remaining -= current_batch_size
    return torch.cat(samples, dim=0)


def sample_from_checkpoint(args: argparse.Namespace) -> tuple[Path, Path]:
    config = load_config(resolve_path(args.config))
    if args.autoencoder_checkpoint is not None:
        config.setdefault("autoencoder", {})["checkpoint"] = str(args.autoencoder_checkpoint)

    sampler_cfg = config.get("sampler", {})
    training_cfg = config.get("training", {})
    flow_model_cfg = config["flow_model"]
    solver = args.solver or str(sampler_cfg.get("solver", "heun"))
    num_steps = args.num_steps if args.num_steps is not None else int(sampler_cfg.get("num_steps", 100))
    seed = args.seed if args.seed is not None else int(training_cfg.get("seed", 42))
    device = select_device(args.device or str(training_cfg.get("device", "auto")))
    num_images = args.num_images if args.num_images is not None else int(training_cfg.get("num_sample_images", 64))
    batch_size = args.batch_size if args.batch_size is not None else num_images
    save_mode = args.save_mode or str(sampler_cfg.get("save_mode", "grid"))
    if save_mode not in {"grid", "individual", "both"}:
        raise ValueError("--save_mode must be one of: grid, individual, both")

    set_seed(seed)
    _, decoder, autoencoder_model_cfg, autoencoder_checkpoint = load_autoencoder(config, device)
    latent_dim = int(flow_model_cfg.get("latent_dim", 16))
    if latent_dim != int(autoencoder_model_cfg["latent_dim"]):
        raise ValueError("flow_model.latent_dim must match autoencoder latent_dim")

    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    model = create_model(config).to(device)
    state_dict = normalize_state_dict(state_dict_from_checkpoint(checkpoint), model)
    model.load_state_dict(state_dict)

    step = checkpoint_step(checkpoint)
    out_path = args.out_path
    if out_path is None:
        out_path = default_output_path(config, args.out_dir, args.out_prefix, step, solver, num_steps)
    else:
        out_path = resolve_path(out_path)
    nrow = args.nrow or int(math.sqrt(num_images))

    samples = sample_in_batches(
        model,
        decoder,
        num_images=num_images,
        batch_size=batch_size,
        latent_dim=latent_dim,
        device=device,
        solver=solver,
        num_steps=num_steps,
        seed=seed,
    )

    probability_path: Path | None = None
    binary_path: Path | None = None
    probability_dir: Path | None = None
    binary_dir: Path | None = None
    if save_mode in {"grid", "both"}:
        probability_path, binary_path = save_image_grid(
            samples,
            out_path,
            nrow=nrow,
            threshold=float(args.threshold),
        )
    if save_mode in {"individual", "both"}:
        probability_dir, binary_dir = save_individual_images(
            samples,
            out_path,
            threshold=float(args.threshold),
        )

    metadata_path = out_path.with_name(f"{out_path.stem}_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": str(resolve_path(args.config)),
                "checkpoint": str(checkpoint_path),
                "checkpoint_step": step,
                "autoencoder_checkpoint": str(autoencoder_checkpoint),
                "device": str(device),
                "num_images": num_images,
                "batch_size": batch_size,
                "latent_dim": latent_dim,
                "seed": seed,
                "solver": solver,
                "num_steps": num_steps,
                "save_mode": save_mode,
                "threshold": float(args.threshold),
                "probability_path": str(probability_path) if probability_path is not None else None,
                "binary_path": str(binary_path) if binary_path is not None else None,
                "probability_dir": str(probability_dir) if probability_dir is not None else None,
                "binary_dir": str(binary_dir) if binary_dir is not None else None,
            },
            handle,
            indent=2,
        )

    if probability_path is not None and binary_path is not None:
        print(f"Wrote probability grid to {probability_path}")
        print(f"Wrote binary grid to {binary_path}")
    if probability_dir is not None and binary_dir is not None:
        print(f"Wrote individual probability images to {probability_dir}")
        print(f"Wrote individual binary images to {binary_dir}")
    print(f"Wrote metadata to {metadata_path}")
    return (
        probability_path if probability_path is not None else probability_dir,
        binary_path if binary_path is not None else binary_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample DFN grids from a trained 16D latent-space Flow Matching checkpoint."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/latent_flow_matching_16.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--autoencoder_checkpoint", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--out_path", type=Path, default=None)
    parser.add_argument("--out_prefix", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--solver", choices=("euler", "heun", "midpoint"), default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument(
        "--save_mode",
        choices=("grid", "individual", "both"),
        default=None,
        help="Save output as a grid, individual images, or both. Defaults to sampler.save_mode or grid.",
    )
    parser.add_argument("--nrow", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    sample_from_checkpoint(parse_args())
