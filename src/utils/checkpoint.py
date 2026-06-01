from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    generator: torch.nn.Module,
    critic: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_c: torch.optim.Optimizer,
    epoch: int,
    step: int,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_c": optimizer_c.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    generator: torch.nn.Module,
    critic: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer | None = None,
    optimizer_c: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, int]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    generator.load_state_dict(checkpoint["generator"])
    critic.load_state_dict(checkpoint["critic"])
    if optimizer_g is not None and "optimizer_g" in checkpoint:
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    if optimizer_c is not None and "optimizer_c" in checkpoint:
        optimizer_c.load_state_dict(checkpoint["optimizer_c"])
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0))
