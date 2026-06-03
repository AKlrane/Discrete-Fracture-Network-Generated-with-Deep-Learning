import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.image_utils import save_image_grid


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    indices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(int(part))
    if not indices:
        raise ValueError("--indices was provided but no indices were parsed")
    return indices


def resolve_explicit_paths(values: list[Path] | None, image_dir: Path) -> list[Path] | None:
    if not values:
        return None

    resolved_paths = []
    for value in values:
        path = value if value.is_absolute() else resolve_path(value)
        if not path.exists():
            candidate = image_dir / value
            if candidate.exists():
                path = candidate
        if not path.exists():
            raise FileNotFoundError(f"Image path not found: {value}")
        resolved_paths.append(path)
    return resolved_paths


def select_image_paths(
    image_dir: Path,
    num_images: int,
    selection: str,
    seed: int,
    indices: list[int] | None,
    explicit_paths: list[Path] | None,
) -> list[Path]:
    if explicit_paths is not None:
        return explicit_paths

    paths = sorted(image_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    if indices is not None:
        selected = []
        for index in indices:
            if index < 0 or index >= len(paths):
                raise IndexError(f"Image index {index} is out of range for {len(paths)} images")
            selected.append(paths[index])
        return selected

    if num_images < 1:
        raise ValueError("--num_images must be a positive integer")
    count = min(num_images, len(paths))
    if selection == "first":
        return paths[:count]
    if selection == "last":
        return paths[-count:]
    if selection == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(len(paths), generator=generator).tolist()
        return [paths[index] for index in order[:count]]
    raise ValueError("--selection must be one of: random, first, last")


def load_images_as_tanh(paths: list[Path], image_size: int | None) -> torch.Tensor:
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("L")
            if image_size is not None:
                image = image.resize((image_size, image_size), Image.Resampling.NEAREST)
            data = torch.from_numpy(np.array(image, dtype=np.float32)).div(255.0)
            tensors.append(data.unsqueeze(0))
    probabilities = torch.stack(tensors, dim=0)
    return probabilities.mul(2.0).sub(1.0)


def default_output_path(image_dir: Path, out_dir: Path | None, out_prefix: str | None) -> Path:
    if out_dir is None:
        out_dir = PROJECT_ROOT / "outputs" / "data_grids"
    else:
        out_dir = resolve_path(out_dir)
    prefix = out_prefix or image_dir.parent.name or image_dir.name
    return out_dir / f"{prefix}.png"


def make_grid(args: argparse.Namespace) -> tuple[Path, Path]:
    config = load_config(resolve_path(args.config)) if args.config is not None else {}
    data_cfg = config.get("data", {})
    image_dir_value = args.image_dir or data_cfg.get("image_dir")
    if image_dir_value is None:
        raise ValueError("Provide --image_dir or a --config file with data.image_dir")
    image_dir = resolve_path(image_dir_value)
    image_size = args.image_size
    if image_size is None and "image_size" in data_cfg:
        image_size = int(data_cfg["image_size"])

    explicit_paths = resolve_explicit_paths(args.paths, image_dir)
    selected_paths = select_image_paths(
        image_dir=image_dir,
        num_images=args.num_images,
        selection=args.selection,
        seed=args.seed,
        indices=parse_indices(args.indices),
        explicit_paths=explicit_paths,
    )
    if image_size is None:
        with Image.open(selected_paths[0]) as image:
            image_size = int(image.size[0])

    images = load_images_as_tanh(selected_paths, image_size=image_size)
    nrow = args.nrow or int(math.sqrt(len(selected_paths))) or 1
    out_path = resolve_path(args.out_path) if args.out_path else default_output_path(
        image_dir,
        args.out_dir,
        args.out_prefix,
    )
    probability_path, binary_path = save_image_grid(
        images,
        out_path,
        nrow=nrow,
        threshold=float(args.threshold),
    )

    metadata_path = out_path.with_name(f"{out_path.stem}_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": str(resolve_path(args.config)) if args.config else None,
                "image_dir": str(image_dir),
                "image_size": image_size,
                "selection": args.selection,
                "seed": args.seed,
                "num_images": len(selected_paths),
                "nrow": nrow,
                "threshold": float(args.threshold),
                "probability_path": str(probability_path),
                "binary_path": str(binary_path),
                "selected_paths": [str(path) for path in selected_paths],
            },
            handle,
            indent=2,
        )

    print(f"Wrote probability grid to {probability_path}")
    print(f"Wrote binary grid to {binary_path}")
    print(f"Wrote metadata to {metadata_path}")
    return probability_path, binary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build training-style probability/binary grids from dataset PNG images."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional config with data.image_dir and data.image_size.")
    parser.add_argument("--image_dir", type=Path, default=None, help="Directory containing PNG images.")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--out_path", type=Path, default=None)
    parser.add_argument("--out_prefix", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=64)
    parser.add_argument("--selection", choices=("random", "first", "last"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated zero-based indices into sorted PNG files.")
    parser.add_argument("--paths", nargs="*", type=Path, default=None, help="Explicit image paths or filenames under image_dir.")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--nrow", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    make_grid(parse_args())
