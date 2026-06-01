import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def sample_angle(rng: np.random.Generator, orientation: str, kappa: float) -> float:
    if orientation == "von_mises":
        return float(rng.vonmises(mu=0.0, kappa=kappa) % np.pi)
    return float(rng.uniform(0.0, np.pi))


def sample_length(
    rng: np.random.Generator,
    min_length: float,
    max_length: float,
    distribution: str,
) -> float:
    if distribution == "power_law":
        raw = rng.pareto(a=2.5) + 1.0
        length = min_length * raw
    else:
        length = rng.lognormal(mean=np.log((min_length + max_length) / 3.0), sigma=0.55)
    return float(np.clip(length, min_length, max_length))


def make_dfn_sample(
    sample_id: int,
    image_size: int,
    rng: np.random.Generator,
    min_fractures: int,
    max_fractures: int,
    min_length: float,
    max_length: float,
    min_width: int,
    max_width: int,
    length_distribution: str,
    orientation: str,
    von_mises_kappa: float,
) -> tuple[np.ndarray, dict]:
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    num_fractures = int(rng.integers(min_fractures, max_fractures + 1))
    fractures = []

    for _ in range(num_fractures):
        center_x = float(rng.uniform(0, image_size - 1))
        center_y = float(rng.uniform(0, image_size - 1))
        length = sample_length(rng, min_length, max_length, length_distribution)
        angle = sample_angle(rng, orientation, von_mises_kappa)
        width = int(rng.integers(min_width, max_width + 1))

        dx = 0.5 * length * np.cos(angle)
        dy = 0.5 * length * np.sin(angle)
        x1 = int(round(center_x - dx))
        y1 = int(round(center_y - dy))
        x2 = int(round(center_x + dx))
        y2 = int(round(center_y + dy))
        cv2.line(image, (x1, y1), (x2, y2), color=255, thickness=width, lineType=cv2.LINE_AA)

        fractures.append(
            {
                "center_x": center_x,
                "center_y": center_y,
                "length": length,
                "angle": angle,
                "width": width,
            }
        )

    image = (image > 0).astype(np.uint8) * 255
    metadata = {
        "sample_id": sample_id,
        "image_size": image_size,
        "num_fractures": num_fractures,
        "fractures": fractures,
    }
    return image, metadata


def generate_dataset(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    metadata_dir = out_dir / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    min_length = args.min_length or args.image_size * 0.08
    max_length = args.max_length or args.image_size * 0.65

    for sample_id in tqdm(range(args.num_samples), desc="Generating DFN samples"):
        image, metadata = make_dfn_sample(
            sample_id=sample_id,
            image_size=args.image_size,
            rng=rng,
            min_fractures=args.min_fractures,
            max_fractures=args.max_fractures,
            min_length=min_length,
            max_length=max_length,
            min_width=args.min_width,
            max_width=args.max_width,
            length_distribution=args.length_distribution,
            orientation=args.orientation,
            von_mises_kappa=args.von_mises_kappa,
        )
        stem = f"dfn_{sample_id:06d}"
        cv2.imwrite(str(image_dir / f"{stem}.png"), image)
        with (metadata_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic 2D DFN PNG dataset.")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--out_dir", type=Path, default=Path("data/synthetic_dfn_128"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_fractures", type=int, default=20)
    parser.add_argument("--max_fractures", type=int, default=80)
    parser.add_argument("--min_length", type=float, default=None)
    parser.add_argument("--max_length", type=float, default=None)
    parser.add_argument("--min_width", type=int, default=1)
    parser.add_argument("--max_width", type=int, default=2)
    parser.add_argument("--length_distribution", choices=("lognormal", "power_law"), default="lognormal")
    parser.add_argument("--orientation", choices=("uniform", "von_mises"), default="uniform")
    parser.add_argument("--von_mises_kappa", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    generate_dataset(parse_args())
