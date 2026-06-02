import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def sample_angle(
    rng: np.random.Generator,
    orientation: str,
    mean_degrees: float,
    kappa: float,
) -> float:
    if orientation == "von_mises":
        mean_radians = np.deg2rad(mean_degrees)
        return float(rng.vonmises(mu=mean_radians, kappa=kappa) % np.pi)
    return float(rng.uniform(0.0, np.pi))


def sample_length(
    rng: np.random.Generator,
    min_length: float,
    max_length: float,
    distribution: str,
    power_law_exponent: float,
) -> float:
    if distribution == "power_law":
        if min_length <= 0 or max_length <= min_length:
            raise ValueError("Power-law length sampling requires 0 < min_length < max_length")
        uniform = float(rng.random())
        if np.isclose(power_law_exponent, 1.0):
            length = min_length * (max_length / min_length) ** uniform
        else:
            exponent = 1.0 - power_law_exponent
            lower = min_length**exponent
            upper = max_length**exponent
            length = (lower + uniform * (upper - lower)) ** (1.0 / exponent)
    else:
        length = rng.lognormal(mean=np.log((min_length + max_length) / 3.0), sigma=0.55)
    return float(np.clip(length, min_length, max_length))


def cascade_probabilities(fractal_dimension: float) -> np.ndarray:
    if not 0.0 < fractal_dimension <= 2.0:
        raise ValueError("fractal_dimension must be in the interval (0, 2]")

    target_sum_squares = 2.0 ** (-fractal_dimension)
    # Split the square into four quadrants. A symmetric one-heavy distribution
    # gives sum(p_i^2) = 2^-Dc while keeping all quadrants reachable.
    dominant_probability = (1.0 + np.sqrt(max(0.0, 12.0 * target_sum_squares - 3.0))) / 4.0
    residual_probability = (1.0 - dominant_probability) / 3.0
    return np.array(
        [dominant_probability, residual_probability, residual_probability, residual_probability],
        dtype=np.float64,
    )


def sample_fractal_center(
    rng: np.random.Generator,
    image_size: int,
    fractal_dimension: float,
    fractal_levels: int,
    cascade_permutations: list[np.ndarray],
) -> tuple[float, float]:
    if fractal_levels < 1:
        raise ValueError("fractal_levels must be at least 1")

    probabilities = cascade_probabilities(fractal_dimension)
    x_min = 0.0
    y_min = 0.0
    width = float(image_size - 1)
    height = float(image_size - 1)

    for level in range(fractal_levels):
        quadrant_probabilities = probabilities[cascade_permutations[level]]

        quadrant = int(rng.choice(4, p=quadrant_probabilities))
        half_width = width * 0.5
        half_height = height * 0.5
        if quadrant % 2 == 1:
            x_min += half_width
        if quadrant >= 2:
            y_min += half_height
        width = half_width
        height = half_height

    center_x = float(rng.uniform(x_min, x_min + width))
    center_y = float(rng.uniform(y_min, y_min + height))
    return center_x, center_y


def sample_center(
    rng: np.random.Generator,
    image_size: int,
    position_distribution: str,
    fractal_dimension: float,
    fractal_levels: int,
    cascade_permutations: list[np.ndarray],
) -> tuple[float, float]:
    if position_distribution == "fractal":
        return sample_fractal_center(
            rng,
            image_size=image_size,
            fractal_dimension=fractal_dimension,
            fractal_levels=fractal_levels,
            cascade_permutations=cascade_permutations,
        )
    return float(rng.uniform(0, image_size - 1)), float(rng.uniform(0, image_size - 1))


def make_cascade_permutations(
    rng: np.random.Generator,
    fractal_levels: int,
    randomize_cascade_orientation: bool,
) -> list[np.ndarray]:
    permutations = []
    for _ in range(fractal_levels):
        permutation = np.arange(4)
        if randomize_cascade_orientation:
            rng.shuffle(permutation)
        permutations.append(permutation)
    return permutations


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
    position_distribution: str,
    fractal_dimension: float,
    fractal_levels: int,
    randomize_cascade_orientation: bool,
    length_distribution: str,
    power_law_exponent: float,
    orientation: str,
    von_mises_mean_degrees: float,
    von_mises_kappa: float,
) -> tuple[np.ndarray, dict]:
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    num_fractures = int(rng.integers(min_fractures, max_fractures + 1))
    fractures = []
    cascade_permutations = make_cascade_permutations(
        rng,
        fractal_levels=fractal_levels,
        randomize_cascade_orientation=randomize_cascade_orientation,
    )

    for _ in range(num_fractures):
        center_x, center_y = sample_center(
            rng,
            image_size=image_size,
            position_distribution=position_distribution,
            fractal_dimension=fractal_dimension,
            fractal_levels=fractal_levels,
            cascade_permutations=cascade_permutations,
        )
        length = sample_length(rng, min_length, max_length, length_distribution, power_law_exponent)
        angle = sample_angle(rng, orientation, von_mises_mean_degrees, von_mises_kappa)
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
        "position_distribution": position_distribution,
        "fractal_dimension": fractal_dimension if position_distribution == "fractal" else None,
        "fractal_levels": fractal_levels if position_distribution == "fractal" else None,
        "length_distribution": length_distribution,
        "power_law_exponent": power_law_exponent if length_distribution == "power_law" else None,
        "orientation": orientation,
        "von_mises_mean_degrees": von_mises_mean_degrees if orientation == "von_mises" else None,
        "von_mises_kappa": von_mises_kappa if orientation == "von_mises" else None,
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
            position_distribution=args.position_distribution,
            fractal_dimension=args.fractal_dimension,
            fractal_levels=args.fractal_levels,
            randomize_cascade_orientation=not args.fixed_cascade_orientation,
            length_distribution=args.length_distribution,
            power_law_exponent=args.power_law_exponent,
            orientation=args.orientation,
            von_mises_mean_degrees=args.von_mises_mean_degrees,
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
    parser.add_argument("--position_distribution", choices=("uniform", "fractal"), default="uniform")
    parser.add_argument("--fractal_dimension", type=float, default=2.0)
    parser.add_argument("--fractal_levels", type=int, default=6)
    parser.add_argument(
        "--fixed_cascade_orientation",
        action="store_true",
        help="Keep the dominant quadrant fixed across cascade levels instead of randomizing it.",
    )
    parser.add_argument("--length_distribution", choices=("lognormal", "power_law"), default="lognormal")
    parser.add_argument("--power_law_exponent", type=float, default=2.5)
    parser.add_argument("--orientation", choices=("uniform", "von_mises"), default="uniform")
    parser.add_argument("--von_mises_mean_degrees", type=float, default=0.0)
    parser.add_argument("--von_mises_kappa", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    generate_dataset(parse_args())
