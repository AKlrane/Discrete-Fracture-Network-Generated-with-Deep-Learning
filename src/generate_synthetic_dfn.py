import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OPTIONS: dict[str, Any] = {
    "num_samples": 10000,
    "image_size": 128,
    "out_dir": Path("data/synthetic_dfn_128"),
    "seed": 42,
    "min_fractures": 20,
    "max_fractures": 80,
    "min_length": None,
    "max_length": None,
    "min_width": 1,
    "max_width": 2,
    "position_distribution": "uniform",
    "fractal_dimension": 2.0,
    "fractal_levels": 6,
    "fixed_cascade_orientation": False,
    "length_distribution": "lognormal",
    "power_law_exponent": 2.5,
    "orientation": "uniform",
    "von_mises_mean_degrees": 0.0,
    "von_mises_kappa": 4.0,
    "unit_system": "pixel",
    "domain_width": 20.0,
    "domain_height": 20.0,
    "conditioning_mode": "none",
    "preexisting_count": None,
    "preexisting_injection_index": 0,
    "remove_isolated_fractures": True,
    "max_attempts": None,
    "prune_connected_fractures_to_target": False,
    "target_connected_fracture_count_min": None,
    "target_connected_fracture_count_max": None,
    "paper_case": None,
    "reference_stats": None,
}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config or {}


def apply_if_present(options: dict[str, Any], config: dict[str, Any], section: str, key: str, option: str) -> None:
    section_config = config.get(section, {})
    if isinstance(section_config, dict) and key in section_config:
        options[option] = section_config[key]


def options_from_config(config: dict[str, Any]) -> dict[str, Any]:
    options = DEFAULT_OPTIONS.copy()

    apply_if_present(options, config, "dataset", "num_samples", "num_samples")
    apply_if_present(options, config, "dataset", "image_size", "image_size")
    apply_if_present(options, config, "dataset", "seed", "seed")
    apply_if_present(options, config, "output", "out_dir", "out_dir")
    apply_if_present(options, config, "fractures", "min_fractures", "min_fractures")
    apply_if_present(options, config, "fractures", "max_fractures", "max_fractures")
    apply_if_present(options, config, "fractures", "min_width", "min_width")
    apply_if_present(options, config, "fractures", "max_width", "max_width")
    apply_if_present(options, config, "position", "distribution", "position_distribution")
    apply_if_present(options, config, "position", "fractal_dimension", "fractal_dimension")
    apply_if_present(options, config, "position", "fractal_levels", "fractal_levels")
    apply_if_present(options, config, "position", "fixed_cascade_orientation", "fixed_cascade_orientation")
    apply_if_present(options, config, "length", "distribution", "length_distribution")
    apply_if_present(options, config, "length", "min", "min_length")
    apply_if_present(options, config, "length", "max", "max_length")
    apply_if_present(options, config, "length", "power_law_exponent", "power_law_exponent")
    apply_if_present(options, config, "orientation", "distribution", "orientation")
    apply_if_present(options, config, "orientation", "von_mises_mean_degrees", "von_mises_mean_degrees")
    apply_if_present(options, config, "orientation", "von_mises_kappa", "von_mises_kappa")
    apply_if_present(options, config, "units", "system", "unit_system")
    apply_if_present(options, config, "units", "domain_width", "domain_width")
    apply_if_present(options, config, "units", "domain_height", "domain_height")

    conditioning = config.get("conditioning", {})
    if isinstance(conditioning, dict):
        options["conditioning_mode"] = conditioning.get("mode", options["conditioning_mode"])
        options["preexisting_count"] = conditioning.get("preexisting_count", options["preexisting_count"])
        options["preexisting_injection_index"] = conditioning.get(
            "injection_index",
            options["preexisting_injection_index"],
        )
        options["remove_isolated_fractures"] = conditioning.get(
            "remove_isolated_fractures",
            options["remove_isolated_fractures"],
        )
        options["max_attempts"] = conditioning.get("max_attempts", options["max_attempts"])
        options["prune_connected_fractures_to_target"] = conditioning.get(
            "prune_connected_fractures_to_target",
            options["prune_connected_fractures_to_target"],
        )
        fracture_count_target = conditioning.get("target_connected_fracture_count", {})
        if isinstance(fracture_count_target, dict):
            options["target_connected_fracture_count_min"] = fracture_count_target.get(
                "min",
                options["target_connected_fracture_count_min"],
            )
            options["target_connected_fracture_count_max"] = fracture_count_target.get(
                "max",
                options["target_connected_fracture_count_max"],
            )

    if "paper_case" in config:
        options["paper_case"] = config["paper_case"]
    if "reference_stats" in config:
        options["reference_stats"] = config["reference_stats"]

    return options


def x_scale(args: argparse.Namespace) -> float:
    if args.unit_system == "physical":
        return (args.image_size - 1) / float(args.domain_width)
    if args.unit_system == "normalized":
        return float(args.image_size - 1)
    return 1.0


def y_scale(args: argparse.Namespace) -> float:
    if args.unit_system == "physical":
        return (args.image_size - 1) / float(args.domain_height)
    if args.unit_system == "normalized":
        return float(args.image_size - 1)
    return 1.0


def length_scale(args: argparse.Namespace) -> float:
    return 0.5 * (x_scale(args) + y_scale(args))


def scale_length_arg(value: float | None, args: argparse.Namespace) -> float | None:
    if value is None:
        return None
    return float(value) * length_scale(args)


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


def line_endpoints(fracture: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
    center_x = float(fracture["center_x"])
    center_y = float(fracture["center_y"])
    length = float(fracture["length"])
    angle = float(fracture["angle"])
    dx = 0.5 * length * np.cos(angle)
    dy = 0.5 * length * np.sin(angle)
    return (
        (int(round(center_x - dx)), int(round(center_y - dy))),
        (int(round(center_x + dx)), int(round(center_y + dy))),
    )


def draw_fractures(fractures: list[dict[str, Any]], image_size: int) -> np.ndarray:
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    for fracture in fractures:
        p1, p2 = line_endpoints(fracture)
        cv2.line(
            image,
            p1,
            p2,
            color=255,
            thickness=int(fracture["width"]),
            lineType=cv2.LINE_AA,
        )
    return (image > 0).astype(np.uint8) * 255


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


def make_random_fractures(
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
) -> list[dict[str, Any]]:
    num_fractures = int(rng.integers(min_fractures, max_fractures + 1))
    fractures = []
    cascade_permutations = make_cascade_permutations(
        rng,
        fractal_levels=fractal_levels,
        randomize_cascade_orientation=randomize_cascade_orientation,
    )

    for fracture_index in range(num_fractures):
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

        fractures.append(
            {
                "id": f"random_{fracture_index:04d}",
                "role": "random",
                "center_x": center_x,
                "center_y": center_y,
                "length": length,
                "angle": angle,
                "angle_degrees": float(np.rad2deg(angle)),
                "width": width,
            }
        )
    return fractures


def assign_preexisting_roles(
    fractures: list[dict[str, Any]],
    injection_index: int,
) -> list[dict[str, Any]]:
    if len(fractures) < 2:
        raise ValueError("conditioning.preexisting_count must be at least 2")
    if not 0 <= injection_index < len(fractures):
        raise ValueError("conditioning.injection_index must point to an existing pre-existing fracture")

    assigned = []
    for index, fracture in enumerate(fractures):
        role = "injection" if index == injection_index else "monitoring"
        assigned_fracture = dict(fracture)
        assigned_fracture["id"] = f"{role}_{index}"
        assigned_fracture["role"] = role
        assigned_fracture["source"] = "sampled_preexisting"
        assigned.append(assigned_fracture)
    return assigned


def sample_preexisting_fractures(
    rng: np.random.Generator,
    args: argparse.Namespace,
    min_length: float,
    max_length: float,
) -> list[dict[str, Any]]:
    if args.preexisting_count is None:
        raise ValueError("conditioning.preexisting_count is required for preexisting_connectivity mode")

    preexisting_count = int(args.preexisting_count)
    candidate_fractures = make_random_fractures(
        image_size=args.image_size,
        rng=rng,
        min_fractures=preexisting_count,
        max_fractures=preexisting_count,
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
    return assign_preexisting_roles(
        candidate_fractures,
        injection_index=int(args.preexisting_injection_index),
    )


def dataset_metadata(
    sample_id: int,
    image_size: int,
    fractures: list[dict[str, Any]],
    args: argparse.Namespace,
    min_length_pixels: float,
    max_length_pixels: float,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "image_size": image_size,
        "num_fractures": len(fractures),
        "position_distribution": args.position_distribution,
        "fractal_dimension": args.fractal_dimension if args.position_distribution == "fractal" else None,
        "fractal_levels": args.fractal_levels if args.position_distribution == "fractal" else None,
        "length_distribution": args.length_distribution,
        "min_length_pixels": min_length_pixels,
        "max_length_pixels": max_length_pixels,
        "power_law_exponent": args.power_law_exponent if args.length_distribution == "power_law" else None,
        "orientation": args.orientation,
        "von_mises_mean_degrees": args.von_mises_mean_degrees if args.orientation == "von_mises" else None,
        "von_mises_kappa": args.von_mises_kappa if args.orientation == "von_mises" else None,
        "unit_system": args.unit_system,
        "domain_width": args.domain_width if args.unit_system == "physical" else None,
        "domain_height": args.domain_height if args.unit_system == "physical" else None,
        "conditioning_mode": args.conditioning_mode,
        "paper_case": args.paper_case,
        "reference_stats": args.reference_stats,
        "fractures": fractures,
    }


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
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    fractures = make_random_fractures(
        image_size=image_size,
        rng=rng,
        min_fractures=min_fractures,
        max_fractures=max_fractures,
        min_length=min_length,
        max_length=max_length,
        min_width=min_width,
        max_width=max_width,
        position_distribution=position_distribution,
        fractal_dimension=fractal_dimension,
        fractal_levels=fractal_levels,
        randomize_cascade_orientation=randomize_cascade_orientation,
        length_distribution=length_distribution,
        power_law_exponent=power_law_exponent,
        orientation=orientation,
        von_mises_mean_degrees=von_mises_mean_degrees,
        von_mises_kappa=von_mises_kappa,
    )

    image = draw_fractures(fractures, image_size)
    metadata = dataset_metadata(sample_id, image_size, fractures, args, min_length, max_length)
    return image, metadata


def fracture_component_labels(labels: np.ndarray, fracture: dict[str, Any], image_size: int) -> set[int]:
    mask = draw_fractures([fracture], image_size) > 0
    return {int(label) for label in np.unique(labels[mask]) if int(label) != 0}


def connectivity_status(
    fractures: list[dict[str, Any]],
    preexisting_fractures: list[dict[str, Any]],
    image_size: int,
) -> tuple[bool, dict[str, Any], np.ndarray]:
    binary = (draw_fractures(fractures, image_size) > 0).astype(np.uint8)
    _, labels = cv2.connectedComponents(binary, connectivity=8)

    injection = next(fracture for fracture in preexisting_fractures if fracture["role"] == "injection")
    injection_labels = fracture_component_labels(labels, injection, image_size)
    monitoring_results = []
    all_connected = bool(injection_labels)

    for fracture in preexisting_fractures:
        if fracture["role"] != "monitoring":
            continue
        labels_for_fracture = fracture_component_labels(labels, fracture, image_size)
        connected = bool(injection_labels & labels_for_fracture)
        all_connected = all_connected and connected
        monitoring_results.append(
            {
                "id": fracture["id"],
                "connected_to_injection": connected,
            }
        )

    return (
        all_connected,
        {
            "injection_id": injection["id"],
            "monitoring": monitoring_results,
        },
        labels,
    )


def remove_isolated_random_fractures(
    random_fractures: list[dict[str, Any]],
    preexisting_fractures: list[dict[str, Any]],
    labels: np.ndarray,
    image_size: int,
) -> list[dict[str, Any]]:
    retained_labels = set()
    for fracture in preexisting_fractures:
        retained_labels.update(fracture_component_labels(labels, fracture, image_size))

    retained_random = []
    for fracture in random_fractures:
        if fracture_component_labels(labels, fracture, image_size) & retained_labels:
            retained_random.append(fracture)
    return retained_random


def prune_random_fractures_to_target_count(
    random_fractures: list[dict[str, Any]],
    preexisting_fractures: list[dict[str, Any]],
    image_size: int,
    target_max_count: int | None,
) -> list[dict[str, Any]]:
    if target_max_count is None:
        return random_fractures

    pruned = list(random_fractures)
    while len(pruned) + len(preexisting_fractures) > target_max_count:
        removed = False
        for index in range(len(pruned) - 1, -1, -1):
            candidate = pruned[:index] + pruned[index + 1 :]
            connected, _, _ = connectivity_status(
                [*candidate, *preexisting_fractures],
                preexisting_fractures,
                image_size,
            )
            if connected:
                pruned = candidate
                removed = True
                break
        if not removed:
            break
    return pruned


def make_conditioned_dfn_sample(
    sample_id: int,
    attempt: int,
    image_size: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
    min_length: float,
    max_length: float,
    preexisting_fractures: list[dict[str, Any]],
) -> tuple[np.ndarray, dict] | None:
    random_fractures = make_random_fractures(
        image_size=image_size,
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
    fractures = [*random_fractures, *preexisting_fractures]
    connected, connectivity, labels = connectivity_status(fractures, preexisting_fractures, image_size)
    if not connected:
        return None

    retained_random = random_fractures
    if args.remove_isolated_fractures:
        retained_random = remove_isolated_random_fractures(
            random_fractures,
            preexisting_fractures,
            labels,
            image_size,
        )
    if args.prune_connected_fractures_to_target:
        retained_random = prune_random_fractures_to_target_count(
            retained_random,
            preexisting_fractures,
            image_size,
            args.target_connected_fracture_count_max,
        )
    final_fractures = [*retained_random, *preexisting_fractures]
    if (
        args.target_connected_fracture_count_min is not None
        and len(final_fractures) < args.target_connected_fracture_count_min
    ):
        return None
    if (
        args.target_connected_fracture_count_max is not None
        and len(final_fractures) > args.target_connected_fracture_count_max
    ):
        return None

    image = draw_fractures(final_fractures, image_size)
    metadata = dataset_metadata(sample_id, image_size, final_fractures, args, min_length, max_length)
    metadata["conditioning"] = {
        "mode": args.conditioning_mode,
        "attempt": attempt,
        "initial_random_fractures": len(random_fractures),
        "retained_random_fractures": len(retained_random),
        "num_preexisting_fractures": len(preexisting_fractures),
        "preexisting_source": "sampled_dataset",
        "preexisting_injection_index": args.preexisting_injection_index,
        "remove_isolated_fractures": args.remove_isolated_fractures,
        "prune_connected_fractures_to_target": args.prune_connected_fractures_to_target,
        "target_connected_fracture_count_min": args.target_connected_fracture_count_min,
        "target_connected_fracture_count_max": args.target_connected_fracture_count_max,
        "connectivity": connectivity,
    }
    return image, metadata


def generate_dataset(args: argparse.Namespace) -> None:
    out_dir = resolve_path(args.out_dir)
    image_dir = out_dir / "images"
    metadata_dir = out_dir / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    min_length = scale_length_arg(args.min_length, args) or args.image_size * 0.08
    max_length = scale_length_arg(args.max_length, args) or args.image_size * 0.65

    if args.conditioning_mode == "preexisting_connectivity":
        preexisting_fractures = sample_preexisting_fractures(
            rng,
            args,
            min_length=min_length,
            max_length=max_length,
        )
        max_attempts = args.max_attempts or max(args.num_samples * 1000, 1000)
        sample_id = 0
        attempt = 0
        progress = tqdm(total=args.num_samples, desc="Generating conditioned DFN samples")
        while sample_id < args.num_samples and attempt < max_attempts:
            attempt += 1
            result = make_conditioned_dfn_sample(
                sample_id=sample_id,
                attempt=attempt,
                image_size=args.image_size,
                rng=rng,
                args=args,
                min_length=min_length,
                max_length=max_length,
                preexisting_fractures=preexisting_fractures,
            )
            if result is None:
                continue
            image, metadata = result
            stem = f"dfn_{sample_id:06d}"
            cv2.imwrite(str(image_dir / f"{stem}.png"), image)
            with (metadata_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            sample_id += 1
            progress.update(1)
        progress.close()
        if sample_id < args.num_samples:
            raise RuntimeError(
                f"Only generated {sample_id} conditioned samples after {attempt} attempts. "
                "Relax connectivity constraints or increase conditioning.max_attempts."
            )
        return

    if args.conditioning_mode != "none":
        raise ValueError("conditioning_mode must be either 'none' or 'preexisting_connectivity'")

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
            args=args,
        )
        stem = f"dfn_{sample_id:06d}"
        cv2.imwrite(str(image_dir / f"{stem}.png"), image)
        with (metadata_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    defaults = DEFAULT_OPTIONS.copy()
    if pre_args.config is not None:
        defaults = options_from_config(load_config(pre_args.config))

    parser = argparse.ArgumentParser(description="Generate a synthetic 2D DFN PNG dataset.")
    parser.add_argument("--config", type=Path, default=pre_args.config)
    parser.add_argument("--num_samples", type=int, default=defaults["num_samples"])
    parser.add_argument("--image_size", type=int, default=defaults["image_size"])
    parser.add_argument("--out_dir", type=Path, default=Path(defaults["out_dir"]))
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--min_fractures", type=int, default=defaults["min_fractures"])
    parser.add_argument("--max_fractures", type=int, default=defaults["max_fractures"])
    parser.add_argument("--min_length", type=float, default=defaults["min_length"])
    parser.add_argument("--max_length", type=float, default=defaults["max_length"])
    parser.add_argument("--min_width", type=int, default=defaults["min_width"])
    parser.add_argument("--max_width", type=int, default=defaults["max_width"])
    parser.add_argument("--position_distribution", choices=("uniform", "fractal"), default=defaults["position_distribution"])
    parser.add_argument("--fractal_dimension", type=float, default=defaults["fractal_dimension"])
    parser.add_argument("--fractal_levels", type=int, default=defaults["fractal_levels"])
    parser.add_argument(
        "--fixed_cascade_orientation",
        action="store_true",
        default=defaults["fixed_cascade_orientation"],
        help="Keep the dominant quadrant fixed across cascade levels instead of randomizing it.",
    )
    parser.add_argument("--length_distribution", choices=("lognormal", "power_law"), default=defaults["length_distribution"])
    parser.add_argument("--power_law_exponent", type=float, default=defaults["power_law_exponent"])
    parser.add_argument("--orientation", choices=("uniform", "von_mises"), default=defaults["orientation"])
    parser.add_argument("--von_mises_mean_degrees", type=float, default=defaults["von_mises_mean_degrees"])
    parser.add_argument("--von_mises_kappa", type=float, default=defaults["von_mises_kappa"])
    parser.add_argument("--unit_system", choices=("pixel", "normalized", "physical"), default=defaults["unit_system"])
    parser.add_argument("--domain_width", type=float, default=defaults["domain_width"])
    parser.add_argument("--domain_height", type=float, default=defaults["domain_height"])
    parser.add_argument(
        "--conditioning_mode",
        choices=("none", "preexisting_connectivity"),
        default=defaults["conditioning_mode"],
    )
    parser.add_argument("--preexisting_count", type=int, default=defaults["preexisting_count"])
    parser.add_argument(
        "--preexisting_injection_index",
        type=int,
        default=defaults["preexisting_injection_index"],
    )
    parser.add_argument("--max_attempts", type=int, default=defaults["max_attempts"])
    parser.set_defaults(prune_connected_fractures_to_target=bool(defaults["prune_connected_fractures_to_target"]))
    pruning_group = parser.add_mutually_exclusive_group()
    pruning_group.add_argument(
        "--prune_connected_fractures_to_target",
        dest="prune_connected_fractures_to_target",
        action="store_true",
    )
    pruning_group.add_argument(
        "--no_prune_connected_fractures_to_target",
        dest="prune_connected_fractures_to_target",
        action="store_false",
    )
    parser.add_argument(
        "--target_connected_fracture_count_min",
        type=int,
        default=defaults["target_connected_fracture_count_min"],
    )
    parser.add_argument(
        "--target_connected_fracture_count_max",
        type=int,
        default=defaults["target_connected_fracture_count_max"],
    )
    parser.set_defaults(remove_isolated_fractures=bool(defaults["remove_isolated_fractures"]))
    isolation_group = parser.add_mutually_exclusive_group()
    isolation_group.add_argument(
        "--remove_isolated_fractures",
        dest="remove_isolated_fractures",
        action="store_true",
    )
    isolation_group.add_argument(
        "--keep_isolated_fractures",
        dest="remove_isolated_fractures",
        action="store_false",
    )
    args = parser.parse_args()
    args.paper_case = defaults["paper_case"]
    args.reference_stats = defaults["reference_stats"]
    return args


if __name__ == "__main__":
    generate_dataset(parse_args())
