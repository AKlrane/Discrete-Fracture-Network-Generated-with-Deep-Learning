from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inversion.common import load_config, resolve_path, write_json
from src.inversion.forward_model import create_forward_model
from src.inversion.fracture_extract import write_segments_csv
from src.inversion.latent_prior import WGANLatentPrior, save_binary_image, save_probability_image
from src.inversion.likelihood import (
    load_observation_points,
    write_pressure_observations,
)


def write_pressure_values(path: Path, points: np.ndarray, pressures: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "pressure"])
        writer.writeheader()
        for point, pressure in zip(points, pressures):
            writer.writerow({"x": float(point[0]), "y": float(point[1]), "pressure": float(pressure)})


def make_case(config: dict, output_dir: Path | None = None) -> None:
    prior = WGANLatentPrior.from_config(config["prior"])
    obs_cfg = config["observations"]
    points = load_observation_points(obs_cfg)
    forward_model = create_forward_model(config, prior, points)
    sigma = float(obs_cfg.get("noise_sigma", 0.05))
    seed = int(config.get("synthetic_reference", {}).get("seed", config.get("sampler", {}).get("seed", 42)))
    z_scale = float(config.get("synthetic_reference", {}).get("z_scale", 1.0))
    rng = np.random.default_rng(seed)
    z_ref = rng.normal(0.0, z_scale, size=int(config["prior"]["latent_dim"]))

    result = forward_model.simulate(z_ref)
    noisy_pressures = result.pressures + rng.normal(0.0, sigma, size=result.pressures.shape)
    pressure_csv = write_pressure_observations(
        obs_cfg["pressure_csv"],
        points,
        noisy_pressures,
        clean_pressures=result.pressures,
    )

    out_dir = output_dir or resolve_path(config["outputs"]["out_dir"]) / "synthetic_reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "z_ref.npy", z_ref)
    write_pressure_values(out_dir / "clean_pressure.csv", points, result.pressures)
    write_pressure_values(out_dir / "noisy_pressure.csv", points, noisy_pressures)
    write_segments_csv(out_dir / "reference_segments.csv", result.segments)
    if result.binary is not None:
        save_binary_image(result.binary, out_dir / "reference_binary.png")
    if result.probability is not None:
        save_probability_image(result.probability, out_dir / "reference_probability.png")
    write_json(
        out_dir / "summary.json",
        {
            "pressure_csv": str(pressure_csv),
            "noise_sigma": sigma,
            "latent_dim": int(config["prior"]["latent_dim"]),
            "reference_z_path": str(out_dir / "z_ref.npy"),
            "segment_count": len(result.segments),
            "forward_metadata": result.metadata,
        },
    )
    print(f"Wrote synthetic pressure observations to {pressure_csv}")
    print(f"Wrote reference artifacts to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic pressure inversion case from a WGAN latent prior.")
    parser.add_argument("--config", type=Path, default=Path("configs/inversion/teng_pressure_ld16.yaml"))
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_case(load_config(args.config), output_dir=args.output_dir)
