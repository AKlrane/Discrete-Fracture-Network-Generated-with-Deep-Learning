from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import resolve_path


@dataclass
class PressureObservations:
    points: np.ndarray
    pressures: np.ndarray


def regular_observation_grid(config: dict[str, Any]) -> np.ndarray:
    grid_cfg = config.get("grid", {})
    nx = int(grid_cfg.get("nx", 7))
    ny = int(grid_cfg.get("ny", 7))
    x_values = np.linspace(float(grid_cfg.get("x_min", 2.0)), float(grid_cfg.get("x_max", 18.0)), nx)
    y_values = np.linspace(float(grid_cfg.get("y_min", 2.0)), float(grid_cfg.get("y_max", 18.0)), ny)
    return np.array([(x, y) for y in y_values for x in x_values], dtype=np.float64)


def load_observation_points(config: dict[str, Any]) -> np.ndarray:
    points_csv = config.get("points_csv")
    if points_csv:
        rows: list[tuple[float, float]] = []
        with resolve_path(points_csv).open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append((float(row["x"]), float(row["y"])))
        if not rows:
            raise ValueError(f"No observation points found in {points_csv}")
        return np.asarray(rows, dtype=np.float64)
    return regular_observation_grid(config)


def load_pressure_observations(path: str | Path) -> PressureObservations:
    rows: list[tuple[float, float, float]] = []
    with resolve_path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Pressure observation CSV has no header: {path}")
        pressure_key = "pressure" if "pressure" in reader.fieldnames else reader.fieldnames[-1]
        for row in reader:
            rows.append((float(row["x"]), float(row["y"]), float(row[pressure_key])))
    if not rows:
        raise ValueError(f"No pressure observations found in {path}")
    array = np.asarray(rows, dtype=np.float64)
    return PressureObservations(points=array[:, :2], pressures=array[:, 2])


def write_pressure_observations(
    path: str | Path,
    points: np.ndarray,
    pressures: np.ndarray,
    clean_pressures: np.ndarray | None = None,
) -> Path:
    out_path = resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["x", "y", "pressure"]
    if clean_pressures is not None:
        fieldnames.append("clean_pressure")
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, point in enumerate(points):
            row: dict[str, float] = {
                "x": float(point[0]),
                "y": float(point[1]),
                "pressure": float(pressures[index]),
            }
            if clean_pressures is not None:
                row["clean_pressure"] = float(clean_pressures[index])
            writer.writerow(row)
    return out_path


class LatentPosterior:
    def __init__(
        self,
        forward_model: Any,
        observations: PressureObservations,
        sigma: float,
        prior_scale: float = 1.0,
        failure_log_prob: float = -1.0e100,
    ) -> None:
        self.forward_model = forward_model
        self.observations = observations
        self.sigma = float(sigma)
        self.prior_scale = float(prior_scale)
        self.failure_log_prob = float(failure_log_prob)
        self.failures: list[str] = []
        if self.sigma <= 0.0:
            raise ValueError("noise sigma must be positive")

    def _log_prob_from_pressures(self, z: np.ndarray, pressures: np.ndarray) -> float:
        log_prior = -0.5 * float(np.sum((z / self.prior_scale) ** 2))
        if pressures.shape != self.observations.pressures.shape:
            self.failures.append(
                f"pressure shape mismatch: {pressures.shape} vs {self.observations.pressures.shape}"
            )
            return self.failure_log_prob
        residual = (pressures - self.observations.pressures) / self.sigma
        if not np.all(np.isfinite(residual)):
            return self.failure_log_prob
        return log_prior - 0.5 * float(np.sum(residual**2))

    def log_prob(self, z: np.ndarray) -> float:
        z_array = np.asarray(z, dtype=np.float64)
        if not np.all(np.isfinite(z_array)):
            return self.failure_log_prob
        try:
            result = self.forward_model.simulate(z_array)
        except Exception as exc:
            self.failures.append(str(exc))
            return self.failure_log_prob
        return self._log_prob_from_pressures(z_array, result.pressures)

    def evaluate(self, z: np.ndarray) -> dict[str, Any]:
        z_array = np.asarray(z, dtype=np.float64)
        result = self.forward_model.simulate(z_array)
        residual = result.pressures - self.observations.pressures
        rmse = float(np.sqrt(np.mean(residual**2)))
        return {
            "log_prob": self._log_prob_from_pressures(z_array, result.pressures),
            "rmse": rmse,
            "pressures": result.pressures,
            "segments": result.segments,
            "binary": result.binary,
            "probability": result.probability,
            "metadata": result.metadata,
        }
