from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import format_command_arg, read_point, resolve_path
from .fracture_extract import (
    FractureSegment,
    extract_fracture_segments,
    point_to_segment_distance,
    write_segments_csv,
)
from .latent_prior import WGANLatentPrior, save_binary_image, save_probability_image


class ForwardModelError(RuntimeError):
    pass


@dataclass
class SimulationResult:
    pressures: np.ndarray
    segments: list[FractureSegment]
    binary: np.ndarray | None
    probability: np.ndarray | None
    metadata: dict[str, Any]


class MockPressureForwardModel:
    """Deterministic pressure surrogate for testing the latent inversion loop."""

    def __init__(
        self,
        prior: WGANLatentPrior,
        observation_points: np.ndarray,
        config: dict[str, Any],
    ) -> None:
        self.prior = prior
        self.observation_points = np.asarray(observation_points, dtype=np.float64)
        self.domain_width = float(config.get("geometry", {}).get("domain_width", 20.0))
        self.domain_height = float(config.get("geometry", {}).get("domain_height", 20.0))
        self.extraction_cfg = config.get("geometry", {}).get("extraction", {})
        wells_cfg = config.get("wells", {})
        self.injection_xy = np.array(read_point(wells_cfg.get("injection", {"x": 2.0, "y": 10.0}), "injection"))
        self.production_xy = np.array(read_point(wells_cfg.get("production", {"x": 18.0, "y": 10.0}), "production"))
        self.injection_pressure = float(wells_cfg.get("injection", {}).get("pressure", 1.0))
        self.production_pressure = float(wells_cfg.get("production", {}).get("pressure", 0.0))
        mock_cfg = config.get("forward", {}).get("mock", {})
        self.fracture_influence = float(mock_cfg.get("fracture_influence", 0.18))
        self.decay_length = float(mock_cfg.get("decay_length", 1.5))
        self.min_pressure = float(mock_cfg.get("min_pressure", -5.0))
        self.max_pressure = float(mock_cfg.get("max_pressure", 5.0))

    def _extract(self, binary: np.ndarray) -> list[FractureSegment]:
        return extract_fracture_segments(
            binary=binary,
            domain_width=self.domain_width,
            domain_height=self.domain_height,
            hough_threshold=int(self.extraction_cfg.get("hough_threshold", 12)),
            min_line_length=int(self.extraction_cfg.get("min_line_length", 6)),
            max_line_gap=int(self.extraction_cfg.get("max_line_gap", 3)),
            invert_y=bool(self.extraction_cfg.get("invert_y", True)),
        )

    def _base_pressure(self) -> tuple[np.ndarray, np.ndarray]:
        points = self.observation_points
        flow_axis = self.production_xy - self.injection_xy
        axis_norm_sq = max(float(np.dot(flow_axis, flow_axis)), 1e-12)
        t = np.clip(((points - self.injection_xy) @ flow_axis) / axis_norm_sq, 0.0, 1.0)
        pressure = self.injection_pressure + (self.production_pressure - self.injection_pressure) * t
        return pressure, t

    def simulate(self, z: np.ndarray) -> SimulationResult:
        generated = self.prior.generate_one(z)
        segments = self._extract(generated.binary)
        pressures, t = self._base_pressure()
        flow_axis = self.production_xy - self.injection_xy
        flow_norm = max(float(np.linalg.norm(flow_axis)), 1e-12)
        flow_direction = flow_axis / flow_norm

        for segment in segments:
            distances = point_to_segment_distance(self.observation_points, segment)
            segment_vec = np.array([segment.x2 - segment.x1, segment.y2 - segment.y1], dtype=np.float64)
            segment_norm = max(float(np.linalg.norm(segment_vec)), 1e-12)
            alignment = abs(float(np.dot(segment_vec / segment_norm, flow_direction)))
            length_weight = segment.length / max(self.domain_width, self.domain_height)
            influence = self.fracture_influence * alignment * length_weight * np.exp(-distances / self.decay_length)
            pressures += influence * (0.5 - t)

        pressures = np.clip(pressures, self.min_pressure, self.max_pressure)
        return SimulationResult(
            pressures=pressures.astype(np.float64),
            segments=segments,
            binary=generated.binary,
            probability=generated.probability,
            metadata={"backend": "mock", "segment_count": len(segments)},
        )


class GEOSForwardModel:
    """Adapter around an external GEOS/GEOSX-style executable and template directory."""

    def __init__(
        self,
        prior: WGANLatentPrior,
        observation_points: np.ndarray,
        config: dict[str, Any],
    ) -> None:
        self.prior = prior
        self.observation_points = np.asarray(observation_points, dtype=np.float64)
        self.domain_width = float(config.get("geometry", {}).get("domain_width", 20.0))
        self.domain_height = float(config.get("geometry", {}).get("domain_height", 20.0))
        self.extraction_cfg = config.get("geometry", {}).get("extraction", {})
        self.geos_cfg = config.get("forward", {}).get("geos", {})
        executable = str(self.geos_cfg.get("executable", "")).strip()
        if not executable:
            raise ForwardModelError("forward.geos.executable is required for backend=geos")
        self.executable = resolve_path(executable) if "/" in executable else Path(executable)
        self.template_dir = self.geos_cfg.get("template_dir")
        self.work_dir = resolve_path(self.geos_cfg.get("work_dir", "outputs/inversion/geos_runs"))
        self.output_pressure_csv = str(self.geos_cfg.get("output_pressure_csv", "pressure_observations.csv"))
        self.timeout_seconds = int(self.geos_cfg.get("timeout_seconds", 600))
        self.arguments = [str(arg) for arg in self.geos_cfg.get("arguments", [])]

    def _extract(self, binary: np.ndarray) -> list[FractureSegment]:
        return extract_fracture_segments(
            binary=binary,
            domain_width=self.domain_width,
            domain_height=self.domain_height,
            hough_threshold=int(self.extraction_cfg.get("hough_threshold", 12)),
            min_line_length=int(self.extraction_cfg.get("min_line_length", 6)),
            max_line_gap=int(self.extraction_cfg.get("max_line_gap", 3)),
            invert_y=bool(self.extraction_cfg.get("invert_y", True)),
        )

    def _run_dir(self, z: np.ndarray) -> Path:
        digest = hashlib.sha1(np.asarray(z, dtype=np.float32).tobytes()).hexdigest()[:16]
        return self.work_dir / digest

    def _read_pressures(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise ForwardModelError(f"GEOS pressure output not found: {path}")
        values: list[float] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ForwardModelError(f"GEOS pressure CSV has no header: {path}")
            pressure_key = "pressure" if "pressure" in reader.fieldnames else reader.fieldnames[-1]
            for row in reader:
                values.append(float(row[pressure_key]))
        pressures = np.asarray(values, dtype=np.float64)
        if pressures.size != self.observation_points.shape[0]:
            raise ForwardModelError(
                f"Expected {self.observation_points.shape[0]} pressure values, got {pressures.size}"
            )
        return pressures

    def simulate(self, z: np.ndarray) -> SimulationResult:
        generated = self.prior.generate_one(z)
        segments = self._extract(generated.binary)
        run_dir = self._run_dir(np.asarray(z, dtype=np.float32))
        run_dir.mkdir(parents=True, exist_ok=True)
        if self.template_dir:
            template_dir = resolve_path(self.template_dir)
            if not template_dir.exists():
                raise ForwardModelError(f"GEOS template_dir not found: {template_dir}")
            shutil.copytree(template_dir, run_dir, dirs_exist_ok=True)

        segments_csv = write_segments_csv(run_dir / "dfn_segments.csv", segments)
        save_binary_image(generated.binary, run_dir / "dfn_binary.png")
        save_probability_image(generated.probability, run_dir / "dfn_probability.png")

        replacements = {
            "run_dir": str(run_dir),
            "segments_csv": str(segments_csv),
            "output_pressure_csv": str(run_dir / self.output_pressure_csv),
        }
        command = [str(self.executable)]
        command.extend(format_command_arg(arg, replacements) for arg in self.arguments)
        completed = subprocess.run(
            command,
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise ForwardModelError(
                f"GEOS command failed with code {completed.returncode}: {completed.stderr.strip()}"
            )

        pressures = self._read_pressures(run_dir / self.output_pressure_csv)
        return SimulationResult(
            pressures=pressures,
            segments=segments,
            binary=generated.binary,
            probability=generated.probability,
            metadata={
                "backend": "geos",
                "run_dir": str(run_dir),
                "segment_count": len(segments),
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            },
        )


def create_forward_model(
    config: dict[str, Any],
    prior: WGANLatentPrior,
    observation_points: np.ndarray,
) -> MockPressureForwardModel | GEOSForwardModel:
    backend = str(config.get("forward", {}).get("backend", "mock")).lower()
    if backend == "mock":
        return MockPressureForwardModel(prior, observation_points, config)
    if backend == "geos":
        return GEOSForwardModel(prior, observation_points, config)
    raise ValueError("forward.backend must be one of: mock, geos")
