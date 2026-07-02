from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path("/home/liuchaoran/geos_cases/min_edfm")
TEMPLATE = ROOT / "cases/base_xml/cross_fracture.xml"
GEOSX = Path("/home/liuchaoran/codes/build-geos-liuchaoran-edfm-release/bin/geosx")
EXPORT_PRESSURE = ROOT / "parametric_studies/export_pressure_with_visit.py"
VISIT_FALLBACK = Path("/home/liuchaoran/.local/bin/visit")


def fmt(values: tuple[float, ...]) -> str:
    return "{ " + ", ".join(f"{v:.8g}" for v in values) + " }"


def unit(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        raise ValueError(f"zero vector: {vec}")
    return tuple(v / norm for v in vec)


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def read_segments(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {key: float(row[key]) for key in ("x1", "y1", "x2", "y2", "length", "angle_degrees")}
            for row in csv.DictReader(handle)
        ]


def read_points(path: Path) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((float(row["x"]), float(row["y"])))
    if not rows:
        raise ValueError(f"no observation points found in {path}")
    return np.asarray(rows, dtype=np.float64)


def remove_template_fractures(geometry: ET.Element) -> None:
    for child in list(geometry):
        if child.tag == "Rectangle":
            geometry.remove(child)


def segment_to_rectangle(
    geometry: ET.Element,
    segment: dict[str, float],
    index: int,
    domain_width: float,
    domain_height: float,
    fracture_width: float,
    overlap: float,
) -> str | None:
    sx = np.clip(segment["x1"] / domain_width, 0.0, 1.0)
    sz = np.clip(segment["y1"] / domain_height, 0.0, 1.0)
    ex = np.clip(segment["x2"] / domain_width, 0.0, 1.0)
    ez = np.clip(segment["y2"] / domain_height, 0.0, 1.0)
    dx = float(ex - sx)
    dz = float(ez - sz)
    length = math.hypot(dx, dz)
    if length <= 1.0e-5:
        return None
    length_vector = unit((dx, 0.0, dz))
    width_vector = (0.0, 1.0, 0.0)
    normal = unit(cross(length_vector, width_vector))
    name = f"GeneratedFracture{index:04d}"
    ET.SubElement(
        geometry,
        "Rectangle",
        {
            "name": name,
            "normal": fmt(normal),
            "origin": fmt((0.5 * (sx + ex), 0.0, 0.5 * (sz + ez))),
            "lengthVector": fmt(length_vector),
            "widthVector": fmt(width_vector),
            "dimensions": fmt((length + 2.0 * overlap, fracture_width)),
        },
    )
    return name


def build_xml(run_dir: Path, request: dict, segments: list[dict[str, float]]) -> Path:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"GEOS template not found: {TEMPLATE}")
    tree = ET.parse(TEMPLATE)
    root = tree.getroot()
    geometry = root.find(".//Geometry")
    generator = root.find(".//EmbeddedSurfaceGenerator")
    if geometry is None or generator is None:
        raise RuntimeError("template missing Geometry or EmbeddedSurfaceGenerator")

    remove_template_fractures(geometry)
    domain_width = float(request.get("domain_width", 20.0))
    domain_height = float(request.get("domain_height", 20.0))
    material = request.get("material", {})
    fracture_width = float(material.get("fracture_width", 0.05))
    overlap = float(material.get("overlap", 0.01))

    names = []
    for index, segment in enumerate(segments):
        name = segment_to_rectangle(geometry, segment, index, domain_width, domain_height, fracture_width, overlap)
        if name is not None:
            names.append(name)
    if not names:
        raise RuntimeError("no valid fracture segments after coordinate conversion")
    generator.set("targetObjects", "{ " + ", ".join(names) + " }")

    aperture = material.get("aperture")
    if aperture is not None:
        region = root.find(".//SurfaceElementRegion[@name='Fracture']")
        if region is not None:
            region.set("defaultAperture", f"{float(aperture):.8g}")

    runtime = request.get("runtime", {})
    nonlinear = root.find(".//NonlinearSolverParameters")
    if nonlinear is not None and runtime.get("newton_max_iter") is not None:
        nonlinear.set("newtonMaxIter", str(int(runtime["newton_max_iter"])))
    linear = root.find(".//LinearSolverParameters")
    if linear is not None and runtime.get("krylov_max_iter") is not None:
        linear.set("krylovMaxIter", str(int(runtime["krylov_max_iter"])))
    events = root.find(".//Events")
    if events is not None and runtime.get("max_time") is not None:
        events.set("maxTime", f"{float(runtime['max_time']):.8g}")
    for event in root.findall(".//PeriodicEvent[@name='solverApplications']"):
        if runtime.get("force_dt") is not None:
            event.set("forceDt", f"{float(runtime['force_dt']):.8g}")
    silo = root.find(".//Silo")
    if silo is not None:
        silo.set("plotFileRoot", "candidate")

    xml_path = run_dir / "candidate.xml"
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path


def write_manifest(batch_dir: Path, case_dir: Path, xml_path: Path, segment_count: int) -> None:
    fields = ["case_id", "xml_path", "source", "fracture_count", "output_dir", "log_path", "status", "exit_code", "runtime_s"]
    row = {
        "case_id": "candidate",
        "xml_path": str(xml_path),
        "source": "generated",
        "fracture_count": str(segment_count),
        "output_dir": str(case_dir),
        "log_path": str(case_dir / "run.log"),
        "status": "ok",
        "exit_code": "0",
        "runtime_s": "",
    }
    with (batch_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def write_case_metadata(dataset_dir: Path) -> None:
    meta_dir = dataset_dir / "cases" / "candidate"
    meta_dir.mkdir(parents=True, exist_ok=True)
    payload = {"xml_features": {"mesh": {"nx": 61, "ny": 3, "nz": 61}}}
    (meta_dir / "case_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_checked(command: list[str], cwd: Path, log_path: Path | None = None, timeout: int | None = None) -> None:
    if log_path is None:
        completed = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout, check=False)
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        return
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=str(cwd), text=True, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
        raise RuntimeError(f"command failed: {' '.join(command)}\n{tail}")


def sample_pressure(npz_path: Path, observation_points: np.ndarray, domain_width: float, domain_height: float) -> np.ndarray:
    data = np.load(npz_path)
    pressure = np.asarray(data["pressure_flat"] if "pressure_flat" in data else data["pressure"], dtype=np.float64).reshape(-1)
    centers = np.asarray(data["cell_centers"], dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] != pressure.size:
        raise RuntimeError(f"invalid pressure export shapes: centers={centers.shape}, pressure={pressure.shape}")
    query_x = observation_points[:, 0] / domain_width
    query_z = observation_points[:, 1] / domain_height
    query = np.column_stack([query_x, np.zeros_like(query_x), query_z])
    values = []
    for point in query:
        distance2 = np.sum((centers - point) ** 2, axis=1)
        values.append(float(pressure[int(np.argmin(distance2))]))
    return np.asarray(values, dtype=np.float64)


def write_pressure_csv(path: Path, points: np.ndarray, pressures: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "pressure"])
        writer.writeheader()
        for point, pressure in zip(points, pressures):
            writer.writerow({"x": float(point[0]), "y": float(point[1]), "pressure": float(pressure)})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one GEOS/EDFM forward case from extracted DFN segments.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    request = json.loads((run_dir / "geos_request.json").read_text(encoding="utf-8"))
    segments = read_segments(run_dir / "dfn_segments.csv")
    points = read_points(run_dir / "observation_points.csv")
    if not GEOSX.exists():
        raise FileNotFoundError(f"geosx not found: {GEOSX}")
    if not EXPORT_PRESSURE.exists():
        raise FileNotFoundError(f"pressure export script not found: {EXPORT_PRESSURE}")
    visit_bin = shutil.which("visit") or (str(VISIT_FALLBACK) if VISIT_FALLBACK.exists() else None)
    if visit_bin is None:
        raise RuntimeError("VisIt executable was not found in WSL PATH or at /home/liuchaoran/.local/bin/visit; cannot export real GEOS pressure fields")

    batch_dir = run_dir / "geos_batch"
    case_dir = batch_dir / "candidate"
    dataset_dir = batch_dir / "dataset"
    case_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    xml_path = build_xml(run_dir, request, segments)
    case_xml = case_dir / "candidate.xml"
    shutil.copy2(xml_path, case_xml)
    geos_timeout = int(request.get("runtime", {}).get("geos_timeout_seconds", 900))
    run_checked([str(GEOSX), "-i", str(case_xml), "-o", str(case_dir / "output")], case_dir, case_dir / "run.log", geos_timeout)

    write_manifest(batch_dir, case_dir, case_xml, len(segments))
    write_case_metadata(dataset_dir)
    run_checked(
        [sys.executable, str(EXPORT_PRESSURE), "--batch-dir", str(batch_dir), "--dataset-dir", str(dataset_dir), "--visit-bin", visit_bin],
        ROOT,
        timeout=int(request.get("runtime", {}).get("visit_timeout_seconds", 600)),
    )

    npz_path = dataset_dir / "pressure_fields" / "candidate" / "pressure_visit_tfinal.npz"
    pressures = sample_pressure(
        npz_path,
        points,
        float(request.get("domain_width", 20.0)),
        float(request.get("domain_height", 20.0)),
    )
    output_csv = run_dir / str(request.get("output_pressure_csv", "pressure_observations.csv"))
    write_pressure_csv(output_csv, points, pressures)
    print(f"Wrote pressure observations: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
