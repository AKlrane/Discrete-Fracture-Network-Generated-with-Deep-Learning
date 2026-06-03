from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FractureSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_degrees: float

    @property
    def midpoint(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))


def _scale_endpoint(
    x: int,
    y: int,
    width: int,
    height: int,
    domain_width: float,
    domain_height: float,
    invert_y: bool,
) -> tuple[float, float]:
    x_physical = float(x) / max(1, width - 1) * domain_width
    y_scaled = float(y) / max(1, height - 1) * domain_height
    y_physical = domain_height - y_scaled if invert_y else y_scaled
    return x_physical, y_physical


def extract_fracture_segments(
    binary: np.ndarray,
    domain_width: float,
    domain_height: float,
    hough_threshold: int = 12,
    min_line_length: int = 6,
    max_line_gap: int = 3,
    invert_y: bool = True,
) -> list[FractureSegment]:
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D binary image, got shape {binary.shape}")

    height, width = binary.shape
    image = ((binary > 0).astype(np.uint8) * 255)
    lines = cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(hough_threshold),
        minLineLength=int(min_line_length),
        maxLineGap=int(max_line_gap),
    )
    if lines is None:
        return []

    segments: list[FractureSegment] = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        px1, py1 = _scale_endpoint(x1, y1, width, height, domain_width, domain_height, invert_y)
        px2, py2 = _scale_endpoint(x2, y2, width, height, domain_width, domain_height, invert_y)
        dx = px2 - px1
        dy = py2 - py1
        length = float(np.hypot(dx, dy))
        if length <= 0.0:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
        segments.append(FractureSegment(px1, py1, px2, py2, length, angle))
    return segments


def write_segments_csv(path: str | Path, segments: list[FractureSegment]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["x1", "y1", "x2", "y2", "length", "angle_degrees"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for segment in segments:
            writer.writerow(asdict(segment))
    return out_path


def point_to_segment_distance(points: np.ndarray, segment: FractureSegment) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    a = np.array([segment.x1, segment.y1], dtype=np.float64)
    b = np.array([segment.x2, segment.y2], dtype=np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return np.linalg.norm(p - a, axis=1)
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    projection = a + t[:, None] * ab
    return np.linalg.norm(p - projection, axis=1)
