import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


METRIC_NAMES = [
    "fracture_pixel_ratio",
    "num_connected_components",
    "largest_component_ratio",
    "mean_component_area",
    "skeleton_length",
    "endpoint_count",
    "junction_count",
    "hough_line_count",
]


def read_binary_image(path: str | Path, image_size: int, threshold: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
        array = np.array(image)
    return (array > threshold).astype(np.uint8)


def load_image_dir(
    image_dir: str | Path,
    image_size: int,
    threshold: int,
    max_images: int | None,
) -> list[tuple[str, np.ndarray]]:
    paths = sorted(Path(image_dir).glob("*.png"))
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {image_dir}")
    return [(path.stem, read_binary_image(path, image_size, threshold)) for path in paths]


def split_grid_image(
    grid_path: str | Path,
    image_size: int,
    threshold: int,
    ncols: int | None,
    nrows: int | None,
    padding: int,
    max_images: int | None,
) -> list[tuple[str, np.ndarray]]:
    with Image.open(grid_path) as image:
        grid = np.array(image.convert("L"))

    height, width = grid.shape
    if ncols is None:
        ncols = max(1, (width - padding) // (image_size + padding))
    if nrows is None:
        nrows = max(1, (height - padding) // (image_size + padding))

    samples = []
    for row in range(nrows):
        for col in range(ncols):
            y0 = padding + row * (image_size + padding)
            x0 = padding + col * (image_size + padding)
            tile = grid[y0 : y0 + image_size, x0 : x0 + image_size]
            if tile.shape != (image_size, image_size):
                continue
            samples.append((f"{Path(grid_path).stem}_{len(samples):03d}", (tile > threshold).astype(np.uint8)))
            if max_images is not None and len(samples) >= max_images:
                return samples

    if not samples:
        raise ValueError(f"Could not split grid image: {grid_path}")
    return samples


def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Return a one-pixel skeleton using Zhang-Suen thinning."""
    image = binary.copy().astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            to_remove = []
            padded = np.pad(image, 1, mode="constant")
            rows, cols = np.nonzero(image)
            for y, x in zip(rows, cols):
                yy, xx = y + 1, x + 1
                p2 = padded[yy - 1, xx]
                p3 = padded[yy - 1, xx + 1]
                p4 = padded[yy, xx + 1]
                p5 = padded[yy + 1, xx + 1]
                p6 = padded[yy + 1, xx]
                p7 = padded[yy + 1, xx - 1]
                p8 = padded[yy, xx - 1]
                p9 = padded[yy - 1, xx - 1]
                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                neighbor_count = int(sum(neighbors))
                transitions = sum(
                    int(neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1)
                    for i in range(8)
                )
                if not (2 <= neighbor_count <= 6 and transitions == 1):
                    continue
                if step == 0:
                    keep = p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0
                else:
                    keep = p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
                if keep:
                    to_remove.append((y, x))
            if to_remove:
                ys, xs = zip(*to_remove)
                image[np.array(ys), np.array(xs)] = 0
                changed = True
    return image


def skeleton_degrees(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return neighbor_count - skeleton


def orientation_histogram(binary: np.ndarray, bins: int) -> tuple[np.ndarray, int]:
    lines = cv2.HoughLinesP(
        (binary * 255).astype(np.uint8),
        rho=1,
        theta=np.pi / 180,
        threshold=12,
        minLineLength=6,
        maxLineGap=3,
    )
    hist = np.zeros(bins, dtype=np.float64)
    if lines is None:
        return hist, 0

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        bin_index = min(bins - 1, int(angle / 180.0 * bins))
        length = float(np.hypot(x2 - x1, y2 - y1))
        hist[bin_index] += length

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist, int(len(lines))


def compute_metrics(name: str, binary: np.ndarray, orientation_bins: int) -> dict[str, float | str]:
    fracture_pixels = int(binary.sum())
    total_pixels = int(binary.size)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)

    skeleton = zhang_suen_thinning(binary)
    degrees = skeleton_degrees(skeleton)
    skeleton_mask = skeleton > 0
    hist, hough_line_count = orientation_histogram(binary, orientation_bins)

    metrics: dict[str, float | str] = {
        "sample_id": name,
        "fracture_pixel_ratio": fracture_pixels / max(1, total_pixels),
        "num_connected_components": float(max(0, num_labels - 1)),
        "largest_component_ratio": float(component_areas.max() / max(1, fracture_pixels)) if component_areas.size else 0.0,
        "mean_component_area": float(component_areas.mean()) if component_areas.size else 0.0,
        "skeleton_length": float(skeleton.sum()),
        "endpoint_count": float(np.logical_and(skeleton_mask, degrees == 1).sum()),
        "junction_count": float(np.logical_and(skeleton_mask, degrees >= 3).sum()),
        "hough_line_count": float(hough_line_count),
    }
    for index, value in enumerate(hist):
        metrics[f"orientation_bin_{index:02d}"] = float(value)
    return metrics


def write_metrics_csv(path: Path, rows: list[dict[str, float | str]], orientation_bins: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", *METRIC_NAMES, *[f"orientation_bin_{i:02d}" for i in range(orientation_bins)]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | str]], orientation_bins: int) -> dict[str, object]:
    summary: dict[str, object] = {"num_samples": len(rows)}
    for metric in METRIC_NAMES:
        values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    orientation = np.array(
        [[float(row[f"orientation_bin_{i:02d}"]) for i in range(orientation_bins)] for row in rows],
        dtype=np.float64,
    )
    summary["orientation_histogram_mean"] = orientation.mean(axis=0).tolist()
    return summary


def compare_summaries(real: dict[str, object], generated: dict[str, object]) -> dict[str, object]:
    comparison: dict[str, object] = {}
    eps = 1e-8
    for metric in METRIC_NAMES:
        real_mean = float(real[metric]["mean"])  # type: ignore[index]
        generated_mean = float(generated[metric]["mean"])  # type: ignore[index]
        comparison[metric] = {
            "real_mean": real_mean,
            "generated_mean": generated_mean,
            "absolute_mean_error": abs(generated_mean - real_mean),
            "relative_mean_error": abs(generated_mean - real_mean) / (abs(real_mean) + eps),
        }

    real_orientation = np.array(real["orientation_histogram_mean"], dtype=np.float64)
    generated_orientation = np.array(generated["orientation_histogram_mean"], dtype=np.float64)
    comparison["orientation_l1_distance"] = float(np.abs(real_orientation - generated_orientation).sum())
    return comparison


def plot_comparison(
    out_path: Path,
    real_rows: list[dict[str, float | str]],
    generated_rows: list[dict[str, float | str]],
    real_summary: dict[str, object],
    generated_summary: dict[str, object],
    orientation_bins: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    hist_metrics = [
        "fracture_pixel_ratio",
        "num_connected_components",
        "largest_component_ratio",
        "skeleton_length",
        "junction_count",
    ]
    for ax, metric in zip(axes.flat[:5], hist_metrics):
        real_values = [float(row[metric]) for row in real_rows]
        generated_values = [float(row[metric]) for row in generated_rows]
        ax.hist(real_values, bins=24, alpha=0.6, label="reference", density=True)
        ax.hist(generated_values, bins=24, alpha=0.6, label="generated", density=True)
        ax.set_title(metric)
        ax.legend()

    ax = axes.flat[5]
    centers = np.linspace(0, 180, orientation_bins, endpoint=False)
    width = 180 / orientation_bins
    real_orientation = np.array(real_summary["orientation_histogram_mean"], dtype=np.float64)
    generated_orientation = np.array(generated_summary["orientation_histogram_mean"], dtype=np.float64)
    ax.bar(centers - width * 0.2, real_orientation, width=width * 0.4, label="reference")
    ax.bar(centers + width * 0.2, generated_orientation, width=width * 0.4, label="generated")
    ax.set_title("orientation_histogram")
    ax.set_xlabel("angle_deg")
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    real_samples = load_image_dir(args.real_dir, args.image_size, args.threshold, args.max_real_images)
    if args.generated_grid is not None:
        generated_samples = split_grid_image(
            args.generated_grid,
            args.image_size,
            args.threshold,
            args.grid_cols,
            args.grid_rows,
            args.grid_padding,
            args.max_generated_images,
        )
    elif args.generated_dir is not None:
        generated_samples = load_image_dir(
            args.generated_dir,
            args.image_size,
            args.threshold,
            args.max_generated_images,
        )
    else:
        raise ValueError("Provide either --generated_dir or --generated_grid")

    real_rows = [
        compute_metrics(name, binary, args.orientation_bins)
        for name, binary in real_samples
    ]
    generated_rows = [
        compute_metrics(name, binary, args.orientation_bins)
        for name, binary in generated_samples
    ]

    real_summary = summarize(real_rows, args.orientation_bins)
    generated_summary = summarize(generated_rows, args.orientation_bins)
    comparison = compare_summaries(real_summary, generated_summary)

    write_metrics_csv(out_dir / "metrics_reference.csv", real_rows, args.orientation_bins)
    write_metrics_csv(out_dir / "metrics_generated.csv", generated_rows, args.orientation_bins)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "reference": real_summary,
                "generated": generated_summary,
                "comparison": comparison,
            },
            handle,
            indent=2,
        )
    with (out_dir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "real_mean", "generated_mean", "absolute_mean_error", "relative_mean_error"])
        for metric in METRIC_NAMES:
            values = comparison[metric]
            writer.writerow(
                [
                    metric,
                    values["real_mean"],
                    values["generated_mean"],
                    values["absolute_mean_error"],
                    values["relative_mean_error"],
                ]
            )
        writer.writerow(["orientation_l1_distance", "", comparison["orientation_l1_distance"], "", ""])

    plot_comparison(
        out_dir / "comparison_plots.png",
        real_rows,
        generated_rows,
        real_summary,
        generated_summary,
        args.orientation_bins,
    )
    print(f"Evaluated {len(real_rows)} reference images and {len(generated_rows)} generated images.")
    print(f"Wrote evaluation outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare reference DFN images with generated DFN images.")
    parser.add_argument("--real_dir", type=Path, required=True, help="Directory of reference synthetic DFN PNGs.")
    parser.add_argument("--generated_dir", type=Path, default=None, help="Directory of generated single-image PNGs.")
    parser.add_argument("--generated_grid", type=Path, default=None, help="Generated grid PNG from save_image_grid.")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--max_real_images", type=int, default=512)
    parser.add_argument("--max_generated_images", type=int, default=None)
    parser.add_argument("--orientation_bins", type=int, default=18)
    parser.add_argument("--grid_rows", type=int, default=None)
    parser.add_argument("--grid_cols", type=int, default=None)
    parser.add_argument("--grid_padding", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
