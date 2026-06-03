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
    "hough_length_mean",
    "hough_length_std",
    "hough_length_median",
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


def detect_hough_lines(
    binary: np.ndarray,
    hough_threshold: int,
    hough_min_line_length: int,
    hough_max_line_gap: int,
) -> np.ndarray:
    skeleton = zhang_suen_thinning(binary)
    lines = cv2.HoughLinesP(
        (skeleton * 255).astype(np.uint8),
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=hough_min_line_length,
        maxLineGap=hough_max_line_gap,
    )
    if lines is None:
        return np.empty((0, 4), dtype=np.float64)
    return lines[:, 0, :].astype(np.float64)


def line_angles_lengths_centers(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lines.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )

    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]
    angles = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    angles = ((angles + 90.0) % 180.0) - 90.0
    lengths = np.hypot(x2 - x1, y2 - y1)
    centers = np.column_stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
    return angles, lengths, centers


def orientation_histogram(lines: np.ndarray, bins: int) -> np.ndarray:
    hist = np.zeros(bins, dtype=np.float64)
    angles, lengths, _ = line_angles_lengths_centers(lines)
    if angles.size == 0:
        return hist

    bin_indices = np.floor((angles + 90.0) / 180.0 * bins).astype(int)
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    for bin_index, length in zip(bin_indices, lengths):
        hist[int(bin_index)] += float(length)

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def length_histogram(lengths: np.ndarray, bins: int, max_length: float) -> np.ndarray:
    if lengths.size == 0:
        return np.zeros(bins, dtype=np.float64)
    hist, _ = np.histogram(lengths, bins=bins, range=(0.0, max_length))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def center_heatmap(lines: np.ndarray, image_size: int, grid_size: int) -> np.ndarray:
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float64)
    _, _, centers = line_angles_lengths_centers(lines)
    if centers.size == 0:
        return heatmap

    xs = np.clip((centers[:, 0] / image_size * grid_size).astype(int), 0, grid_size - 1)
    ys = np.clip((centers[:, 1] / image_size * grid_size).astype(int), 0, grid_size - 1)
    np.add.at(heatmap, (ys, xs), 1.0)
    total = heatmap.sum()
    if total > 0:
        heatmap /= total
    return heatmap


def compute_metrics(
    name: str,
    binary: np.ndarray,
    orientation_bins: int,
    length_bins: int,
    max_line_length: float,
    hough_threshold: int,
    hough_min_line_length: int,
    hough_max_line_gap: int,
) -> dict[str, float | str]:
    fracture_pixels = int(binary.sum())
    total_pixels = int(binary.size)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)

    skeleton = zhang_suen_thinning(binary)
    degrees = skeleton_degrees(skeleton)
    skeleton_mask = skeleton > 0
    lines = detect_hough_lines(
        binary,
        hough_threshold,
        hough_min_line_length,
        hough_max_line_gap,
    )
    orientation_hist = orientation_histogram(lines, orientation_bins)
    _, lengths, _ = line_angles_lengths_centers(lines)
    length_hist = length_histogram(lengths, length_bins, max_line_length)

    metrics: dict[str, float | str] = {
        "sample_id": name,
        "fracture_pixel_ratio": fracture_pixels / max(1, total_pixels),
        "num_connected_components": float(max(0, num_labels - 1)),
        "largest_component_ratio": float(component_areas.max() / max(1, fracture_pixels)) if component_areas.size else 0.0,
        "mean_component_area": float(component_areas.mean()) if component_areas.size else 0.0,
        "skeleton_length": float(skeleton.sum()),
        "endpoint_count": float(np.logical_and(skeleton_mask, degrees == 1).sum()),
        "junction_count": float(np.logical_and(skeleton_mask, degrees >= 3).sum()),
        "hough_line_count": float(len(lines)),
        "hough_length_mean": float(lengths.mean()) if lengths.size else 0.0,
        "hough_length_std": float(lengths.std(ddof=0)) if lengths.size else 0.0,
        "hough_length_median": float(np.median(lengths)) if lengths.size else 0.0,
    }
    for index, value in enumerate(orientation_hist):
        metrics[f"orientation_bin_{index:02d}"] = float(value)
    for index, value in enumerate(length_hist):
        metrics[f"length_bin_{index:02d}"] = float(value)
    return metrics


def write_metrics_csv(path: Path, rows: list[dict[str, float | str]], orientation_bins: int, length_bins: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        *METRIC_NAMES,
        *[f"orientation_bin_{i:02d}" for i in range(orientation_bins)],
        *[f"length_bin_{i:02d}" for i in range(length_bins)],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | str]], orientation_bins: int, length_bins: int) -> dict[str, object]:
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
    length = np.array(
        [[float(row[f"length_bin_{i:02d}"]) for i in range(length_bins)] for row in rows],
        dtype=np.float64,
    )
    summary["length_histogram_mean"] = length.mean(axis=0).tolist()
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
    real_length = np.array(real["length_histogram_mean"], dtype=np.float64)
    generated_length = np.array(generated["length_histogram_mean"], dtype=np.float64)
    comparison["length_l1_distance"] = float(np.abs(real_length - generated_length).sum())
    return comparison


def plot_comparison(
    out_path: Path,
    real_rows: list[dict[str, float | str]],
    generated_rows: list[dict[str, float | str]],
    real_summary: dict[str, object],
    generated_summary: dict[str, object],
    orientation_bins: int,
    length_bins: int,
    max_line_length: float,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    hist_metrics = [
        "fracture_pixel_ratio",
        "hough_line_count",
        "hough_length_mean",
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
    centers = np.linspace(-90, 90, orientation_bins, endpoint=False)
    width = 180 / orientation_bins
    real_orientation = np.array(real_summary["orientation_histogram_mean"], dtype=np.float64)
    generated_orientation = np.array(generated_summary["orientation_histogram_mean"], dtype=np.float64)
    ax.bar(centers - width * 0.2, real_orientation, width=width * 0.4, label="reference")
    ax.bar(centers + width * 0.2, generated_orientation, width=width * 0.4, label="generated")
    ax.set_title("orientation_histogram")
    ax.set_xlabel("angle_deg")
    ax.legend()

    ax = axes.flat[6]
    length_centers = np.linspace(0, max_line_length, length_bins, endpoint=False)
    length_width = max_line_length / length_bins
    real_length = np.array(real_summary["length_histogram_mean"], dtype=np.float64)
    generated_length = np.array(generated_summary["length_histogram_mean"], dtype=np.float64)
    ax.bar(length_centers - length_width * 0.2, real_length, width=length_width * 0.4, label="reference")
    ax.bar(length_centers + length_width * 0.2, generated_length, width=length_width * 0.4, label="generated")
    ax.set_title("line_length_histogram")
    ax.set_xlabel("length_px")
    ax.legend()

    axes.flat[7].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def compute_overlay(samples: list[tuple[str, np.ndarray]]) -> np.ndarray:
    if not samples:
        raise ValueError("Cannot compute overlay from an empty sample set.")
    stack = np.stack([binary.astype(np.float64) for _, binary in samples], axis=0)
    return stack.mean(axis=0)


def save_overlay_image(path: Path, overlay: np.ndarray, cmap: str = "inferno") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(overlay, cmap=cmap, vmin=0.0, vmax=max(float(overlay.max()), 1e-8))
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_heatmap_image(path: Path, heatmap: np.ndarray, title: str, cmap: str = "viridis") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(heatmap, cmap=cmap, vmin=0.0, vmax=max(float(heatmap.max()), 1e-8))
    plt.title(title)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_overlay_comparison(
    out_path: Path,
    real_overlay: np.ndarray,
    generated_overlay: np.ndarray,
) -> dict[str, float]:
    difference = np.abs(real_overlay - generated_overlay)
    vmax = max(float(real_overlay.max()), float(generated_overlay.max()), 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    images = [
        (real_overlay, "GT overlay", "inferno", vmax),
        (generated_overlay, "GAN overlay", "inferno", vmax),
        (difference, "|GT - GAN|", "magma", max(float(difference.max()), 1e-8)),
    ]
    for ax, (image, title, cmap, local_vmax) in zip(axes, images):
        im = ax.imshow(image, cmap=cmap, vmin=0.0, vmax=local_vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return {
        "overlay_mae": float(difference.mean()),
        "overlay_mse": float(np.mean((real_overlay - generated_overlay) ** 2)),
        "overlay_max_abs_error": float(difference.max()),
    }


def compute_center_heatmap_for_samples(
    samples: list[tuple[str, np.ndarray]],
    image_size: int,
    grid_size: int,
    hough_threshold: int,
    hough_min_line_length: int,
    hough_max_line_gap: int,
) -> np.ndarray:
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float64)
    for _, binary in samples:
        lines = detect_hough_lines(
            binary,
            hough_threshold,
            hough_min_line_length,
            hough_max_line_gap,
        )
        heatmap += center_heatmap(lines, image_size, grid_size)
    total = heatmap.sum()
    if total > 0:
        heatmap /= total
    return heatmap


def plot_center_heatmap_comparison(
    out_path: Path,
    real_heatmap: np.ndarray,
    generated_heatmap: np.ndarray,
) -> dict[str, float]:
    difference = np.abs(real_heatmap - generated_heatmap)
    vmax = max(float(real_heatmap.max()), float(generated_heatmap.max()), 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    images = [
        (real_heatmap, "GT line centers", "viridis", vmax),
        (generated_heatmap, "GAN line centers", "viridis", vmax),
        (difference, "|GT - GAN|", "magma", max(float(difference.max()), 1e-8)),
    ]
    for ax, (image, title, cmap, local_vmax) in zip(axes, images):
        im = ax.imshow(image, cmap=cmap, vmin=0.0, vmax=local_vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return {
        "center_heatmap_mae": float(difference.mean()),
        "center_heatmap_mse": float(np.mean((real_heatmap - generated_heatmap) ** 2)),
        "center_heatmap_l1_distance": float(difference.sum()),
    }


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
        compute_metrics(
            name,
            binary,
            args.orientation_bins,
            args.length_bins,
            args.max_line_length,
            args.hough_threshold,
            args.hough_min_line_length,
            args.hough_max_line_gap,
        )
        for name, binary in real_samples
    ]
    generated_rows = [
        compute_metrics(
            name,
            binary,
            args.orientation_bins,
            args.length_bins,
            args.max_line_length,
            args.hough_threshold,
            args.hough_min_line_length,
            args.hough_max_line_gap,
        )
        for name, binary in generated_samples
    ]

    real_summary = summarize(real_rows, args.orientation_bins, args.length_bins)
    generated_summary = summarize(generated_rows, args.orientation_bins, args.length_bins)
    comparison = compare_summaries(real_summary, generated_summary)
    real_overlay = compute_overlay(real_samples)
    generated_overlay = compute_overlay(generated_samples)
    overlay_metrics = plot_overlay_comparison(
        out_dir / "overlay_comparison.png",
        real_overlay,
        generated_overlay,
    )
    comparison.update(overlay_metrics)
    real_center_heatmap = compute_center_heatmap_for_samples(
        real_samples,
        args.image_size,
        args.center_grid_size,
        args.hough_threshold,
        args.hough_min_line_length,
        args.hough_max_line_gap,
    )
    generated_center_heatmap = compute_center_heatmap_for_samples(
        generated_samples,
        args.image_size,
        args.center_grid_size,
        args.hough_threshold,
        args.hough_min_line_length,
        args.hough_max_line_gap,
    )
    center_heatmap_metrics = plot_center_heatmap_comparison(
        out_dir / "center_heatmap_comparison.png",
        real_center_heatmap,
        generated_center_heatmap,
    )
    comparison.update(center_heatmap_metrics)

    write_metrics_csv(out_dir / "metrics_reference.csv", real_rows, args.orientation_bins, args.length_bins)
    write_metrics_csv(out_dir / "metrics_generated.csv", generated_rows, args.orientation_bins, args.length_bins)
    save_overlay_image(out_dir / "overlay_reference.png", real_overlay)
    save_overlay_image(out_dir / "overlay_generated.png", generated_overlay)
    save_overlay_image(out_dir / "overlay_abs_difference.png", np.abs(real_overlay - generated_overlay), cmap="magma")
    save_heatmap_image(out_dir / "center_heatmap_reference.png", real_center_heatmap, "GT line centers")
    save_heatmap_image(out_dir / "center_heatmap_generated.png", generated_center_heatmap, "GAN line centers")
    save_heatmap_image(
        out_dir / "center_heatmap_abs_difference.png",
        np.abs(real_center_heatmap - generated_center_heatmap),
        "|GT - GAN| line centers",
        cmap="magma",
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "reference": real_summary,
                "generated": generated_summary,
                "comparison": comparison,
                "settings": {
                    "threshold": args.threshold,
                    "hough_threshold": args.hough_threshold,
                    "hough_min_line_length": args.hough_min_line_length,
                    "hough_max_line_gap": args.hough_max_line_gap,
                    "orientation_bins": args.orientation_bins,
                    "length_bins": args.length_bins,
                    "max_line_length": args.max_line_length,
                    "center_grid_size": args.center_grid_size,
                },
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
        writer.writerow(["length_l1_distance", "", comparison["length_l1_distance"], "", ""])
        writer.writerow(["overlay_mae", "", comparison["overlay_mae"], "", ""])
        writer.writerow(["overlay_mse", "", comparison["overlay_mse"], "", ""])
        writer.writerow(["overlay_max_abs_error", "", comparison["overlay_max_abs_error"], "", ""])
        writer.writerow(["center_heatmap_mae", "", comparison["center_heatmap_mae"], "", ""])
        writer.writerow(["center_heatmap_mse", "", comparison["center_heatmap_mse"], "", ""])
        writer.writerow(["center_heatmap_l1_distance", "", comparison["center_heatmap_l1_distance"], "", ""])

    plot_comparison(
        out_dir / "comparison_plots.png",
        real_rows,
        generated_rows,
        real_summary,
        generated_summary,
        args.orientation_bins,
        args.length_bins,
        args.max_line_length,
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
    parser.add_argument("--length_bins", type=int, default=16)
    parser.add_argument("--max_line_length", type=float, default=128.0)
    parser.add_argument("--center_grid_size", type=int, default=16)
    parser.add_argument("--hough_threshold", type=int, default=8)
    parser.add_argument("--hough_min_line_length", type=int, default=8)
    parser.add_argument("--hough_max_line_gap", type=int, default=3)
    parser.add_argument("--grid_rows", type=int, default=None)
    parser.add_argument("--grid_cols", type=int, default=None)
    parser.add_argument("--grid_padding", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
