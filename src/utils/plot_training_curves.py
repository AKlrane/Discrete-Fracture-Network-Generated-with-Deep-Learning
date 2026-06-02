import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EXCLUDED_COLUMNS = {"epoch", "step", "global_step"}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_float(value: str | None) -> float:
    if value is None:
        return math.nan
    value = value.strip()
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def load_csv_columns(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")

        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    columns: dict[str, np.ndarray] = {}
    for name in fieldnames:
        columns[name] = np.asarray([parse_float(row.get(name)) for row in rows], dtype=float)
    return columns, fieldnames


def choose_x_axis(columns: dict[str, np.ndarray], num_rows: int) -> tuple[str, np.ndarray]:
    for name in ("step", "global_step"):
        values = columns.get(name)
        if values is not None and np.isfinite(values).any():
            return name, values
    values = columns.get("epoch")
    if values is not None and np.isfinite(values).any():
        return "epoch", values
    return "row", np.arange(num_rows, dtype=float)


def is_plottable_metric(name: str, values: np.ndarray, excluded_columns: set[str]) -> bool:
    if name in excluded_columns:
        return False

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return False
    if np.allclose(finite_values, 0.0):
        return False
    return True


def exponential_moving_average(values: np.ndarray, alpha: float) -> np.ndarray:
    if values.size == 0:
        return values

    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]
    for index in range(1, values.size):
        smoothed[index] = alpha * values[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def plot_csv(
    csv_path: Path,
    output_path: Path,
    excluded_columns: set[str],
    ema_alpha: float | None,
    ema_label: str | None,
    title: str | None = None,
) -> list[str]:
    columns, fieldnames = load_csv_columns(csv_path)
    if not fieldnames:
        raise ValueError(f"{csv_path} has no columns")

    num_rows = len(next(iter(columns.values()))) if columns else 0
    x_label, x_values = choose_x_axis(columns, num_rows)
    metric_names = [
        name
        for name in fieldnames
        if is_plottable_metric(name, columns[name], excluded_columns)
    ]
    if not metric_names:
        raise ValueError(f"{csv_path} has no plottable numeric metric columns")

    num_metrics = len(metric_names)
    num_cols = 2 if num_metrics > 1 else 1
    num_rows_plot = math.ceil(num_metrics / num_cols)
    figure, axes = plt.subplots(
        num_rows_plot,
        num_cols,
        figsize=(6.5 * num_cols, 3.2 * num_rows_plot),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()

    for axis, metric_name in zip(axes_flat, metric_names):
        values = columns[metric_name]
        mask = np.isfinite(x_values) & np.isfinite(values)
        x_plot = x_values[mask]
        y_plot = values[mask]
        if ema_alpha is None:
            axis.plot(x_plot, y_plot, linewidth=1.3)
        else:
            axis.plot(x_plot, y_plot, linewidth=0.8, alpha=0.28, label="raw")
            axis.plot(
                x_plot,
                exponential_moving_average(y_plot, ema_alpha),
                linewidth=1.6,
                label=ema_label or "EMA",
            )
            axis.legend(fontsize="small")
        axis.set_title(metric_name)
        axis.set_xlabel(x_label)
        axis.grid(True, alpha=0.3)

    for axis in axes_flat[num_metrics:]:
        axis.set_visible(False)

    figure.suptitle(title or csv_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return metric_names


def make_output_stem(csv_path: Path, disambiguate: bool) -> str:
    if not disambiguate:
        return csv_path.stem
    try:
        path = csv_path.relative_to(PROJECT_ROOT).with_suffix("")
    except ValueError:
        path = csv_path.with_suffix("")
    parts = [part.strip("/\\") for part in path.parts if part.strip("/\\")]
    return "_".join(parts) or csv_path.stem


def default_output_path(csv_path: Path, output_dir: Path | None, disambiguate: bool) -> Path:
    output_stem = make_output_stem(csv_path, disambiguate)
    if output_dir is not None:
        return output_dir / f"{output_stem}_curves.png"
    return csv_path.with_name(f"{output_stem}_curves.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot numeric training curves from train_log.csv or Lightning metrics.csv."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="CSV log files to plot, for example outputs/.../train_log.csv or metrics.csv.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to writing next to each CSV.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Additional column names to skip.",
    )
    parser.add_argument(
        "--ema_span",
        type=int,
        default=20,
        help="EMA span. The smoothing alpha is 2 / (span + 1). Default: 20.",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=None,
        help="EMA alpha in (0, 1]. Overrides --ema_span when set.",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Disable EMA overlay and plot only raw curves.",
    )
    return parser.parse_args()


def resolve_ema(args: argparse.Namespace) -> tuple[float | None, str | None]:
    if args.no_ema:
        return None, None
    if args.ema_alpha is not None:
        if not 0.0 < args.ema_alpha <= 1.0:
            raise ValueError("--ema_alpha must be in (0, 1]")
        return float(args.ema_alpha), f"EMA alpha={args.ema_alpha:g}"
    if args.ema_span < 1:
        raise ValueError("--ema_span must be a positive integer")
    alpha = 2.0 / (float(args.ema_span) + 1.0)
    return alpha, f"EMA span={args.ema_span}"


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.out_dir) if args.out_dir is not None else None
    excluded_columns = DEFAULT_EXCLUDED_COLUMNS | set(args.exclude)
    csv_paths = [resolve_path(csv_file) for csv_file in args.csv_files]
    disambiguate = output_dir is not None and len(csv_paths) > 1
    ema_alpha, ema_label = resolve_ema(args)

    for csv_path in csv_paths:
        output_path = default_output_path(csv_path, output_dir, disambiguate)
        metric_names = plot_csv(csv_path, output_path, excluded_columns, ema_alpha, ema_label)
        print(
            f"Wrote {output_path} with {len(metric_names)} curves: "
            f"{', '.join(metric_names)}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
