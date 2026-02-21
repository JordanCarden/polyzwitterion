"""Generate optimization metric plots from optimisation results.

This script reads ``optimisation_results.csv`` and visualizes per-generation
minimum, average, and total values of the tracked metric.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _load_metric(path: Path) -> tuple[str, dict[int, list[float]]]:
    """Return per-generation metric lists from CSV.

    Args:
        path: CSV file path.

    Returns:
        Tuple of metric column name and generation-to-values map.
    """
    metrics: dict[int, list[float]] = {}
    with path.open() as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        metric_col = "total_loss" if "total_loss" in reader.fieldnames else "err"
        for row in reader:
            gen = int(row["gen"])
            value = float(row[metric_col])
            metrics.setdefault(gen, []).append(value)
    return metric_col, metrics


def _compute_stats(
    metrics: dict[int, list[float]],
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Compute minimum, average, and sum of metric values per generation.

    Args:
        metrics: Mapping from generation to list of values.

    Returns:
        Tuple of ``(generations, min_values, avg_values, sum_values)``.
    """
    generations = sorted(metrics.keys())
    min_values = [min(metrics[g]) for g in generations]
    avg_values = [float(np.mean(metrics[g])) for g in generations]
    sum_values = [sum(metrics[g]) for g in generations]
    return generations, min_values, avg_values, sum_values


def plot_errors(csv_path: Path) -> None:
    """Create plots of optimization metric per generation.

    Args:
        csv_path: CSV file with columns including ``gen`` and ``total_loss`` or ``err``.
    """
    metric_col, metrics = _load_metric(csv_path)
    generations, min_vals, avg_vals, sum_vals = _compute_stats(metrics)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # --- Plot 1: Minimum metric ---
    ax1.plot(generations, min_vals, marker="o", linestyle="-", color="b")
    ax1.set_title(f"Minimum {metric_col} per Generation", fontsize=14)
    ax1.set_ylabel(f"Minimum {metric_col}", fontsize=12)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    z1 = np.polyfit(generations, min_vals, 1)
    p1 = np.poly1d(z1)
    ax1.plot(
        generations,
        p1(generations),  # type: ignore[arg-type]
        "r--",
        label=f"Trendline (y={z1[0]:.2f}x+{z1[1]:.2f})",
    )
    ax1.legend()

    # --- Plot 2: Average metric ---
    ax2.plot(generations, avg_vals, marker="s", linestyle="-", color="g")
    ax2.set_title(f"Average {metric_col} per Generation", fontsize=14)
    ax2.set_ylabel(f"Average {metric_col}", fontsize=12)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    z2 = np.polyfit(generations, avg_vals, 1)
    p2 = np.poly1d(z2)
    ax2.plot(
        generations,
        p2(generations),  # type: ignore[arg-type]
        "r--",
        label=f"Trendline (y={z2[0]:.2f}x+{z2[1]:.2f})",
    )
    ax2.legend()

    # --- Plot 3: Sum of metric ---
    ax3.plot(
        generations,
        sum_vals,
        marker="^",
        linestyle="-",
        color="purple",
    )
    ax3.set_title(f"Sum of {metric_col} per Generation", fontsize=14)
    ax3.set_ylabel(f"Sum of {metric_col}", fontsize=12)
    ax3.set_xlabel("Generation", fontsize=12)
    ax3.grid(True, which="both", linestyle="--", linewidth=0.5)
    z3 = np.polyfit(generations, sum_vals, 1)
    p3 = np.poly1d(z3)
    ax3.plot(
        generations,
        p3(generations),  # type: ignore[arg-type]
        "r--",
        label=f"Trendline (y={z3[0]:.2f}x+{z3[1]:.2f})",
    )
    ax3.legend()

    plt.tight_layout(pad=3.0)
    plt.show()


def main() -> None:
    """Entry point for script execution."""
    csv_path = REPO / "optimisation_results.csv"
    plot_errors(csv_path)


if __name__ == "__main__":
    main()
