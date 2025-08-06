"""Generate error plots from optimisation results.

This script reads ``optimisation_results.csv`` and visualizes error metrics per
optimization generation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _load_errors(path: Path) -> dict[int, list[float]]:
    """Return per-generation error lists from CSV.

    Args:
        path: CSV file path.

    Returns:
        Mapping of generation number to list of errors.
    """
    errors: dict[int, list[float]] = {}
    with path.open() as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            gen = int(row["gen"])
            err = float(row["err"])
            errors.setdefault(gen, []).append(err)
    return errors


def _compute_stats(
    errors: dict[int, list[float]],
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Compute minimum, average, and sum of errors per generation.

    Args:
        errors: Mapping from generation to list of errors.

    Returns:
        Tuple of ``(generations, min_errors, avg_errors, sum_errors)``.
    """
    generations = sorted(errors.keys())
    min_errors = [min(errors[g]) for g in generations]
    avg_errors = [float(np.mean(errors[g])) for g in generations]
    sum_errors = [sum(errors[g]) for g in generations]
    return generations, min_errors, avg_errors, sum_errors


def plot_errors(csv_path: Path) -> None:
    """Create plots of error metrics per generation.

    Args:
        csv_path: CSV file with columns including ``gen`` and ``err``.
    """
    errors = _load_errors(csv_path)
    generations, min_errors, avg_errors, sum_errors = _compute_stats(errors)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # --- Plot 1: Minimum Error ---
    ax1.plot(generations, min_errors, marker="o", linestyle="-", color="b")
    ax1.set_title("Minimum Error per Generation", fontsize=14)
    ax1.set_ylabel("Minimum Error", fontsize=12)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    z1 = np.polyfit(generations, min_errors, 1)
    p1 = np.poly1d(z1)
    ax1.plot(
        generations,
        p1(generations),
        "r--",
        label=f"Trendline (y={z1[0]:.2f}x+{z1[1]:.2f})",
    )
    ax1.legend()

    # --- Plot 2: Average Error ---
    ax2.plot(generations, avg_errors, marker="s", linestyle="-", color="g")
    ax2.set_title("Average Error per Generation", fontsize=14)
    ax2.set_ylabel("Average Error", fontsize=12)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    z2 = np.polyfit(generations, avg_errors, 1)
    p2 = np.poly1d(z2)
    ax2.plot(
        generations,
        p2(generations),
        "r--",
        label=f"Trendline (y={z2[0]:.2f}x+{z2[1]:.2f})",
    )
    ax2.legend()

    # --- Plot 3: Sum of Error ---
    ax3.plot(
        generations,
        sum_errors,
        marker="^",
        linestyle="-",
        color="purple",
    )
    ax3.set_title("Sum of Error per Generation", fontsize=14)
    ax3.set_ylabel("Sum of Error", fontsize=12)
    ax3.set_xlabel("Generation", fontsize=12)
    ax3.grid(True, which="both", linestyle="--", linewidth=0.5)
    z3 = np.polyfit(generations, sum_errors, 1)
    p3 = np.poly1d(z3)
    ax3.plot(
        generations,
        p3(generations),
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
