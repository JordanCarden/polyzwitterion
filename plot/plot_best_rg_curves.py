#!/usr/bin/env python3
"""Plot best-child simulated Rg distributions against target distributions."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/admin/jcarde7/polyzwitterion")
PLOT_DIR = ROOT / "plot"
RESULTS_CSV = ROOT / "optimisation_results.csv"
SYSTEMS = ("10mer", "30mer", "60mer")


def find_best_row(path: Path) -> dict[str, str]:
    best: dict[str, str] | None = None
    best_loss = float("inf")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loss = float(row["total_loss"])
            if loss < best_loss:
                best_loss = loss
                best = row
    if best is None:
        raise RuntimeError(f"No rows found in {path}")
    return best


def load_distribution(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1]


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    best = find_best_row(RESULTS_CSV)
    gen = int(best["gen"])
    child = int(best["child"])
    total_loss = float(best["total_loss"])

    for system in SYSTEMS:
        sim = (
            ROOT
            / "3_simulations"
            / f"generation_{gen:02d}"
            / f"child_{child:02d}"
            / system
            / "Rg_distribution.dat"
        )
        target = ROOT / "1_input" / system / "target_rg_distribution.dat"
        if not sim.exists():
            raise FileNotFoundError(f"Missing simulated distribution: {sim}")
        if not target.exists():
            raise FileNotFoundError(f"Missing target distribution: {target}")

        x_sim, y_sim = load_distribution(sim)
        x_tgt, y_tgt = load_distribution(target)

        plt.figure(figsize=(8, 5))
        plt.plot(x_tgt, y_tgt, label=f"{system} target", linewidth=2.2)
        plt.plot(x_sim, y_sim, label=f"{system} simulated (best)", linewidth=2.2)
        plt.xlabel("Rg (A)")
        plt.ylabel("Probability")
        plt.title(
            f"{system}: Target vs Best Simulated\n"
            f"generation={gen:02d}, child={child:02d}, total_loss={total_loss:.10f}"
        )
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        out = PLOT_DIR / f"best_rg_compare_{system}.png"
        plt.savefig(out, dpi=180)
        plt.close()
        print(out)

    summary = PLOT_DIR / "best_rg_compare_summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"best_generation: {gen:02d}",
                f"best_child: {child:02d}",
                f"best_total_loss: {total_loss:.10f}",
                f"eps: {best['eps']}",
            ]
        )
        + "\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
