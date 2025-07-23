"""Optimize Emin parameters using CMA-ES."""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

import cma
import numpy as np

_ALLOWED_LINE_START = 13  # zero-based index for first optimizable line
_NUM_PARAMS = 10


def _load_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return f.readlines()


def _write_lines(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def _parse_parameters(lines: list[str]) -> np.ndarray:
    params = []
    pattern = re.compile(r"^(\S+\s+\S+\s+)([-+]?[0-9]*\.?[0-9]+)")
    for i in range(_ALLOWED_LINE_START, _ALLOWED_LINE_START + _NUM_PARAMS):
        match = pattern.search(lines[i])
        if not match:
            raise ValueError(f"Failed to parse parameter line: {lines[i]}")
        params.append(float(match.group(2)))
    return np.array(params)


def _update_parameters(lines: list[str], params: np.ndarray) -> list[str]:
    new_lines = lines[:]
    pattern = re.compile(r"^(\S+\s+\S+\s+)([-+]?[0-9]*\.?[0-9]+)(.*)$")
    for offset, value in enumerate(params):
        idx = _ALLOWED_LINE_START + offset
        match = pattern.match(new_lines[idx])
        if not match:
            raise ValueError(f"Failed to parse parameter line: {new_lines[idx]}")
        prefix, _, suffix = match.groups()
        new_val = f"{value:.11f}"
        if suffix.endswith("\n"):
            new_lines[idx] = f"{prefix}{new_val}{suffix}"
        else:
            new_lines[idx] = f"{prefix}{new_val}{suffix}\n"
    return new_lines


def _load_valid_values(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        values = [float(line.strip()) for line in f if line.strip()]
    return np.array(sorted(values))


def _map_to_valid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    idx = np.abs(valid[:, None] - values).argmin(axis=0)
    return valid[idx]


def _run_simulation(executable: str, config: Path, cwd: Path) -> bool:
    try:
        subprocess.run([executable, str(config)], cwd=cwd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        logging.error("Simulation failed: %s", exc)
        return False


def _compute_rg(script: Path, cwd: Path) -> float | None:
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logging.error("Rg calculation failed: %s", exc)
        return None
    match = re.search(r"Average Rg of last 20 frames: ([0-9.]+)", result.stdout)
    if not match:
        logging.error("Failed to parse Rg output: %s", result.stdout)
        return None
    return float(match.group(1))


def _prepare_individual(
    base_par: Path,
    base_cfg: Path,
    work_dir: Path,
    params: np.ndarray,
) -> tuple[Path, Path]:
    lines = _load_lines(base_par)
    updated = _update_parameters(lines, params)
    par_path = work_dir / base_par.name
    _write_lines(par_path, updated)

    cfg_lines = _load_lines(base_cfg)
    new_cfg = []
    for line in cfg_lines:
        if line.strip().startswith("parameters") and base_par.name in line:
            new_cfg.append(f"parameters                  {par_path}\n")
        else:
            new_cfg.append(line)
    cfg_path = work_dir / base_cfg.name
    _write_lines(cfg_path, new_cfg)
    return par_path, cfg_path


def run_cma(
    base_par: Path,
    base_cfg: Path,
    valid_values: Path,
    output_dir: Path,
    executable: str,
    script: Path,
    generations: int,
    population: int,
    sigma: float,
) -> None:
    lines = _load_lines(base_par)
    x0 = _parse_parameters(lines)
    valid = _load_valid_values(valid_values)

    es = cma.CMAEvolutionStrategy(x0, sigma, {"popsize": population})
    output_dir.mkdir(parents=True, exist_ok=True)

    for gen in range(generations):
        solutions = es.ask()
        mapped = [_map_to_valid(np.array(s), valid) for s in solutions]
        fitness = []
        gen_dir = output_dir / f"generation_{gen+1:02d}"
        gen_dir.mkdir(exist_ok=True)
        for idx, params in enumerate(mapped):
            indiv_dir = gen_dir / f"individual_{idx+1:02d}"
            indiv_dir.mkdir(exist_ok=True)
            _, cfg_path = _prepare_individual(base_par, base_cfg, indiv_dir, params)
            if not _run_simulation(executable, cfg_path, indiv_dir):
                fitness.append(1e9)
                continue
            rg = _compute_rg(script, indiv_dir)
            if rg is None:
                fitness.append(1e9)
                continue
            diff = abs(rg - 36)
            fitness.append(diff)
            logging.info(
                "Gen %d Indiv %d: params=%s Rg=%.3f diff=%.3f",
                gen + 1,
                idx + 1,
                params,
                rg,
                diff,
            )
        es.tell(mapped, fitness)
        es.disp()
    best_params = es.result.xbest
    best_params_mapped = _map_to_valid(best_params, valid)
    result_par = output_dir / "best_parameters.par"
    updated = _update_parameters(lines, best_params_mapped)
    _write_lines(result_par, updated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize Emin parameters with CMA-ES")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=11)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--output_dir", type=Path, default=Path("3_simulations"))
    parser.add_argument("--executable", default="NAMD_2.14_Linux-x86_64-multicore")
    parser.add_argument("--base_par", type=Path, default=Path("0_parameters/base.par"))
    parser.add_argument("--base_cfg", type=Path, default=Path("0_parameters/config.namd"))
    parser.add_argument("--valid_values", type=Path, default=Path("0_parameters/valid_parameter_values.txt"))
    parser.add_argument("--rg_script", type=Path, default=Path("2_scripts/calculate_rg.py"))
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()

    logging.basicConfig(
        filename=str(args.log) if args.log else None,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_cma(
        base_par=args.base_par,
        base_cfg=args.base_cfg,
        valid_values=args.valid_values,
        output_dir=args.output_dir,
        executable=args.executable,
        script=args.rg_script,
        generations=args.generations,
        population=args.population,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()
