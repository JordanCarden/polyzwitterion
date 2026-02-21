import csv
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cma
import MDAnalysis as mda
import numpy as np
import paramiko


BASE = Path("/work/jcarde7/polyzwitterion")
REPO = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO / "1_templates"
COMMON_DIR = TEMPLATES_DIR / "common"
NAMD = BASE / "NAMD_2.14_Linux-x86_64-multicore/namd2"

POP = 12
GEN_MAX = 30
ES_SIGMA = 0.03
PENALTY_LOSS = 1_000.0

FORCE_FIELD_FILE = "ff_nbfix_opt.par"
PBE_FILE = "ff_pbe_cg.par"
LAST_FRAMES = 1000
SELECT_EXPR = "not (resname HOH ION IAN)"

BIN_EDGES = np.arange(0.0, 40.0 + 0.5, 0.5)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

PAIR_SPECS = [
    (("W", "N1"), -0.74423518165, -0.60891969407),
    (("W", "C1"), -0.48658699808, -0.39811663480),
    (("C1", "TQ5"), -0.57492351817, -0.47039196941),
    (("N1", "TQ5"), -2.03095124283, -1.66168738049),
    (("TQ5", "Q1"), -1.09106118547, -0.89268642447),
    (("TQ5", "Q2"), -1.61298279158, -1.31971319312),
]
LOW_BOUNDS = [low for _, low, _ in PAIR_SPECS]
HIGH_BOUNDS = [high for _, _, high in PAIR_SPECS]
PAIR_ORDER = [pair for pair, _, _ in PAIR_SPECS]

_SSH_CLIENT: paramiko.SSHClient | None = None


@dataclass(frozen=True)
class SystemSpec:
    """Definition for one target system in the multi-objective fit."""

    name: str
    template_dir: Path
    target_distribution: Path
    weight: float
    output_dcd: str


SYSTEMS = [
    SystemSpec(
        name="10mer",
        template_dir=TEMPLATES_DIR / "10mer",
        target_distribution=TEMPLATES_DIR / "10mer" / "target_rg_hist.dat",
        weight=0.25,
        output_dcd="state_continue.dcd",
    ),
    SystemSpec(
        name="30mer",
        template_dir=TEMPLATES_DIR / "30mer",
        target_distribution=TEMPLATES_DIR / "30mer" / "target_rg_hist.dat",
        weight=0.35,
        output_dcd="state_continue.dcd",
    ),
    SystemSpec(
        name="60mer",
        template_dir=TEMPLATES_DIR / "60mer",
        target_distribution=TEMPLATES_DIR / "60mer" / "target_rg_hist.dat",
        weight=0.40,
        output_dcd="state_continue.dcd",
    ),
]


def _get_ssh() -> paramiko.SSHClient:
    """Return a connected singleton SSH client.

    Returns:
        A connected `paramiko.SSHClient`.
    """
    global _SSH_CLIENT
    if _SSH_CLIENT is None:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect("qbd.loni.org", username="jcarde7")
        _SSH_CLIENT = client
    return _SSH_CLIENT


def _ssh_exec(cmd: str) -> str:
    """Execute *cmd* remotely and return stdout.

    Args:
        cmd: Command to run on the HPC login node.

    Returns:
        Standard output with trailing whitespace stripped.
    """
    _, stdout, _ = _get_ssh().exec_command(cmd)
    return stdout.read().decode().strip()


def load_distribution(path: Path) -> np.ndarray:
    """Load and normalize a binned Rg distribution onto canonical bins."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(1, 2)

    probs = np.zeros_like(BIN_CENTERS, dtype=float)
    for rg_value, prob in raw:
        idx = int(round((rg_value - BIN_CENTERS[0]) / 0.5))
        if 0 <= idx < len(probs) and np.isclose(BIN_CENTERS[idx], rg_value):
            probs[idx] += prob
        else:
            nearest = int(np.argmin(np.abs(BIN_CENTERS - rg_value)))
            probs[nearest] += prob

    total = float(np.sum(probs))
    if total > 0:
        probs /= total
    return probs


def load_initial_eps(path: Path) -> list[float]:
    """Load starting epsilon values for the six optimized NBFIX pairs."""
    values: dict[tuple[str, str], float] = {}
    wanted = set(PAIR_ORDER)
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0]
        parts = line.split()
        if len(parts) < 4:
            continue
        pair = (parts[0], parts[1])
        if pair in wanted:
            values[pair] = float(parts[2])

    missing = [pair for pair in PAIR_ORDER if pair not in values]
    if missing:
        raise ValueError(f"Missing pair(s) in force field: {missing}")
    return [values[pair] for pair in PAIR_ORDER]


def clamp_eps(vec: np.ndarray | list[float]) -> list[float]:
    """Clamp a candidate vector to fixed pairwise bounds."""
    clipped = []
    for v, low, high in zip(vec, LOW_BOUNDS, HIGH_BOUNDS):
        clipped.append(float(np.clip(v, low, high)))
    return clipped


def rewrite_force_field(path: Path, eps: list[float]) -> None:
    """Rewrite only the six target pair epsilons in the force-field file."""
    if len(eps) != len(PAIR_ORDER):
        raise ValueError("epsilon vector length mismatch")

    repl = {pair: value for pair, value in zip(PAIR_ORDER, eps)}
    out: list[str] = []
    for line in path.read_text().splitlines():
        head, hashmark, tail = line.partition("#")
        parts = head.split()
        if len(parts) >= 4 and (parts[0], parts[1]) in repl:
            parts[2] = f"{repl[(parts[0], parts[1])]:.11f}"
            rewritten = "  ".join(parts)
            line = f"{rewritten}  {hashmark}{tail}" if hashmark else rewritten
        out.append(line)
    path.write_text("\n".join(out) + "\n")


def stage_system(system: SystemSpec, dest: Path) -> None:
    """Prepare one system run directory with config/input/parameter files."""
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy(COMMON_DIR / FORCE_FIELD_FILE, dest / FORCE_FIELD_FILE)
    shutil.copy(COMMON_DIR / PBE_FILE, dest / PBE_FILE)
    shutil.copy(system.template_dir / "system.psf", dest / "system.psf")
    shutil.copy(system.template_dir / "system.pdb", dest / "system.pdb")
    shutil.copy(system.template_dir / "state_init.restart.coor", dest / "state_init.restart.coor")
    shutil.copy(system.template_dir / "state_init.restart.vel", dest / "state_init.restart.vel")
    shutil.copy(system.template_dir / "state_init.restart.xsc", dest / "state_init.restart.xsc")
    shutil.copy(system.template_dir / "cfg.namd", dest / "cfg.namd")


def create_generation(gen: int) -> Path:
    """Create directory tree for one generation with 3 system runs per child."""
    gdir = REPO / "3_simulations" / f"generation_{gen:02d}"
    gdir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, POP + 1):
        child = gdir / f"child_{idx:02d}"
        child.mkdir(exist_ok=True)
        for system in SYSTEMS:
            stage_system(system, child / system.name)
    return gdir


def apply_eps_to_child(child: Path, eps: list[float]) -> None:
    """Apply one epsilon vector to all system force-field files in a child."""
    for system in SYSTEMS:
        rewrite_force_field(child / system.name / FORCE_FIELD_FILE, eps)


def write_slurm(gen: int) -> None:
    """Generate SLURM script to run 10/30/60 mer trajectories per child."""
    script = REPO / "2_scripts" / "slurm_submit_generation.sh"
    base = BASE / "3_simulations" / f"generation_{gen:02d}"
    sims = " ".join(f"child_{i:02d}" for i in range(1, POP + 1))
    system_loop = " ".join(system.name for system in SYSTEMS)
    script.write_text(
        f"""#!/bin/bash
#SBATCH -J XPLODE_HPC
#SBATCH -A loni_membrane_c
#SBATCH -p workq
#SBATCH -N 3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -t 72:00:00
#SBATCH -o {base}/out_%j.txt
#SBATCH -e {base}/err_%j.txt
#SBATCH --distribution=block:block

NAMD={NAMD}
BASE={base}
RESUME_HELPER={BASE / "2_scripts" / "build_resume_config.sh"}
MAX_RETRIES=3
RETRY_SLEEP_SEC=30
SIMS=({sims})

for s in "${{SIMS[@]}}"; do
  srun -N1 --ntasks=1 --cpus-per-task=16 --exclusive bash -lc "
set -euo pipefail
max_retries=3
retry_sleep_sec=30
for sys in {system_loop}; do
  run_dir=\\"$BASE/$s/\\$sys\\"
  log=\\"$BASE/$s/\\$sys/output.txt\\"
  attempt=1
  while true; do
    conf_path=\\$(\\"$RESUME_HELPER\\" \\"\\$run_dir\\")
    if [[ \\"\\$conf_path\\" == \\"__COMPLETE__\\" ]]; then
      echo \\"[skip] \\$run_dir already completed run steps\\" >>\\"\\$log\\"
      break
    fi

    echo \\"[$(date)] attempt \\$attempt: starting \\$run_dir with \\$conf_path\\" >>\\"\\$log\\"
    if \\"$NAMD\\" +p16 \\"\\$conf_path\\" >>\\"\\$log\\" 2>&1; then
      echo \\"[$(date)] success: \\$run_dir finished on attempt \\$attempt\\" >>\\"\\$log\\"
      break
    else
      status=\\$?
    fi

    if (( attempt > max_retries )); then
      echo \\"[$(date)] giving up: \\$run_dir failed after \\$attempt attempts (exit \\$status)\\" >>\\"\\$log\\"
      exit \\$status
    fi
    echo \\"[$(date)] retrying: \\$run_dir failed on attempt \\$attempt (exit \\$status), sleeping \\$retry_sleep_sec s\\" >>\\"\\$log\\"
    attempt=\\$((attempt + 1))
    sleep \\$retry_sleep_sec
  done
done
" &
done
wait
"""
    )
    script.chmod(0o755)


def submit_and_wait() -> None:
    """Submit the SLURM job and block until completion."""
    remote = BASE / "2_scripts" / "slurm_submit_generation.sh"
    job_id = _ssh_exec(f"sbatch {remote}").split()[-1]
    while _ssh_exec(f"squeue -h -j {job_id}"):
        time.sleep(300)


def calculate_distribution(run_dir: Path, dcd_name: str) -> tuple[np.ndarray, float, float]:
    """Calculate normalized Rg distribution and save to `Rg_distribution.dat`."""
    psf = run_dir / "system.psf"
    dcd = run_dir / dcd_name
    universe = mda.Universe(psf, dcd)
    selection = universe.select_atoms(SELECT_EXPR)

    frame_count = len(universe.trajectory)
    start = max(0, frame_count - LAST_FRAMES)
    rg_values: list[float] = []
    for _ in universe.trajectory[start:]:
        com = selection.center_of_mass()
        diff = selection.positions - com
        squared = np.sum(diff**2, axis=1)
        rg = np.sqrt(np.sum(selection.masses * squared) / np.sum(selection.masses))
        rg_values.append(float(rg))

    counts, _ = np.histogram(rg_values, bins=BIN_EDGES)
    probs = counts.astype(float)
    if np.sum(probs) == 0:
        raise ValueError(f"No Rg counts in {run_dir}")
    probs /= np.sum(probs)

    np.savetxt(
        run_dir / "Rg_distribution.dat",
        np.column_stack((BIN_CENTERS, probs)),
        fmt="%.2f %.12f",
    )
    peak = float(BIN_CENTERS[int(np.argmax(probs))])
    mean = float(np.mean(rg_values))
    return probs, peak, mean


def peak_loss(sim_peak: float, target_peak: float) -> float:
    """Peak-only loss based on absolute peak-position mismatch."""
    return float(abs(sim_peak - target_peak))


def evaluate_child(
    child: Path, targets: dict[str, np.ndarray]
) -> tuple[float, dict[str, dict[str, float]]]:
    """Evaluate one child candidate across all systems."""
    total_loss = 0.0
    metrics: dict[str, dict[str, float]] = {}
    for system in SYSTEMS:
        target = targets[system.name]
        target_peak = float(BIN_CENTERS[int(np.argmax(target))])
        try:
            _, sim_peak, sim_mean = calculate_distribution(child / system.name, system.output_dcd)
            loss = peak_loss(sim_peak, target_peak)
        except Exception as exc:  # keep optimizer alive on failed child runs
            print(f"failed {child.name}/{system.name}: {exc}")
            loss = PENALTY_LOSS
            sim_peak = float("nan")
            sim_mean = float("nan")
        weighted = system.weight * loss
        total_loss += weighted
        metrics[system.name] = {
            "loss": loss,
            "weighted_loss": weighted,
            "sim_peak": sim_peak,
            "target_peak": target_peak,
            "sim_mean": sim_mean,
        }
    return total_loss, metrics


def log_csv(
    gen: int,
    child: int,
    eps: list[float],
    total_loss: float,
    metrics: dict[str, dict[str, float]],
) -> None:
    """Append one optimization record to summary and expanded CSV files."""
    eps_vec = "[" + " ".join(f"{e:.6f}" for e in eps) + "]"

    summary_path = REPO / "optimisation_results.csv"
    summary_header = [
        "gen",
        "child",
        "eps",
        "total_loss",
        "loss_60mer",
        "loss_30mer",
        "loss_10mer",
    ]
    with summary_path.open("a", newline="") as f:
        w = csv.writer(f)
        if summary_path.stat().st_size == 0:
            w.writerow(summary_header)
        w.writerow(
            [
                gen,
                child,
                eps_vec,
                f"{total_loss:.10f}",
                f"{metrics['60mer']['loss']:.10f}",
                f"{metrics['30mer']['loss']:.10f}",
                f"{metrics['10mer']['loss']:.10f}",
            ]
        )

    expanded_path = REPO / "optimisation_results_expanded.csv"
    expanded_header = [
        "gen",
        "child",
        "eps",
        "total_loss",
        "loss_60mer",
        "weighted_loss_60mer",
        "peak_sim_60mer",
        "peak_target_60mer",
        "mean_sim_60mer",
        "loss_30mer",
        "weighted_loss_30mer",
        "peak_sim_30mer",
        "peak_target_30mer",
        "mean_sim_30mer",
        "loss_10mer",
        "weighted_loss_10mer",
        "peak_sim_10mer",
        "peak_target_10mer",
        "mean_sim_10mer",
    ]
    with expanded_path.open("a", newline="") as f:
        w = csv.writer(f)
        if expanded_path.stat().st_size == 0:
            w.writerow(expanded_header)
        w.writerow(
            [
                gen,
                child,
                eps_vec,
                f"{total_loss:.10f}",
                f"{metrics['60mer']['loss']:.10f}",
                f"{metrics['60mer']['weighted_loss']:.10f}",
                f"{metrics['60mer']['sim_peak']:.6f}",
                f"{metrics['60mer']['target_peak']:.6f}",
                f"{metrics['60mer']['sim_mean']:.6f}",
                f"{metrics['30mer']['loss']:.10f}",
                f"{metrics['30mer']['weighted_loss']:.10f}",
                f"{metrics['30mer']['sim_peak']:.6f}",
                f"{metrics['30mer']['target_peak']:.6f}",
                f"{metrics['30mer']['sim_mean']:.6f}",
                f"{metrics['10mer']['loss']:.10f}",
                f"{metrics['10mer']['weighted_loss']:.10f}",
                f"{metrics['10mer']['sim_peak']:.6f}",
                f"{metrics['10mer']['target_peak']:.6f}",
                f"{metrics['10mer']['sim_mean']:.6f}",
            ]
        )


def log_generation_timing(gen: int, wall_seconds: float) -> None:
    """Append per-generation wall-clock runtime to a CSV file."""
    path = REPO / "optimisation_generation_timing.csv"
    header = ["gen", "wall_seconds", "wall_minutes"]
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if path.stat().st_size == 0:
            w.writerow(header)
        w.writerow([gen, f"{wall_seconds:.3f}", f"{wall_seconds / 60.0:.3f}"])


def write_runtime_summary(total_seconds: float, completed_generations: int) -> None:
    """Write total optimization runtime summary."""
    (REPO / "optimisation_runtime_summary.txt").write_text(
        "\n".join(
            [
                f"completed_generations: {completed_generations}",
                f"total_wall_seconds: {total_seconds:.3f}",
                f"total_wall_minutes: {total_seconds / 60.0:.3f}",
                f"total_wall_hours: {total_seconds / 3600.0:.3f}",
            ]
        )
        + "\n"
    )


def main() -> None:
    """Run global CMA-ES over 6 epsilons for weighted 10/30/60 distribution fit."""
    targets = {system.name: load_distribution(system.target_distribution) for system in SYSTEMS}
    init_eps = load_initial_eps(COMMON_DIR / FORCE_FIELD_FILE)

    (REPO / "optimisation_results.csv").write_text("")
    (REPO / "optimisation_results_expanded.csv").write_text("")
    (REPO / "optimisation_generation_timing.csv").write_text("")

    total_start = time.perf_counter()
    completed_generations = 0

    es = cma.CMAEvolutionStrategy(
        init_eps,
        ES_SIGMA,
        {"popsize": POP, "bounds": [LOW_BOUNDS, HIGH_BOUNDS]},
    )
    gen = 1

    while not es.stop() and gen <= GEN_MAX:
        generation_start = time.perf_counter()
        gdir = create_generation(gen)
        raw = es.ask()
        vecs = [clamp_eps(v) for v in raw]
        for idx, v in enumerate(vecs, 1):
            apply_eps_to_child(gdir / f"child_{idx:02d}", v)

        write_slurm(gen)
        submit_and_wait()

        fitness: list[float] = []
        for idx in range(1, POP + 1):
            child = gdir / f"child_{idx:02d}"
            total_loss, metrics = evaluate_child(child, targets)
            fitness.append(total_loss)
            log_csv(gen, idx, vecs[idx - 1], total_loss, metrics)
            print(
                f"gen {gen:02d} child {idx:02d}  "
                f"total={total_loss:.6f}  "
                f"60={metrics['60mer']['loss']:.6f}  "
                f"30={metrics['30mer']['loss']:.6f}  "
                f"10={metrics['10mer']['loss']:.6f}  "
                f"eps=[{' '.join(f'{e:.4f}' for e in vecs[idx - 1])}]"
            )

        es.tell(vecs, fitness)
        generation_seconds = time.perf_counter() - generation_start
        log_generation_timing(gen, generation_seconds)
        print(f"gen {gen:02d} runtime={generation_seconds / 60.0:.2f} min")
        completed_generations += 1
        gen += 1

    total_seconds = time.perf_counter() - total_start
    write_runtime_summary(total_seconds, completed_generations)
    print(f"total optimization runtime={total_seconds / 3600.0:.2f} h")
    print("best ε:", es.result.xbest, "fitness:", es.result.fbest)
    if _SSH_CLIENT is not None:
        _SSH_CLIENT.close()


if __name__ == "__main__":
    main()
