from __future__ import annotations

import bisect
import csv
import shutil
import subprocess
import time
from pathlib import Path
from typing import List

import cma
import MDAnalysis as mda
import numpy as np

LOGIN = "jcarde7@qbd.loni.org"
BASE = Path("/work/jcarde7/polyzwitterion")
POP = 12
TARGET_RG = 36.0
GEN_MAX = 30

PAIRS = [
    ("W", "N1"), ("W", "C1"), ("W", "Q1"), ("W", "Q2"),
    ("C1", "N1"), ("C1", "Q1"), ("C1", "Q2"),
    ("N1", "Q1"), ("N1", "Q2"), ("Q1", "Q2"),
]

REPO = Path(__file__).resolve().parent.parent
NAMD = BASE / "NAMD_2.14_Linux-x86_64-multicore-CUDA/namd2"


def load_allowed() -> list[float]:
    txt = REPO / "0_parameters" / "valid_parameter_values.txt"
    return sorted(abs(float(x)) for x in txt.read_text().split())


ALLOWED = load_allowed()
LOW, HIGH = ALLOWED[0], ALLOWED[-1]
INIT_EPS = [np.median(ALLOWED)] * 10


def snap_to_allowed(vec: list[float]) -> list[float]:
    out = []
    for v in vec:
        mag = abs(v)
        idx = bisect.bisect_left(ALLOWED, mag)
        if idx == 0:
            out.append(ALLOWED[0])
        elif idx == len(ALLOWED):
            out.append(ALLOWED[-1])
        else:
            left, right = ALLOWED[idx - 1], ALLOWED[idx]
            out.append(right if right - mag < mag - left else left)
    return out


def create_generation(gen: int) -> Path:
    gdir = REPO / "3_simulations" / f"generation_{gen:02d}"
    gdir.mkdir(parents=True, exist_ok=True)
    for i in range(1, POP + 1):
        cdir = gdir / f"child_{i:02d}"
        cdir.mkdir(exist_ok=True)
        for src in ("nbfix.par", "config.namd"):
            shutil.copy(REPO / "0_parameters" / src, cdir)
    return gdir


def edit_nbfix(path: Path, eps: list[float]) -> None:
    repl = {p: f"-{v:.8f}" for p, v in zip(PAIRS, eps)}
    out: List[str] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and (parts[0], parts[1]) in repl:
            parts[2] = repl[(parts[0], parts[1])]
            line = "   ".join(parts)
        out.append(line)
    path.write_text("\n".join(out) + "\n")


def write_slurm(gen: int) -> None:
    script = REPO / "2_scripts" / "submit.sh"
    base = BASE / "3_simulations" / f"generation_{gen:02d}"
    sims = " ".join(f"child_{i:02d}" for i in range(1, POP + 1))
    script.write_text(f"""#!/bin/bash
#SBATCH -J XPLODE_HPC
#SBATCH -A loni_pdrug
#SBATCH -p gpu4
#SBATCH -N 3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -t 72:00:00
#SBATCH -o {base}/out_%j.txt
#SBATCH -e {base}/err_%j.txt
#SBATCH --distribution=block:block

module load cuda
NAMD={NAMD}
BASE={base}
SIMS=({sims})

for s in "${{SIMS[@]}}"; do
  conf="$BASE/$s/config.namd"
  log="$BASE/$s/output.txt"
  srun -N1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --exclusive "$NAMD" +p16 "$conf" >"$log" 2>&1 &
done
wait
""")
    script.chmod(0o755)


def submit_and_wait() -> None:
    remote = BASE / "2_scripts" / "submit.sh"
    job_id = subprocess.check_output(
        ["ssh", LOGIN, "sbatch", str(remote)], text=True
    ).strip().split()[-1]
    while subprocess.check_output(
        ["ssh", LOGIN, "squeue", "-h", "-j", job_id], text=True
    ).strip():
        time.sleep(60)


def rg_last5(child: Path) -> float:
    psf = REPO / "1_input" / "polymer_drug_solvate.psf"
    dcd = child / "run_pr.dcd"
    u = mda.Universe(psf, dcd)
    sel = u.select_atoms("not (name W HOH)")
    values = []
    for _ in u.trajectory[-5:]:
        com = sel.center_of_mass()
        diff = sel.positions - com
        sq = np.sum(diff ** 2, axis=1)
        values.append(np.sqrt(np.sum(sel.masses * sq) / np.sum(sel.masses)))
    return float(np.mean(values))


def log_csv(path: Path, gen: int, child: int,
            eps: list[float], rg: float, err: float) -> None:
    header = ["gen", "child", "eps", "Rg", "err"]
    write_header = not path.exists()
    eps_vec = "[" + " ".join(f"{e:.6f}" for e in eps) + "]"
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([gen, child, eps_vec, f"{rg:.6f}", f"{err:.6f}"])


def main() -> None:
    es = cma.CMAEvolutionStrategy(
        INIT_EPS, 0.2, {"popsize": POP, "bounds": [[LOW] * 10, [HIGH] * 10]}
    )
    csv_file = REPO / "optimisation_results.csv"
    gen = 1
    while not es.stop() and gen <= GEN_MAX:
        gdir = create_generation(gen)
        raw = es.ask()
        vecs = [snap_to_allowed(v) for v in raw]
        for idx, v in enumerate(vecs, 1):
            edit_nbfix(gdir / f"child_{idx:02d}" / "nbfix.par", v)
        write_slurm(gen)
        submit_and_wait()
        fitness = []
        for idx in range(1, POP + 1):
            child = gdir / f"child_{idx:02d}"
            rg = rg_last5(child)
            err = abs(rg - TARGET_RG)
            fitness.append(err)
            log_csv(csv_file, gen, idx, vecs[idx - 1], rg, err)
            print(f"gen {gen:02d} child {idx:02d}  Rg={rg:6.2f}  |Δ|={err:5.2f}  "
                  f"eps=[{' '.join(f'{e:.2f}' for e in vecs[idx - 1])}]")
        es.tell(vecs, fitness)
        gen += 1
    print("best ε:", es.result.xbest, "fitness:", es.result.fbest)


if __name__ == "__main__":
    main()
