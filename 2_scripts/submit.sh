#!/bin/bash
#SBATCH -J XPLODE_HPC
#SBATCH -A loni_pdrug
#SBATCH -p workq
#SBATCH -N 3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -t 72:00:00
#SBATCH -o /work/jcarde7/polyzwitterion/3_simulations/generation_01/out_%j.txt
#SBATCH -e /work/jcarde7/polyzwitterion/3_simulations/generation_01/err_%j.txt
#SBATCH --distribution=block:block

NAMD=/work/jcarde7/polyzwitterion/NAMD_2.14_Linux-x86_64-multicore/namd2
BASE=/work/jcarde7/polyzwitterion/3_simulations/generation_01
SIMS=(child_01 child_02 child_03 child_04 child_05 child_06 child_07 child_08 child_09 child_10 child_11 child_12)

for s in "${SIMS[@]}"; do
  conf="$BASE/$s/config.namd"
  log="$BASE/$s/output.txt"
  srun -N1 --ntasks=1 --cpus-per-task=16 \
       --exclusive "$NAMD" +p16 "$conf" >"$log" 2>&1 &
done
wait
