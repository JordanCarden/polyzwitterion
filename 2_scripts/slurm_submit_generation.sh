#!/bin/bash
#SBATCH -J XPLODE_HPC
#SBATCH -A loni_membrane_c
#SBATCH -p workq
#SBATCH -N 3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -t 72:00:00
#SBATCH -o /work/jcarde7/polyzwitterion/3_simulations/generation_02/out_%j.txt
#SBATCH -e /work/jcarde7/polyzwitterion/3_simulations/generation_02/err_%j.txt
#SBATCH --distribution=block:block

NAMD=/work/jcarde7/polyzwitterion/NAMD_2.14_Linux-x86_64-multicore/namd2
BASE=/work/jcarde7/polyzwitterion/3_simulations/generation_02
RESUME_HELPER=/work/jcarde7/polyzwitterion/2_scripts/build_resume_config.sh
MAX_RETRIES=3
RETRY_SLEEP_SEC=30
SIMS=(child_01 child_02 child_03 child_04 child_05 child_06 child_07 child_08 child_09 child_10 child_11 child_12)

for s in "${SIMS[@]}"; do
  srun -N1 --ntasks=1 --cpus-per-task=16 --exclusive bash -lc "
set -euo pipefail
max_retries=3
retry_sleep_sec=30
for sys in 10mer 30mer 60mer; do
  run_dir=\"$BASE/$s/\$sys\"
  log=\"$BASE/$s/\$sys/output.txt\"
  attempt=1
  while true; do
    conf_path=\$(\"$RESUME_HELPER\" \"\$run_dir\")
    if [[ \"\$conf_path\" == \"__COMPLETE__\" ]]; then
      echo \"[skip] \$run_dir already completed run steps\" >>\"\$log\"
      break
    fi

    echo \"[$(date)] attempt \$attempt: starting \$run_dir with \$conf_path\" >>\"\$log\"
    if \"$NAMD\" +p16 \"\$conf_path\" >>\"\$log\" 2>&1; then
      echo \"[$(date)] success: \$run_dir finished on attempt \$attempt\" >>\"\$log\"
      break
    else
      status=\$?
    fi

    if (( attempt > max_retries )); then
      echo \"[$(date)] giving up: \$run_dir failed after \$attempt attempts (exit \$status)\" >>\"\$log\"
      exit \$status
    fi
    echo \"[$(date)] retrying: \$run_dir failed on attempt \$attempt (exit \$status), sleeping \$retry_sleep_sec s\" >>\"\$log\"
    attempt=\$((attempt + 1))
    sleep \$retry_sleep_sec
  done
done
" &
done
wait
