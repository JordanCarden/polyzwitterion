#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <run_dir> [config_name]" >&2
  exit 2
fi

run_dir="$1"
if [[ $# -eq 2 ]]; then
  config_name="$2"
else
  shopt -s nullglob
  candidates=()
  for f in "$run_dir"/*.namd; do
    [[ "$f" == *.autoresume.namd ]] && continue
    candidates+=("$f")
  done
  if (( ${#candidates[@]} != 1 )); then
    echo "config_name not provided and found ${#candidates[@]} non-autoresume .namd files in $run_dir" >&2
    exit 2
  fi
  config_name="$(basename "${candidates[0]}")"
fi
base_conf="$run_dir/$config_name"
resume_conf="$run_dir/${config_name%.namd}.autoresume.namd"

if [[ ! -f "$base_conf" ]]; then
  echo "$base_conf not found" >&2
  exit 1
fi

total_steps="$(awk '$1=="run"{print $2; exit}' "$base_conf")"
if [[ -z "${total_steps:-}" || ! "$total_steps" =~ ^[0-9]+$ ]]; then
  total_steps=500000
fi

first_step="$(awk '$1=="firsttimestep"{print $2; exit}' "$base_conf")"
if [[ -z "${first_step:-}" || ! "$first_step" =~ ^[0-9]+$ ]]; then
  first_step=0
fi
final_step=$((first_step + total_steps))

output_name="$(awk '$1=="set" && $2=="outputname"{print $3; exit}' "$base_conf")"
if [[ -z "${output_name:-}" ]]; then
  output_name="run_pr"
fi

restart_xsc="$run_dir/${output_name}.restart.xsc"
restart_coor="$run_dir/${output_name}.restart.coor"
restart_vel="$run_dir/${output_name}.restart.vel"

if [[ -s "$restart_xsc" && -s "$restart_coor" && -s "$restart_vel" ]]; then
  restart_step="$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "$restart_xsc")"
  if [[ "$restart_step" =~ ^[0-9]+$ ]]; then
    if (( restart_step >= final_step )); then
      echo "__COMPLETE__"
      exit 0
    fi

    remaining=$((final_step - restart_step))
    restart_xsc_name="${output_name}.restart.xsc"
    restart_coor_name="${output_name}.restart.coor"
    restart_vel_name="${output_name}.restart.vel"
    awk -v first="$restart_step" -v remaining="$remaining" \
        -v restart_xsc="$restart_xsc_name" \
        -v restart_coor="$restart_coor_name" \
        -v restart_vel="$restart_vel_name" '
    BEGIN { bc=0; bv=0; es=0; ft=0; rn=0 }
    {
      if ($1=="bincoordinates") { print "bincoordinates " restart_coor; bc=1; next }
      if ($1=="binvelocities")  { print "binvelocities  " restart_vel; bv=1; next }
      if ($1=="extendedSystem") { print "extendedSystem " restart_xsc; es=1; next }
      if ($1=="firsttimestep")  { printf "firsttimestep      %d\n", first; ft=1; next }
      if ($1=="run")            { printf "run %d\n", remaining; rn=1; next }
      print
    }
    END {
      if (!bc) print "bincoordinates " restart_coor
      if (!bv) print "binvelocities  " restart_vel
      if (!es) print "extendedSystem " restart_xsc
      if (!ft) printf "firsttimestep      %d\n", first
      if (!rn) printf "run %d\n", remaining
    }' "$base_conf" > "$resume_conf"

    echo "$resume_conf"
    exit 0
  fi
fi

echo "$base_conf"
