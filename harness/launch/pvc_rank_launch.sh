#!/usr/bin/env bash
# Convert one Slurm task per PVC tile into torch.distributed env:// variables.

set -Eeuo pipefail

: "${SLURM_PROCID:?SLURM_PROCID is required}"
: "${SLURM_NTASKS:?SLURM_NTASKS is required}"
: "${SLURM_LOCALID:?SLURM_LOCALID is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"

(( SLURM_LOCALID < 8 )) || { printf 'FATAL: local rank %s exceeds eight PVC tiles/node\n' "$SLURM_LOCALID" >&2; exit 1; }

export RANK="$SLURM_PROCID"
export WORLD_SIZE="$SLURM_NTASKS"
export LOCAL_RANK="$SLURM_LOCALID"
export GROUP_RANK="${SLURM_NODEID:-0}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-8}"

# oneCCL 1.13 otherwise guesses a worker CPU outside Slurm's task cpuset on
# Stampede3 and pthread_create fails with EINVAL. Bind its single worker to an
# actually allowed CPU and provide the local topology that the srun launcher
# does not expose using oneCCL's Hydra variables.
allowed_cpus="$(awk '/^Cpus_allowed_list:/ {print $2; exit}' /proc/self/status)"
[[ -n "$allowed_cpus" ]] || {
  printf 'FATAL: cannot determine the Slurm task CPU affinity\n' >&2
  exit 1
}
first_cpu="${allowed_cpus%%,*}"
first_cpu="${first_cpu%%-*}"
[[ "$first_cpu" =~ ^[0-9]+$ ]] || {
  printf 'FATAL: cannot parse first allowed CPU from %s\n' "$allowed_cpus" >&2
  exit 1
}
export CCL_WORKER_COUNT="${CCL_WORKER_COUNT:-1}"
export CCL_WORKER_AFFINITY="${CCL_WORKER_AFFINITY:-$first_cpu}"
export CCL_LOCAL_RANK="${CCL_LOCAL_RANK:-$LOCAL_RANK}"
export CCL_LOCAL_SIZE="${CCL_LOCAL_SIZE:-$LOCAL_WORLD_SIZE}"

exec "$@"
