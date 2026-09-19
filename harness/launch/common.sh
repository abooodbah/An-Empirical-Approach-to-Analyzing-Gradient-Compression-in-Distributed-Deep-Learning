#!/usr/bin/env bash
# Shared, intentionally small helpers for the TACC paper-workload launchers.

set -Eeuo pipefail

launch_die() {
  printf 'FATAL: %s\n' "$*" >&2
  return 1
}

require_slurm_job() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || launch_die "run this file with sbatch, not on a login node"
  command -v srun >/dev/null 2>&1 || launch_die "srun is not available"
  command -v scontrol >/dev/null 2>&1 || launch_die "scontrol is not available"
}

project_root_from_launch_dir() {
  local launch_dir="$1"
  (cd "${launch_dir}/../../.." && pwd -P)
}

init_run_dir() {
  local platform="$1"
  local default_root="${SCRATCH:?SCRATCH is not set}/acm_pivot_v9/tacc_paper_workloads/runs"
  RUN_ROOT="${PAPER_RUN_ROOT:-$default_root}"
  mkdir -p "$RUN_ROOT"

  local staging
  staging="$(mktemp -d "${RUN_ROOT}/.${platform}-${SLURM_JOB_ID}.XXXXXX")"
  RUN_DIR="${RUN_ROOT}/${platform}-${SLURM_JOB_ID}"
  [[ ! -e "$RUN_DIR" ]] || launch_die "refusing to overwrite existing run directory: $RUN_DIR"

  mkdir -p "$staging/provenance" "$staging/logs" "$staging/results"
  {
    printf 'schema_version\t1\n'
    printf 'platform\t%s\n' "$platform"
    printf 'slurm_job_id\t%s\n' "$SLURM_JOB_ID"
    printf 'slurm_cluster\t%s\n' "${SLURM_CLUSTER_NAME:-unknown}"
    printf 'slurm_partition\t%s\n' "${SLURM_JOB_PARTITION:-unknown}"
    printf 'slurm_job_nodelist\t%s\n' "${SLURM_JOB_NODELIST:-unknown}"
    printf 'slurm_submit_dir\t%s\n' "${SLURM_SUBMIT_DIR:-unknown}"
    printf 'utc_started\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$staging/run_metadata.tsv"

  mv "$staging" "$RUN_DIR"
  export RUN_DIR RUN_ROOT
}

collect_base_provenance() {
  local project_root="$1"
  scontrol show job "$SLURM_JOB_ID" >"$RUN_DIR/provenance/slurm_job.txt"
  scontrol show hostnames "$SLURM_JOB_NODELIST" >"$RUN_DIR/provenance/hosts.txt"
  {
    printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
    printf 'SLURM_JOB_NAME=%s\n' "${SLURM_JOB_NAME:-unknown}"
    printf 'SLURM_JOB_NUM_NODES=%s\n' "${SLURM_JOB_NUM_NODES:-unknown}"
    printf 'SLURM_NTASKS=%s\n' "${SLURM_NTASKS:-unknown}"
    printf 'SLURM_TIMELIMIT=%s\n' "${SLURM_TIMELIMIT:-unknown}"
    printf 'WORK=%s\n' "${WORK:-unset}"
    printf 'SCRATCH=%s\n' "${SCRATCH:-unset}"
  } >"$RUN_DIR/provenance/environment_allowlist.txt"

  if type module >/dev/null 2>&1; then
    module -t list >"$RUN_DIR/provenance/modules.txt" 2>&1 || true
  fi
  if command -v git >/dev/null 2>&1 && git -C "$project_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$project_root" rev-parse HEAD >"$RUN_DIR/provenance/git_head.txt"
    git -C "$project_root" status --short >"$RUN_DIR/provenance/git_status_short.txt"
  fi
}

slurm_time_limit() {
  scontrol show job -o "$SLURM_JOB_ID" \
    | tr ' ' '\n' \
    | awk -F= '$1 == "TimeLimit" && !found {value=$2; found=1} END {if (found) print value; else exit 1}'
}

time_limit_seconds() {
  local value="$1"
  local days=0
  if [[ "$value" == *-* ]]; then
    days="${value%%-*}"
    value="${value#*-}"
  fi
  local first second third
  IFS=: read -r first second third <<<"$value"
  local hours=0 minutes=0 seconds=0
  if [[ -n "${third:-}" ]]; then
    hours="$first"
    minutes="$second"
    seconds="$third"
  elif [[ -n "${second:-}" ]]; then
    minutes="$first"
    seconds="$second"
  else
    minutes="$first"
  fi
  [[ "$days" =~ ^[0-9]+$ && "$hours" =~ ^[0-9]+$ && "$minutes" =~ ^[0-9]+$ && "$seconds" =~ ^[0-9]+$ ]] || \
    launch_die "cannot parse Slurm TimeLimit=$1"
  printf '%s\n' "$((days * 86400 + hours * 3600 + minutes * 60 + seconds))"
}

record_and_enforce_cost() {
  local nodes="$1"
  local rate="$2"
  local per_job_cap="$3"
  local limit seconds charge
  limit="$(slurm_time_limit)" || launch_die "cannot read allocated Slurm time limit"
  seconds="$(time_limit_seconds "$limit")"
  charge="$(awk -v n="$nodes" -v r="$rate" -v s="$seconds" 'BEGIN {printf "%.4f", n*r*s/3600.0}')"
  {
    printf 'allocated_time_limit\t%s\n' "$limit"
    printf 'allocated_nodes\t%s\n' "$nodes"
    printf 'billing_rate_su_per_node_hour\t%s\n' "$rate"
    printf 'allocated_charge_upper_bound_su\t%s\n' "$charge"
    printf 'launcher_per_job_su_cap\t%s\n' "$per_job_cap"
  } >>"$RUN_DIR/run_metadata.tsv"
  awk -v charge="$charge" -v cap="$per_job_cap" 'BEGIN {exit !(charge <= cap + 0.00001)}' || \
    launch_die "allocated charge bound ${charge} SU exceeds this launcher's ${per_job_cap}-SU cap"
}

hash_launch_inputs() {
  local output="$RUN_DIR/provenance/input_hashes.sha256"
  shift 0
  : >"$output"
  local path
  for path in "$@"; do
    [[ -f "$path" ]] || launch_die "required input does not exist: $path"
    sha256sum "$path" >>"$output"
  done
}

receipt_value() {
  local receipt="$1"
  local key="$2"
  awk -F '\t' -v wanted="$key" '$1 == wanted {print $2; found=1; exit} END {if (!found) exit 1}' "$receipt"
}

validate_receipt() {
  local receipt="$1"
  local expected_benchmark_sha="$2"
  [[ -r "$receipt" ]] || launch_die "qualification receipt is not readable: $receipt"
  [[ "$(receipt_value "$receipt" status)" == "PASS" ]] || launch_die "qualification receipt is not PASS: $receipt"
  [[ "$(receipt_value "$receipt" benchmark_sha256)" == "$expected_benchmark_sha" ]] || \
    launch_die "benchmark hash differs from the qualified benchmark"
}

finalize_manifest() {
  [[ -n "${RUN_DIR:-}" && -d "$RUN_DIR" ]] || return 0
  local tmp="${RUN_ROOT}/.manifest-${SLURM_JOB_ID}-$$"
  (
    cd "$RUN_DIR"
    find . -type f ! -name MANIFEST.sha256 -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 -r sha256sum
  ) >"$tmp"
  mv "$tmp" "$RUN_DIR/MANIFEST.sha256"
}

install_exit_receipt_trap() {
  trap 'launch_exit_receipt "$?"' EXIT
}

launch_exit_receipt() {
  local rc="$1"
  trap - EXIT
  set +e
  if [[ -n "${RUN_DIR:-}" && -d "$RUN_DIR" ]]; then
    local status=FAIL
    [[ "$rc" -eq 0 ]] && status=PASS
    {
      printf 'status\t%s\n' "$status"
      printf 'exit_code\t%s\n' "$rc"
      printf 'utc_finished\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >"$RUN_DIR/.status.tmp"
    mv "$RUN_DIR/.status.tmp" "$RUN_DIR/STATUS.tsv"
    finalize_manifest
  fi
  exit "$rc"
}
