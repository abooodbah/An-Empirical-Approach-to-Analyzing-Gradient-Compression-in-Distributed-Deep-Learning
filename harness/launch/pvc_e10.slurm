#!/usr/bin/env bash
#SBATCH --job-name=redctl-e10
#SBATCH --account=OTH21057
#SBATCH --partition=pvc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:55:00
#SBATCH --export=ALL

# E10 (diagnostic/E10_PROTOCOL.md): a bucketed gather on the 1B workload at
# 8, 16, and 32 ranks with its memory account. RED_SET=s1 (2 nodes) or s2
# (4 nodes). Stack and plumbing as in pvc_v3.slurm; the harness is v5.2.

set -Eeuo pipefail

SCRIPT_DIR="${RED_LAUNCH_DIR:?set RED_LAUNCH_DIR to the staged launch directory}"
[[ -r "$SCRIPT_DIR/common.sh" ]] || { printf 'FATAL: common.sh not found at %s\n' "$SCRIPT_DIR" >&2; exit 1; }
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd -P)"
source "$SCRIPT_DIR/common.sh"
require_slurm_job

RED_ROOT="${RED_ROOT:?set RED_ROOT to the staged campaign root}"
BENCHMARK="$RED_ROOT/control_workload_benchmark_v5_2.py"
COLLECTIVE_PROBE="$RED_ROOT/collective_probe.py"
PROBE="$SCRIPT_DIR/pvc_ccl_exact_probe.py"
RED_SET="${RED_SET:?set RED_SET to s1 or s2}"
RED_TAG="${RED_TAG:-E10}"
RED_SU_CAP="${RED_SU_CAP:-5.5}"
PVC_MODULE="${PVC_MODULE:-intel/23.1}"
PVC_VENV="${PVC_VENV:-/scratch/projects/compilers/intel23.1/intelpython/python3.9/envs/pytorch-gpu}"
PVC_PYTHON="$PVC_VENV/bin/python"
CAMPAIGN_ID="${CAMPAIGN_ID:-redctl-e10-2026-10}"
PAPER_SEED="${PAPER_SEED:-2026010043}"

case "$RED_SET" in s1|s2) ;; *) launch_die "RED_SET must be s1 or s2" ;; esac
[[ -f "$BENCHMARK" ]] || launch_die "missing benchmark: $BENCHMARK"
[[ -f "$PROBE" ]] || launch_die "missing probe: $PROBE"
[[ -x "$PVC_PYTHON" ]] || launch_die "PVC python is not executable: $PVC_PYTHON"

NODES="${SLURM_JOB_NUM_NODES:?}"
WORLD=$((NODES * 8))

init_run_dir "redctl-${RED_TAG}-${NODES}n"
install_exit_receipt_trap
printf 'RUN_DIR=%s\n' "$RUN_DIR"
collect_base_provenance "$RED_ROOT"
record_and_enforce_cost "$NODES" 3 "$RED_SU_CAP"
hash_launch_inputs "$SCRIPT_DIR/common.sh" "$SCRIPT_DIR/pvc_rank_launch.sh" "$PROBE" "${BASH_SOURCE[0]}" "$BENCHMARK"
want="$(cut -d' ' -f1 "$RED_ROOT/CODE_v52.sha256")"
[[ "$(sha256sum "$BENCHMARK" | cut -d' ' -f1)" == "$want" ]] || launch_die "harness v5.2 differs from CODE_v52.sha256"
mkdir -p "$RUN_DIR/probes" "$RUN_DIR/progress"

set +u
module purge >"$RUN_DIR/provenance/intel_module.out" 2>&1
module load "$PVC_MODULE" >>"$RUN_DIR/provenance/intel_module.out" 2>&1
set -u
PVC_SITE_PACKAGES="$PVC_VENV/lib/python3.9/site-packages"
export LD_LIBRARY_PATH="$PVC_SITE_PACKAGES/torch/lib:$PVC_SITE_PACKAGES/intel_extension_for_pytorch/lib:$PVC_VENV/lib:${LD_LIBRARY_PATH:-}"

{
  printf 'campaign_set\t%s\n' "$RED_SET"
  printf 'allocation_tag\t%s\n' "$RED_TAG"
  printf 'campaign_id\t%s\n' "$CAMPAIGN_ID"
  printf 'paper_seed\t%s\n' "$PAPER_SEED"
  printf 'nodes\t%s\n' "$NODES"
  printf 'world_size\t%s\n' "$WORLD"
  printf 'benchmark_sha256\t%s\n' "$(sha256sum "$BENCHMARK" | awk '{print $1}')"
  printf 'intel_collective_path\toneccl_host_staged_ofi_tcp\n'
} >>"$RUN_DIR/run_metadata.tsv"

export CCL_LOG_LEVEL="${CCL_LOG_LEVEL:-info}"
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-ofi}"
export CCL_ALLREDUCE="${CCL_ALLREDUCE:-ring}"
export CCL_ALLGATHERV="${CCL_ALLGATHERV:-naive}"
export CCL_BARRIER="${CCL_BARRIER:-ring}"
export CCL_BCAST="${CCL_BCAST:-naive}"
export CCL_REDUCE_SCATTER="${CCL_REDUCE_SCATTER:-ring}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export FI_LOG_LEVEL="${FI_LOG_LEVEL:-warn}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export FI_PROVIDER="${PVC_FI_PROVIDER:-tcp}"
export FI_TCP_IFACE="${PVC_FI_TCP_IFACE:-ibp56s0}"

mapfile -t hosts <"$RUN_DIR/provenance/hosts.txt"
[[ "${#hosts[@]}" -eq "$NODES" ]] || launch_die "expected $NODES allocated PVC nodes"
export MASTER_ADDR="${hosts[0]}"

join_hosts() {
  local count="$1"
  local joined="${hosts[0]}"
  local index
  for ((index = 1; index < count; index++)); do
    joined+=",${hosts[index]}"
  done
  printf '%s\n' "$joined"
}

export MASTER_PORT=$((31700 + SLURM_JOB_ID % 500))
srun --exclusive --nodes=1 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=4 \
  --cpu-bind=cores --nodelist="${hosts[0]}" \
  --kill-on-bad-exit=1 --wait=0 \
  --output="$RUN_DIR/logs/affinity-discovery.%t.out" \
  --error="$RUN_DIR/logs/affinity-discovery.%t.err" \
  bash "$SCRIPT_DIR/pvc_rank_launch.sh" "$PVC_PYTHON" "$PROBE" --mode tile \
  --output-dir "$RUN_DIR/provenance/affinity-tiles"
[[ "$(find "$RUN_DIR/provenance/affinity-tiles" -name 'tile_*.json' -type f | wc -l)" -eq 8 ]] || \
  launch_die "affinity discovery did not produce eight tile receipts"
export CCL_WORKER_AFFINITY="$("$PVC_PYTHON" "$PROBE" --mode affinity --output-dir "$RUN_DIR/provenance/affinity-tiles")"
printf 'ccl_worker_affinity\t%s\n' "$CCL_WORKER_AFFINITY" >>"$RUN_DIR/run_metadata.tsv"

if (( NODES > 1 )); then
  export MASTER_PORT=$((31900 + SLURM_JOB_ID % 500))
  srun --exclusive --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=4 \
    --cpu-bind=cores \
    --nodelist="$(join_hosts 2)" --distribution=block:block --kill-on-bad-exit=1 --wait=0 \
    --output="$RUN_DIR/logs/inter-node-ccl-gate.%t.out" \
    --error="$RUN_DIR/logs/inter-node-ccl-gate.%t.err" \
    bash "$SCRIPT_DIR/pvc_rank_launch.sh" "$PVC_PYTHON" "$PROBE" --mode ccl \
    --output-dir "$RUN_DIR/provenance/inter-node-ccl-gate"
  [[ "$(find "$RUN_DIR/provenance/inter-node-ccl-gate" -name 'ccl_rank_*.json' -type f | wc -l)" -eq 2 ]] || \
    launch_die "inter-node CCL gate did not produce two rank receipts"
fi

step_index=0
run_n() {  # run_n <nodes> <condition> <step time limit> <script> <args...>
  local nodes="$1" condition="$2" limit="$3" script="$4"; shift 4
  local world=$((nodes * 8)) rc=0
  step_index=$((step_index + 1))
  export MASTER_PORT=$((32400 + (SLURM_JOB_ID % 200) * 50 + step_index))
  export CCL_ALLREDUCE="${RED_BASE_CCL_ALLREDUCE:-ring}"
  echo "$(date -u +%FT%TZ) start ${condition}" >>"$RUN_DIR/provenance/timeline.txt"
  srun --exclusive --nodes="$nodes" --ntasks="$world" --ntasks-per-node=8 --cpus-per-task=4 \
    --cpu-bind=none --time="$limit" --nodelist="$(join_hosts "$nodes")" --distribution=block:block \
    --kill-on-bad-exit=1 --wait=0 \
    --output="$RUN_DIR/logs/${condition}.%t.out" --error="$RUN_DIR/logs/${condition}.%t.err" \
    bash "$SCRIPT_DIR/pvc_rank_launch.sh" "$PVC_PYTHON" "$script" "$@" || rc=$?
  echo "$(date -u +%FT%TZ) done ${condition} rc=${rc}" >>"$RUN_DIR/provenance/timeline.txt"
  echo "${condition} rc=${rc}" >>"$RUN_DIR/provenance/conditions.txt"
  return "$rc"
}

BUCKET=16777216
common_args=(--accelerator xpu --backend ccl --output-dir "$RUN_DIR/results" --seed "$PAPER_SEED" --campaign-id "$CAMPAIGN_ID")
smoke() {  # returns 1 if the backend check fails, 2 if the 1B check fails
  run_n "$NODES" "smoke-transformer" 00:04:00 "$BENCHMARK" --profile qualification --model transformer \
    --comm-mode gpu-quant-bucketed --exchange-bucket-elems 1000003 --verify-equivalence --batch-size 4 \
    --iters 3 --warmup-iters 1 --skip-calibration \
    --run-id "redE10-ws${WORLD}-smoke-transformer" "${common_args[@]}" || return 1
  run_n "$NODES" "smoke-llama" "$1" "$BENCHMARK" --profile qualification --model llama1b \
    --comm-mode gpu-quant-bucketed --exchange-bucket-elems "$BUCKET" --batch-size 2 --iters 2 --warmup-iters 1 \
    --skip-calibration --distributed-timeout-s 600 \
    --run-id "redE10-ws${WORLD}-smoke-llama" "${common_args[@]}" || return 2
}
stop_after_smoke() {  # cancels the other E10 job by its recorded id, never by name or user
  local rc="$1" other
  echo "FATAL: smoke run failed (code $rc); no timed run was attempted" >&2
  printf 'smoke\tFAIL_%s\n' "$rc" >>"$RUN_DIR/run_metadata.tsv"
  [[ "$RED_SET" == s1 ]] && other=s2 || other=s1
  if [[ "$rc" == 1 || "$RED_SET" == s1 ]] && [[ -r "$RED_ROOT/e10_${other}_jobid" ]]; then
    local id; id="$(tr -cd '0-9' <"$RED_ROOT/e10_${other}_jobid")"
    if [[ -n "$id" ]]; then
      scancel "$id" && printf 'cancelled_sibling\t%s\n' "$id" >>"$RUN_DIR/run_metadata.tsv"
    fi
  fi
  exit 1
}
probe() {
  run_n "$NODES" "probe-default" 00:08:00 "$COLLECTIVE_PROBE" --label default \
    --output "$RUN_DIR/probes/default.json" --sizes 4,1024,1048576,16777216,67108864,268435456 \
    --max-rank-bytes 9663676416 --accelerator xpu --backend ccl || true
}
e10() {  # e10 <nodes> <mode> <measured> <warm-up> <step time limit>
  local nodes="$1" mode="$2" iters="$3" warm="$4" limit="$5" extra=()
  case "$mode" in *-bucketed) extra=(--exchange-bucket-elems "$BUCKET") ;; esac
  run_n "$nodes" "llama-ws$((nodes * 8))-${mode}" "$limit" "$BENCHMARK" \
    --profile extension --model llama1b --comm-mode "$mode" --timing-mode phase --sync-every 1 \
    --batch-size 2 --iters "$iters" --warmup-iters "$warm" --skip-calibration --distributed-timeout-s 900 \
    --progress-dir "$RUN_DIR/progress" ${extra[@]+"${extra[@]}"} \
    --run-id "redE10-ws$((nodes * 8))-llama1b-${mode}" "${common_args[@]}" || true
}
# shellcheck source=/dev/null
source "$SCRIPT_DIR/e10_sets.sh"
printf 'completed_conditions\t%s\n' "$step_index" >>"$RUN_DIR/run_metadata.tsv"
