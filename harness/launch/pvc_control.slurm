#!/usr/bin/env bash
#SBATCH --job-name=redctl-pvc
#SBATCH --account=OTH21057
#SBATCH --partition=pvc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --export=ALL

# Reducible-int8 / local-EF / instrumentation-control campaign on the Intel
# track (2026-09-18). Reuses the July stack recipe and launch plumbing; the
# benchmark is control_workload_benchmark.py (frozen July harness plus the
# control modes). Node count, walltime, and the condition set come from
# sbatch: -N, -t, and RED_SET=full|mid|repeat, RED_TAG=A|B.

set -Eeuo pipefail

SCRIPT_DIR="${RED_LAUNCH_DIR:?set RED_LAUNCH_DIR to the staged launch directory}"
[[ -r "$SCRIPT_DIR/common.sh" ]] || { printf 'FATAL: common.sh not found at %s\n' "$SCRIPT_DIR" >&2; exit 1; }
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd -P)"
source "$SCRIPT_DIR/common.sh"
require_slurm_job

RED_ROOT="${RED_ROOT:?set RED_ROOT to the staged campaign root}"
BENCHMARK="$RED_ROOT/control_workload_benchmark.py"
PROBE="$SCRIPT_DIR/pvc_ccl_exact_probe.py"
RED_SET="${RED_SET:?set RED_SET to full, mid, or repeat}"
RED_TAG="${RED_TAG:?set RED_TAG to the allocation tag, e.g. A or B}"
RED_SU_CAP="${RED_SU_CAP:-12}"
PVC_MODULE="${PVC_MODULE:-intel/23.1}"
PVC_VENV="${PVC_VENV:-/scratch/projects/compilers/intel23.1/intelpython/python3.9/envs/pytorch-gpu}"
PVC_PYTHON="$PVC_VENV/bin/python"
CAMPAIGN_ID="${CAMPAIGN_ID:-redctl-2026-09-18}"
PAPER_SEED="${PAPER_SEED:-2026010043}"

[[ "$RED_SET" == full || "$RED_SET" == mid || "$RED_SET" == repeat ]] || \
  launch_die "RED_SET must be full, mid, or repeat"
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
  printf 'pvc_module\t%s\n' "$PVC_MODULE"
  printf 'pvc_venv\t%s\n' "$PVC_VENV"
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
run_condition() {
  local condition="$1"
  shift
  step_index=$((step_index + 1))
  export MASTER_PORT=$((32400 + (SLURM_JOB_ID % 200) * 50 + step_index))
  srun --exclusive --nodes="$NODES" --ntasks="$WORLD" --ntasks-per-node=8 --cpus-per-task=4 \
    --cpu-bind=none \
    --nodelist="$(join_hosts "$NODES")" --distribution=block:block \
    --kill-on-bad-exit=1 --wait=0 \
    --output="$RUN_DIR/logs/${condition}.%t.out" --error="$RUN_DIR/logs/${condition}.%t.err" \
    bash "$SCRIPT_DIR/pvc_rank_launch.sh" "$PVC_PYTHON" "$BENCHMARK" "$@"
}

common_args=(--accelerator xpu --backend ccl --output-dir "$RUN_DIR/results" --seed "$PAPER_SEED" --campaign-id "$CAMPAIGN_ID")
rid="red${RED_TAG}-ws${WORLD}"
cal_ref="${rid}-resnet50-native"
reuse=(--skip-calibration --calibration-reference "$cal_ref")

# Cheap in-allocation smokes gate the new code paths before paper conditions.
run_condition "smoke-reducible" \
  --profile qualification --model transformer --comm-mode gpu-quant-reducible --sync-every 1 \
  --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration \
  --run-id "${rid}-smoke-reducible" "${common_args[@]}"
run_condition "smoke-localef" \
  --profile qualification --model transformer --comm-mode gpu-topk-localef --sync-every 1 \
  --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration \
  --run-id "${rid}-smoke-localef" "${common_args[@]}"
run_condition "smoke-wall" \
  --profile qualification --model transformer --comm-mode native --timing-mode wall --sync-every 1 \
  --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration \
  --run-id "${rid}-smoke-wall" "${common_args[@]}"

paper_condition() {
  local model="$1" mode="$2"
  shift 2
  run_condition "paper-${model}-${mode}" \
    --profile paper --model "$model" --comm-mode "$mode" --sync-every 1 \
    --batch-size 32 --iters 40 --warmup-iters 10 \
    --run-id "${rid}-${model}-${mode}" "$@" "${common_args[@]}"
}
wall_condition() {
  local model="$1" mode="$2"
  run_condition "wall-${model}-${mode}" \
    --profile paper --model "$model" --comm-mode "$mode" --timing-mode wall --sync-every 1 \
    --batch-size 32 --iters 40 --warmup-iters 10 \
    --run-id "${rid}-${model}-${mode}-wall" "${reuse[@]}" "${common_args[@]}"
}

# Calibration source first, at this world size.
paper_condition resnet50 native
paper_condition transformer native "${reuse[@]}"

case "$RED_SET" in
  full)
    for model in resnet50 transformer; do
      for mode in gpu-quant gpu-quant-reducible gpu-topk gpu-topk-localef powersgd ddp-overlap; do
        paper_condition "$model" "$mode" "${reuse[@]}"
      done
    done
    for model in resnet50 transformer; do
      for mode in native gpu-quant gpu-topk powersgd; do
        wall_condition "$model" "$mode"
      done
    done
    ;;
  mid)
    for model in resnet50 transformer; do
      for mode in gpu-quant gpu-quant-reducible; do
        paper_condition "$model" "$mode" "${reuse[@]}"
      done
    done
    ;;
  repeat)
    for model in resnet50 transformer; do
      for mode in gpu-quant gpu-topk powersgd ddp-overlap gpu-quant-reducible; do
        paper_condition "$model" "$mode" "${reuse[@]}"
      done
    done
    ;;
esac

printf 'completed_conditions\t%s\n' "$step_index" >>"$RUN_DIR/run_metadata.tsv"
