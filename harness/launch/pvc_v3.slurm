#!/usr/bin/env bash
#SBATCH --job-name=redctl-v3
#SBATCH --account=OTH21057
#SBATCH --partition=pvc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --export=ALL

# v3 extension campaign on the Intel track (2026-09-19): the 1B-parameter
# LLaMA-style workload and the production DDP communication hooks. Same
# stack and plumbing as pvc_control.slurm; benchmark is the v3 harness.
# RED_SET=v3smoke|v3n1|v3n2|v3n4 selects the condition set.

set -Eeuo pipefail

SCRIPT_DIR="${RED_LAUNCH_DIR:?set RED_LAUNCH_DIR to the staged launch directory}"
[[ -r "$SCRIPT_DIR/common.sh" ]] || { printf 'FATAL: common.sh not found at %s\n' "$SCRIPT_DIR" >&2; exit 1; }
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd -P)"
source "$SCRIPT_DIR/common.sh"
require_slurm_job

RED_ROOT="${RED_ROOT:?set RED_ROOT to the staged campaign root}"
BENCHMARK="$RED_ROOT/${RED_BENCHMARK:-control_workload_benchmark_v3.py}"
PROBE="$SCRIPT_DIR/pvc_ccl_exact_probe.py"
RED_SET="${RED_SET:?set RED_SET to v3smoke, v3n1, v3n2, or v3n4}"
RED_TAG="${RED_TAG:-V}"
RED_SU_CAP="${RED_SU_CAP:-12}"
PVC_MODULE="${PVC_MODULE:-intel/23.1}"
PVC_VENV="${PVC_VENV:-/scratch/projects/compilers/intel23.1/intelpython/python3.9/envs/pytorch-gpu}"
PVC_PYTHON="$PVC_VENV/bin/python"
CAMPAIGN_ID="${CAMPAIGN_ID:-redctl-v3-2026-09-19}"
PAPER_SEED="${PAPER_SEED:-2026010043}"

case "$RED_SET" in v3smoke|v3n1|v3n2|v3n4|r8|r16|r16b|r32|psgd1b) ;; *) launch_die "RED_SET must be v3smoke, v3n1, v3n2, v3n4, r8, r16, r16b, r32, or psgd1b" ;; esac
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
  # Hook conditions may pin a different oneCCL all-reduce algorithm when the
  # ring algorithm lacks the 16-bit dtype; recorded per condition.
  if [[ "$condition" == *hook* && -n "${RED_HOOK_CCL_ALLREDUCE:-}" ]]; then
    export CCL_ALLREDUCE="$RED_HOOK_CCL_ALLREDUCE"
    printf 'hook_ccl_allreduce\t%s\t%s\n' "$condition" "$CCL_ALLREDUCE" >>"$RUN_DIR/run_metadata.tsv"
  else
    export CCL_ALLREDUCE="${RED_BASE_CCL_ALLREDUCE:-ring}"
  fi
  srun --exclusive --nodes="$NODES" --ntasks="$WORLD" --ntasks-per-node=8 --cpus-per-task=4 \
    --cpu-bind=none --time="${RED_STEP_TIME:-00:20:00}" \
    --nodelist="$(join_hosts "$NODES")" --distribution=block:block \
    --kill-on-bad-exit=1 --wait=0 \
    --output="$RUN_DIR/logs/${condition}.%t.out" --error="$RUN_DIR/logs/${condition}.%t.err" \
    bash "$SCRIPT_DIR/pvc_rank_launch.sh" "$PVC_PYTHON" "$BENCHMARK" "$@"
}
run_condition_optional() {
  if ! run_condition "$@"; then
    printf 'optional_condition_failed\t%s\n' "$1" >>"$RUN_DIR/run_metadata.tsv"
  fi
}

common_args=(--accelerator xpu --backend ccl --output-dir "$RUN_DIR/results" --seed "$PAPER_SEED" --campaign-id "$CAMPAIGN_ID")
rid="red${RED_TAG}-ws${WORLD}"
small_ref="redA-ws${WORLD}-resnet50-native"
reuse=(--skip-calibration --calibration-reference "$small_ref")

# LLaMA-style workload: extension profile, local batch 2, 40 measured
# iterations, calibration skipped (the shape triad is the measurement).
llama() {
  local mode="$1" timing="${2:-phase}"
  local suffix=""
  [[ "$timing" == wall ]] && suffix="-wall"
  run_condition "llama-${mode}${suffix}" \
    --profile extension --model llama1b --comm-mode "$mode" --timing-mode "$timing" --sync-every 1 \
    --batch-size 2 --iters 40 --warmup-iters 10 --skip-calibration --distributed-timeout-s 900 \
    --run-id "${rid}-llama1b-${mode}${suffix}" "${common_args[@]}"
}
llama_optional() {
  local mode="$1"
  run_condition_optional "llama-${mode}-optional" \
    --profile extension --model llama1b --comm-mode "$mode" --sync-every 1 \
    --batch-size 2 --iters 40 --warmup-iters 10 --skip-calibration --distributed-timeout-s 900 \
    --run-id "${rid}-llama1b-${mode}" "${common_args[@]}"
}
small() {
  local model="$1" mode="$2" timing="${3:-phase}"
  local suffix=""
  [[ "$timing" == wall ]] && suffix="-wall"
  run_condition "hook-${model}-${mode}${suffix}" \
    --profile paper --model "$model" --comm-mode "$mode" --timing-mode "$timing" --sync-every 1 \
    --batch-size 32 --iters 40 --warmup-iters 10 \
    --run-id "${rid}-${model}-${mode}${suffix}" "${reuse[@]}" "${common_args[@]}"
}

case "$RED_SET" in
  v3smoke)
    run_condition "smoke-llama-native" \
      --profile qualification --model llama1b --comm-mode native --sync-every 1 \
      --batch-size 1 --iters 2 --warmup-iters 1 --skip-calibration --distributed-timeout-s 600 \
      --run-id "${rid}-smoke-llama-native" "${common_args[@]}"
    run_condition "smoke-llama-int8hook" \
      --profile qualification --model llama1b --comm-mode ddp-int8-hook --sync-every 1 \
      --batch-size 1 --iters 2 --warmup-iters 1 --skip-calibration --distributed-timeout-s 600 \
      --run-id "${rid}-smoke-llama-int8hook" "${common_args[@]}"
    run_condition "smoke-tf-fp16hook" \
      --profile qualification --model transformer --comm-mode ddp-fp16-hook --sync-every 1 \
      --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration \
      --run-id "${rid}-smoke-tf-fp16hook" "${common_args[@]}"
    run_condition "smoke-tf-int8hook-wall" \
      --profile qualification --model transformer --comm-mode ddp-int8-hook --timing-mode wall --sync-every 1 \
      --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration \
      --run-id "${rid}-smoke-tf-int8hook-wall" "${common_args[@]}"
    ;;
  v3n1)
    for mode in native gpu-quant gpu-quant-reducible ddp-int8-hook ddp-fp16-hook powersgd; do llama "$mode"; done
    for mode in native ddp-int8-hook ddp-fp16-hook; do llama "$mode" wall; done
    for model in transformer resnet50; do
      for mode in ddp-int8-hook ddp-fp16-hook; do small "$model" "$mode"; small "$model" "$mode" wall; done
    done
    ;;
  v3n2)
    for mode in native gpu-quant gpu-quant-reducible ddp-int8-hook ddp-fp16-hook; do llama "$mode"; done
    for mode in ddp-int8-hook ddp-fp16-hook; do llama "$mode" wall; done
    ;;
  v3n4)
    for model in transformer resnet50; do
      for mode in ddp-int8-hook ddp-fp16-hook; do small "$model" "$mode" wall; done
    done
    for mode in native gpu-quant-reducible ddp-int8-hook ddp-fp16-hook; do llama "$mode"; done
    for mode in ddp-int8-hook ddp-fp16-hook; do llama "$mode" wall; done
    llama_optional gpu-quant
    ;;
  # Reruns after the first chain: the 1B PowerSGD condition aborts in oneCCL
  # (zero-length all-reduce), the 1B whole-gradient gather exhausts device
  # memory at 16 ranks and above, and the 1B int8 hook page-faults at 32
  # ranks; those three are excluded here and reported as such.
  r8)
    for mode in native ddp-int8-hook ddp-fp16-hook; do llama "$mode" wall; done
    for model in transformer resnet50; do
      for mode in ddp-int8-hook ddp-fp16-hook; do small "$model" "$mode"; small "$model" "$mode" wall; done
    done
    ;;
  r16)
    for mode in gpu-quant-reducible ddp-int8-hook ddp-fp16-hook; do llama "$mode"; done
    for mode in ddp-int8-hook ddp-fp16-hook; do llama "$mode" wall; done
    ;;
  r16b)
    llama ddp-fp16-hook
    llama ddp-fp16-hook wall
    ;;
  psgd1b)
    # rerun of the one 1B condition that aborted in oneCCL (zero-length
    # all-reduce inside the official PowerSGD hook); v4 harness with the
    # empty-all-reduce guard, phase and wall.
    llama powersgd
    llama powersgd wall
    ;;
  r32)
    llama ddp-fp16-hook
    llama ddp-fp16-hook wall
    llama native wall
    for model in transformer resnet50; do
      for mode in ddp-int8-hook ddp-fp16-hook; do small "$model" "$mode"; done
    done
    ;;
esac

printf 'completed_conditions\t%s\n' "$step_index" >>"$RUN_DIR/run_metadata.tsv"
