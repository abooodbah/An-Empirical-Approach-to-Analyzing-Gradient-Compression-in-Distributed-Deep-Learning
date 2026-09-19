#!/usr/bin/env bash
#SBATCH --job-name=redctl-h100
#SBATCH --account=OTH21057
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --signal=B:TERM@60
#SBATCH --export=ALL

# H100 2x4 control campaign (2026-09-18): wall-clock instrumentation control
# (phase vs uninstrumented) plus the shared-scale reducible int8 mode, over
# the same host-staged NDR path as the July campaign. Container, venv, RDMA
# sidecar, and per-condition transport validation are reused verbatim from
# the frozen July release; only the benchmark and the condition list differ.

set -Eeuo pipefail

SCRIPT_DIR="${RED_LAUNCH_DIR:?set RED_LAUNCH_DIR to the staged launch directory}"
[[ -r "$SCRIPT_DIR/common.sh" ]] || { printf 'FATAL: common.sh not found at %s\n' "$SCRIPT_DIR" >&2; exit 1; }
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd -P)"
source "$SCRIPT_DIR/common.sh"
require_slurm_job

RED_ROOT="${RED_ROOT:?set RED_ROOT to the staged campaign root}"
RELEASE_ROOT="${RED_H100_RELEASE_ROOT:?set RED_H100_RELEASE_ROOT to the frozen July release root}"
BENCHMARK="$RED_ROOT/control_workload_benchmark.py"
PAPER_RDMA_ROOT="$RELEASE_ROOT/experiments/tacc_paper_workloads/rdma_runtime"
TRANSPORT_VALIDATOR="$RELEASE_ROOT/experiments/tacc_paper_workloads/h100_nccl_transport.py"
NODE_TORCHRUN="$RELEASE_ROOT/experiments/tacc_paper_workloads/launch/h100_node_torchrun.sh"
CONTAINER_IMAGE="${H100_CONTAINER_IMAGE:-${WORK:?WORK is not set}/acm_pivot_v9/containers/pytorch_2.4.1-cuda12.1-cudnn9-runtime.sif}"
H100_VENV="${H100_VENV:-${SCRATCH:?SCRATCH is not set}/acm_pivot_v9/real_workloads_runtime/venv-pytorch241}"
CAMPAIGN_ID="${CAMPAIGN_ID:-redctl-2026-09-18}"
PAPER_SEED="${PAPER_SEED:-2026010043}"
RED_SU_CAP="${RED_SU_CAP:-6}"

for required in "$BENCHMARK" "$TRANSPORT_VALIDATOR" "$NODE_TORCHRUN" \
  "$PAPER_RDMA_ROOT/tacc_rdma_bridge.py" "$PAPER_RDMA_ROOT/rdma_container_preflight.py" \
  "$PAPER_RDMA_ROOT/configs/tacc_h100_rdma_layout.json" "$PAPER_RDMA_ROOT/launch/rdma_sidecar.sh"; do
  [[ -s "$required" ]] || launch_die "missing input: $required"
done
[[ -r "$CONTAINER_IMAGE" ]] || launch_die "missing H100 container: $CONTAINER_IMAGE"
[[ -L "$H100_VENV/bin/python" || -x "$H100_VENV/bin/python" ]] || \
  launch_die "missing H100 venv python: $H100_VENV/bin/python"

init_run_dir redctl-h100
install_exit_receipt_trap
H100_RUN_COMPLETE=0
h100_exit_receipt() {
  local rc="$1"
  if [[ "$rc" -eq 0 && "$H100_RUN_COMPLETE" -ne 1 ]]; then
    rc=125
  fi
  launch_exit_receipt "$rc"
}
trap 'h100_exit_receipt "$?"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
printf 'RUN_DIR=%s\n' "$RUN_DIR"
collect_base_provenance "$RED_ROOT"
record_and_enforce_cost 2 4 "$RED_SU_CAP"
hash_launch_inputs "$SCRIPT_DIR/common.sh" "${BASH_SOURCE[0]}" "$BENCHMARK" "$NODE_TORCHRUN" "$TRANSPORT_VALIDATOR"
sha256sum "$CONTAINER_IMAGE" >"$RUN_DIR/provenance/container.sha256"

{
  printf 'campaign_id\t%s\n' "$CAMPAIGN_ID"
  printf 'paper_seed\t%s\n' "$PAPER_SEED"
  printf 'world_size\t8\n'
  printf 'nodes\t2\n'
  printf 'ranks_per_node\t4\n'
  printf 'benchmark_sha256\t%s\n' "$(sha256sum "$BENCHMARK" | awk '{print $1}')"
  printf 'transport_path\tnccl_ib_host_staged\n'
  printf 'expected_gdr_level\tLOC\n'
} >>"$RUN_DIR/run_metadata.tsv"

export PROJECT_ROOT="$RED_ROOT" RUN_DIR RUN_ROOT CONTAINER_IMAGE H100_VENV PAPER_RDMA_ROOT
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA="=mlx5_0:1"
export NCCL_SOCKET_IFNAME="=ibp92s0"
export NCCL_NET_GDR_LEVEL=LOC
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV

mapfile -t hosts <"$RUN_DIR/provenance/hosts.txt"
[[ "${#hosts[@]}" -eq 2 ]] || launch_die "expected exactly two allocated hosts"
export MASTER_ADDR="${hosts[0]}"

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 --wait=0 \
  --output="$RUN_DIR/provenance/h100-node.%t.out" \
  bash -lc 'hostname; nvidia-smi -L; command -v ibv_devinfo >/dev/null && ibv_devinfo -l || true'

runtime="$(command -v apptainer || command -v singularity || true)"
if [[ -z "$runtime" ]] && type module >/dev/null 2>&1; then
  set +u
  module load tacc-apptainer >"$RUN_DIR/provenance/apptainer_module.out" 2>&1 || \
    module load apptainer >"$RUN_DIR/provenance/apptainer_module.out" 2>&1 || true
  set -u
  runtime="$(command -v apptainer || command -v singularity || true)"
fi
[[ -n "$runtime" ]] || launch_die "apptainer/singularity is unavailable"
export CONTAINER_RUNTIME="$runtime"

step_index=0
run_condition() {
  local condition="${1}-seed${PAPER_SEED}"
  shift
  step_index=$((step_index + 1))
  export CONDITION_ID="$condition"
  export RDMA_RECEIPT_DIR="$RUN_DIR/provenance/rdma/$condition"
  mkdir -p "$RDMA_RECEIPT_DIR"
  export MASTER_PORT=$((29400 + (SLURM_JOB_ID % 500) * 20 + step_index))
  srun --exclusive --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=16 \
    --kill-on-bad-exit=1 --wait=0 \
    --output="$RUN_DIR/logs/${condition}.controller.%t.out" \
    --error="$RUN_DIR/logs/${condition}.controller.%t.err" \
    bash "$NODE_TORCHRUN" "$BENCHMARK" "$@"

  local evidence="$RUN_DIR/provenance/${condition}.nccl_transport.json"
  local -a nccl_logs=()
  shopt -s nullglob
  nccl_logs=("$RUN_DIR"/logs/nccl."$condition".*.log)
  shopt -u nullglob
  python3 "$TRANSPORT_VALIDATOR" \
    --condition-id "$condition" \
    --expected-hosts "$RUN_DIR/provenance/hosts.txt" \
    --output "$evidence" \
    "${nccl_logs[@]}" || \
    launch_die "condition $condition did not prove two-host host-staged NCCL NET/IB"
}

common_args=(--accelerator cuda --backend nccl --output-dir "$RUN_DIR/results" --seed "$PAPER_SEED" --campaign-id "$CAMPAIGN_ID")
rid="redH-ws8"
cal_ref="${rid}-resnet50-native"
reuse=(--skip-calibration --calibration-reference "$cal_ref")

run_condition "smoke-reducible" \
  --profile qualification --model transformer --comm-mode gpu-quant-reducible --sync-every 1 \
  --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration --distributed-timeout-s 90 \
  --run-id "${rid}-smoke-reducible" "${common_args[@]}"
run_condition "smoke-wall" \
  --profile qualification --model transformer --comm-mode native --timing-mode wall --sync-every 1 \
  --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration --distributed-timeout-s 90 \
  --run-id "${rid}-smoke-wall" "${common_args[@]}"

run_condition "paper-resnet50-native" \
  --profile paper --model resnet50 --comm-mode native --sync-every 1 \
  --batch-size 32 --iters 40 --warmup-iters 10 \
  --run-id "${cal_ref}" "${common_args[@]}"
run_condition "paper-transformer-native" \
  --profile paper --model transformer --comm-mode native --sync-every 1 \
  --batch-size 32 --iters 40 --warmup-iters 10 \
  --run-id "${rid}-transformer-native" "${reuse[@]}" "${common_args[@]}"

for model in resnet50 transformer; do
  for mode in gpu-quant gpu-topk powersgd gpu-quant-reducible; do
    run_condition "paper-${model}-${mode}" \
      --profile paper --model "$model" --comm-mode "$mode" --sync-every 1 \
      --batch-size 32 --iters 40 --warmup-iters 10 \
      --run-id "${rid}-${model}-${mode}" "${reuse[@]}" "${common_args[@]}"
  done
done
for model in resnet50 transformer; do
  for mode in native gpu-quant gpu-topk powersgd; do
    run_condition "wall-${model}-${mode}" \
      --profile paper --model "$model" --comm-mode "$mode" --timing-mode wall --sync-every 1 \
      --batch-size 32 --iters 40 --warmup-iters 10 \
      --run-id "${rid}-${model}-${mode}-wall" "${reuse[@]}" "${common_args[@]}"
  done
done

printf 'completed_conditions\t%s\n' "$step_index" >>"$RUN_DIR/run_metadata.tsv"
H100_RUN_COMPLETE=1
