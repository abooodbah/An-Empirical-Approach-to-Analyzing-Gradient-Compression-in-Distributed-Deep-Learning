#!/usr/bin/env bash
#SBATCH --job-name=paper-h100-2x4
#SBATCH --account=OTH21057
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --signal=B:TERM@60
#SBATCH --export=ALL

# Two-node H100 continuity run over host-staged NDR InfiniBand. Default is a
# cheap qualification; CAMPAIGN_PHASE=full replays the paper matrix and adds a
# separately labelled overlap-enabled DDP control.

set -Eeuo pipefail

SCRIPT_DIR="${PAPER_LAUNCH_DIR:-${SLURM_SUBMIT_DIR:-}/experiments/tacc_paper_workloads/launch}"
[[ -r "$SCRIPT_DIR/common.sh" ]] || {
  printf 'FATAL: set PAPER_LAUNCH_DIR to the staged launch directory; common.sh not found at %s\n' "$SCRIPT_DIR" >&2
  exit 1
}
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd -P)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
require_slurm_job

PROJECT_ROOT="$(project_root_from_launch_dir "$SCRIPT_DIR")"
BENCHMARK="${PAPER_BENCHMARK:-$PROJECT_ROOT/experiments/tacc_paper_workloads/paper_workload_benchmark.py}"
PAPER_RDMA_ROOT="$PROJECT_ROOT/experiments/tacc_paper_workloads/rdma_runtime"
TRANSPORT_VALIDATOR="$PROJECT_ROOT/experiments/tacc_paper_workloads/h100_nccl_transport.py"
LAUNCH_CONTRACT_TOOL="$PROJECT_ROOT/experiments/tacc_paper_workloads/h100_launch_contract.py"
CONTAINER_IMAGE="${H100_CONTAINER_IMAGE:-${WORK:?WORK is not set}/acm_pivot_v9/containers/pytorch_2.4.1-cuda12.1-cudnn9-runtime.sif}"
H100_VENV="${H100_VENV:-${SCRATCH:?SCRATCH is not set}/acm_pivot_v9/real_workloads_runtime/venv-pytorch241}"
CAMPAIGN_PHASE="${CAMPAIGN_PHASE:-qualification}"
CAMPAIGN_ID="${CAMPAIGN_ID:-tpds-2026-01-0043-tacc-20260717}"
PAPER_SEED="${PAPER_SEED:-2026010043}"

[[ "$CAMPAIGN_PHASE" == qualification || "$CAMPAIGN_PHASE" == full ]] || \
  launch_die "CAMPAIGN_PHASE must be qualification or full"
[[ "$PAPER_SEED" =~ ^[0-9]+$ ]] || launch_die "PAPER_SEED must be an unsigned integer"
(( PAPER_SEED <= 4294967295 )) || launch_die "PAPER_SEED must fit in an unsigned 32-bit integer"
[[ -f "$BENCHMARK" ]] || launch_die "missing benchmark: $BENCHMARK"
[[ -r "$CONTAINER_IMAGE" ]] || launch_die "missing H100 container: $CONTAINER_IMAGE"
[[ -L "$H100_VENV/bin/python" || -x "$H100_VENV/bin/python" ]] || \
  launch_die "missing H100 venv Python link/executable: $H100_VENV/bin/python"

contract_inputs=(
  "$SCRIPT_DIR/common.sh"
  "$SCRIPT_DIR/h100_2node.slurm"
  "$SCRIPT_DIR/h100_node_torchrun.sh"
  "$BENCHMARK"
  "$TRANSPORT_VALIDATOR"
  "$LAUNCH_CONTRACT_TOOL"
  "$PAPER_RDMA_ROOT/tacc_rdma_bridge.py"
  "$PAPER_RDMA_ROOT/rdma_container_preflight.py"
  "$PAPER_RDMA_ROOT/configs/tacc_h100_rdma_layout.json"
  "$PAPER_RDMA_ROOT/launch/rdma_sidecar.sh"
)
for required in "${contract_inputs[@]}"; do
  [[ -s "$required" ]] || launch_die "missing H100 launch-contract input: $required"
done

benchmark_sha="$(sha256sum "$BENCHMARK" | awk '{print $1}')"
launch_contract_sha="$(
  python3 "$LAUNCH_CONTRACT_TOOL" digest --root "$PROJECT_ROOT" "${contract_inputs[@]}"
)" || launch_die "could not compute the H100 launch-contract digest"
[[ "$launch_contract_sha" =~ ^[0-9a-f]{64}$ ]] || \
  launch_die "H100 launch-contract digest is malformed"
if [[ "$CAMPAIGN_PHASE" == full ]]; then
  : "${H100_QUALIFICATION_RECEIPT:?full phase requires H100_QUALIFICATION_RECEIPT=/absolute/path/H100_QUALIFIED.tsv}"
  python3 "$LAUNCH_CONTRACT_TOOL" validate-receipt "$H100_QUALIFICATION_RECEIPT" \
    --benchmark-sha256 "$benchmark_sha" \
    --launch-contract-sha256 "$launch_contract_sha" || \
    launch_die "H100 qualification receipt does not authorize this host-staged NDR/IB launch contract"
fi

init_run_dir h100
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
# Slurm can terminate the batch shell while an srun step is being reaped. Make
# the exit receipt fail closed instead of inheriting a stale zero status. The
# explicit completion flag also prevents a zero-returning interrupted srun
# from ever producing PASS.
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
printf 'RUN_DIR=%s\n' "$RUN_DIR"
collect_base_provenance "$PROJECT_ROOT"
record_and_enforce_cost 2 4 2
hash_inputs=("${contract_inputs[@]}")
if [[ "$CAMPAIGN_PHASE" == full ]]; then
  hash_inputs+=("$H100_QUALIFICATION_RECEIPT")
fi
hash_launch_inputs "${hash_inputs[@]}"
python3 "$LAUNCH_CONTRACT_TOOL" manifest --root "$PROJECT_ROOT" \
  --output "$RUN_DIR/provenance/h100_launch_contract.json" \
  "${contract_inputs[@]}" >"$RUN_DIR/provenance/h100_launch_contract.sha256"
sha256sum "$CONTAINER_IMAGE" >"$RUN_DIR/provenance/container.sha256"

{
  printf 'campaign_phase\t%s\n' "$CAMPAIGN_PHASE"
  printf 'campaign_id\t%s\n' "$CAMPAIGN_ID"
  printf 'paper_seed\t%s\n' "$PAPER_SEED"
  printf 'world_size\t8\n'
  printf 'nodes\t2\n'
  printf 'ranks_per_node\t4\n'
  printf 'local_batch_size\t32\n'
  printf 'script_default_walltime_minutes\t15\n'
  printf 'script_default_charge_upper_bound_su\t2.0\n'
  printf 'launch_contract_sha256\t%s\n' "$launch_contract_sha"
  printf 'transport_path\tnccl_ib_host_staged\n'
  printf 'expected_hca\tmlx5_0\n'
  printf 'expected_port\t1\n'
  printf 'expected_nccl_ib_hca\t=mlx5_0:1\n'
  printf 'expected_socket_ifname\t=ibp92s0\n'
  printf 'expected_gdr_level\tLOC\n'
  printf 'gdr_policy\tchannel-level GDRDMA is forbidden; host-staged NET/IB is required\n'
  printf 'comparison_note\tTACC 2x4 world-size-8 continuity run; hardware/software differ from the paper 1x8 system\n'
} >>"$RUN_DIR/run_metadata.tsv"

export PROJECT_ROOT RUN_DIR RUN_ROOT CONTAINER_IMAGE H100_VENV PAPER_RDMA_ROOT
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
  bash -lc 'hostname; nvidia-smi -L; nvidia-smi topo -m; nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id --format=csv,noheader; command -v ibv_devinfo >/dev/null && ibv_devinfo || true; command -v ip >/dev/null && ip -br link || true'

runtime="$(command -v apptainer || command -v singularity || true)"
if [[ -z "$runtime" ]] && type module >/dev/null 2>&1; then
  # TACC's Apptainer module sources Bash-completion code that references
  # optional variables. Vendor module initialization must run with nounset
  # relaxed, then the experiment's fail-closed setting is restored.
  set +u
  module load tacc-apptainer >"$RUN_DIR/provenance/apptainer_module.out" 2>&1 || \
    module load apptainer >"$RUN_DIR/provenance/apptainer_module.out" 2>&1 || true
  set -u
  runtime="$(command -v apptainer || command -v singularity || true)"
fi
[[ -n "$runtime" ]] || launch_die "apptainer/singularity is unavailable"
export CONTAINER_RUNTIME="$runtime"
runtime_resolved="$(readlink -f "$runtime")"
[[ -x "$runtime_resolved" ]] || launch_die "resolved container runtime is not executable: $runtime_resolved"
printf 'container_runtime\t%s\n' "$runtime_resolved" >>"$RUN_DIR/run_metadata.tsv"
sha256sum "$runtime_resolved" >"$RUN_DIR/provenance/container_runtime.sha256"
"$runtime" exec --nv --bind "$PROJECT_ROOT:$PROJECT_ROOT,$RUN_ROOT:$RUN_ROOT,${WORK}:${WORK},${SCRATCH}:${SCRATCH}" \
  "$CONTAINER_IMAGE" "$H100_VENV/bin/python" -c \
  'import json, torch; print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(), "cuda_devices": torch.cuda.device_count()}, sort_keys=True))' \
  >"$RUN_DIR/provenance/container_torch.json"

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
    bash "$SCRIPT_DIR/h100_node_torchrun.sh" "$BENCHMARK" "$@"

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
    launch_die "condition $condition did not prove two-host host-staged NCCL NET/IB without Socket/GDRDMA"
}

common_args=(--accelerator cuda --backend nccl --output-dir "$RUN_DIR/results" --seed "$PAPER_SEED" --campaign-id "$CAMPAIGN_ID")
if [[ "$CAMPAIGN_PHASE" == qualification ]]; then
  run_condition q-resnet-native \
    --profile qualification --model resnet50 --comm-mode native --sync-every 1 \
    --batch-size 4 --iters 3 --warmup-iters 1 --distributed-timeout-s 90 \
    --run-id "h100-q-resnet-native-seed${PAPER_SEED}" "${common_args[@]}"
  run_condition q-transformer-overlap \
    --profile qualification --model transformer --comm-mode ddp-overlap --sync-every 1 \
    --batch-size 4 --iters 3 --warmup-iters 1 --distributed-timeout-s 90 \
    --run-id "h100-q-transformer-overlap-seed${PAPER_SEED}" "${common_args[@]}"
  for mode in gpu-topk gpu-quant cpu-quant powersgd; do
    run_condition "q-transformer-${mode}" \
      --profile qualification --model transformer --comm-mode "$mode" --sync-every 1 \
      --batch-size 4 --iters 2 --warmup-iters 1 --skip-calibration --distributed-timeout-s 90 \
      --run-id "h100-q-transformer-${mode}-seed${PAPER_SEED}" "${common_args[@]}"
  done
  {
    printf 'status\tPASS\n'
    printf 'platform\th100-2x4\n'
    printf 'job_id\t%s\n' "$SLURM_JOB_ID"
    printf 'benchmark_sha256\t%s\n' "$benchmark_sha"
    printf 'launch_contract_sha256\t%s\n' "$launch_contract_sha"
    printf 'transport_path\tnccl_ib_host_staged\n'
    printf 'expected_hca\tmlx5_0\n'
    printf 'expected_port\t1\n'
    printf 'expected_nccl_ib_hca\t=mlx5_0:1\n'
    printf 'expected_socket_ifname\t=ibp92s0\n'
    printf 'expected_gdr_level\tLOC\n'
    printf 'receipt_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$RUN_DIR/H100_QUALIFIED.tsv"
else
  for model in resnet50 transformer; do
    for sync_every in 1 2 4; do
      run_condition "paper-${model}-native-s${sync_every}" \
        --profile paper --model "$model" --comm-mode native --sync-every "$sync_every" \
        --batch-size 32 --iters 40 --warmup-iters 10 \
        --run-id "h100-paper-${model}-native-s${sync_every}-seed${PAPER_SEED}" "${common_args[@]}"
      run_condition "paper-${model}-topk-s${sync_every}" \
        --profile paper --model "$model" --comm-mode gpu-topk --sync-every "$sync_every" \
        --batch-size 32 --iters 40 --warmup-iters 10 \
        --run-id "h100-paper-${model}-topk-s${sync_every}-seed${PAPER_SEED}" "${common_args[@]}"
    done
    for mode in gpu-quant cpu-quant powersgd; do
      run_condition "paper-${model}-${mode}" \
        --profile paper --model "$model" --comm-mode "$mode" --sync-every 1 \
        --batch-size 32 --iters 40 --warmup-iters 10 \
        --run-id "h100-paper-${model}-${mode}-seed${PAPER_SEED}" "${common_args[@]}"
    done
    run_condition "validation-${model}-ddp-overlap" \
      --profile paper --model "$model" --comm-mode ddp-overlap --sync-every 1 \
      --batch-size 32 --iters 40 --warmup-iters 10 \
      --run-id "h100-validation-${model}-ddp-overlap-seed${PAPER_SEED}" "${common_args[@]}"
  done
fi

printf 'completed_conditions\t%s\n' "$step_index" >>"$RUN_DIR/run_metadata.tsv"
H100_RUN_COMPLETE=1
