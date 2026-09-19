#!/usr/bin/env bash
#SBATCH --job-name=paper-pvc-1to4
#SBATCH --account=OTH21057
#SBATCH --partition=pvc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --export=ALL

# One four-node allocation, nested 1->2->3->4 scale steps. Default scope is a
# non-paper smoke sequence. PVC_CAMPAIGN_SCOPE=paper enables the frozen matrix.

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
PROBE="$SCRIPT_DIR/pvc_ccl_exact_probe.py"
PVC_MODULE="${PVC_MODULE:-intel/23.1}"
PVC_STACK_SETUP="${PVC_STACK_SETUP:-}"
PVC_VENV="${PVC_VENV:-/scratch/projects/compilers/intel23.1/intelpython/python3.9/envs/pytorch-gpu}"
PVC_PYTHON="${PVC_PYTHON:-$PVC_VENV/bin/python}"
PVC_CAMPAIGN_SCOPE="${PVC_CAMPAIGN_SCOPE:-smoke}"
CAMPAIGN_ID="${CAMPAIGN_ID:-tpds-2026-01-0043-tacc-20260717}"
PAPER_SEED="${PAPER_SEED:-2026010043}"
: "${PVC_QUALIFICATION_RECEIPT:?set PVC_QUALIFICATION_RECEIPT to an absolute PVC_STACK_QUALIFIED.tsv path}"

[[ "$PVC_CAMPAIGN_SCOPE" == smoke || "$PVC_CAMPAIGN_SCOPE" == paper ]] || \
  launch_die "PVC_CAMPAIGN_SCOPE must be smoke or paper"
[[ "$PAPER_SEED" =~ ^[0-9]+$ ]] || launch_die "PAPER_SEED must be an unsigned integer"
(( PAPER_SEED <= 4294967295 )) || launch_die "PAPER_SEED must fit in an unsigned 32-bit integer"
type module >/dev/null 2>&1 || launch_die "TACC module command is unavailable"
[[ -z "$PVC_STACK_SETUP" || -r "$PVC_STACK_SETUP" ]] || launch_die "PVC_STACK_SETUP is unreadable: $PVC_STACK_SETUP"
[[ -x "$PVC_PYTHON" ]] || launch_die "PVC Python is not executable: $PVC_PYTHON"
[[ -f "$BENCHMARK" ]] || launch_die "missing benchmark: $BENCHMARK"
[[ -f "$PROBE" ]] || launch_die "missing probe: $PROBE"

benchmark_sha="$(sha256sum "$BENCHMARK" | awk '{print $1}')"
validate_receipt "$PVC_QUALIFICATION_RECEIPT" "$benchmark_sha"
[[ "$(receipt_value "$PVC_QUALIFICATION_RECEIPT" probe_sha256)" == "$(sha256sum "$PROBE" | awk '{print $1}')" ]] || \
  launch_die "probe hash differs from the qualified probe"

init_run_dir pvc-1to4
install_exit_receipt_trap
printf 'RUN_DIR=%s\n' "$RUN_DIR"
collect_base_provenance "$PROJECT_ROOT"
if [[ "$PVC_CAMPAIGN_SCOPE" == smoke ]]; then
  record_and_enforce_cost 4 3 3
else
  record_and_enforce_cost 4 3 24
fi
hash_launch_inputs "$SCRIPT_DIR/common.sh" "$SCRIPT_DIR/pvc_rank_launch.sh" "$PROBE" "${BASH_SOURCE[0]}" "$BENCHMARK" "$PVC_QUALIFICATION_RECEIPT"

set +u
module purge >"$RUN_DIR/provenance/intel_module.out" 2>&1
module load "$PVC_MODULE" >>"$RUN_DIR/provenance/intel_module.out" 2>&1
if [[ -n "$PVC_STACK_SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$PVC_STACK_SETUP" >"$RUN_DIR/provenance/custom_stack_setup.out" 2>"$RUN_DIR/provenance/custom_stack_setup.err"
fi
PVC_SITE_PACKAGES="$PVC_VENV/lib/python3.9/site-packages"
export LD_LIBRARY_PATH="$PVC_SITE_PACKAGES/torch/lib:$PVC_SITE_PACKAGES/intel_extension_for_pytorch/lib:$PVC_VENV/lib:${LD_LIBRARY_PATH:-}"
set -u
command -v "$PVC_PYTHON" >/dev/null 2>&1 || launch_die "PVC Python is unavailable: $PVC_PYTHON"

{
  printf 'campaign_scope\t%s\n' "$PVC_CAMPAIGN_SCOPE"
  printf 'campaign_id\t%s\n' "$CAMPAIGN_ID"
  printf 'paper_seed\t%s\n' "$PAPER_SEED"
  printf 'pvc_module\t%s\n' "$PVC_MODULE"
  printf 'pvc_venv\t%s\n' "$PVC_VENV"
  printf 'scale_sequence_nodes\t1,2,3,4\n'
  printf 'ranks_per_node\t8\n'
  printf 'world_size_sequence\t8,16,24,32\n'
  printf 'scaling_semantics\tweak scaling: fixed local batch; global work grows with world size\n'
  printf 'script_default_walltime_minutes\t15\n'
  printf 'script_default_charge_upper_bound_su\t3.0\n'
  printf 'paper_walltime_upper_bound_minutes\t120\n'
  printf 'paper_charge_upper_bound_su\t24.0\n'
  printf 'qualification_receipt\t%s\n' "$PVC_QUALIFICATION_RECEIPT"
  printf 'intel_collective_path\toneccl_host_staged_ofi_tcp\n'
  printf 'device_direct_collectives\tfalse\n'
  printf 'workload_cpu_bind\tnone (oneCCL workers use recorded CCL_WORKER_AFFINITY)\n'
  printf 'worker_affinity_strategy\tallocation-derived unique NUMA-0 CPUs\n'
} >>"$RUN_DIR/run_metadata.tsv"

export CCL_LOG_LEVEL="${CCL_LOG_LEVEL:-info}"
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-ofi}"
# Force documented host-staged oneCCL algorithms; the Intel23 device-direct
# kernels.spv module is incompatible with the installed PVC runtime.
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
[[ "${#hosts[@]}" -eq 4 ]] || launch_die "expected exactly four allocated PVC nodes"
export MASTER_ADDR="${hosts[0]}"

# Revalidate the active stack, not merely the receipt file. This prevents a
# receipt from one ABI being reused with a different module or venv.
[[ "$(receipt_value "$PVC_QUALIFICATION_RECEIPT" pvc_module)" == "$PVC_MODULE" ]] || \
  launch_die "PVC module differs from the qualified module"
[[ "$(receipt_value "$PVC_QUALIFICATION_RECEIPT" pvc_venv)" == "$PVC_VENV" ]] || \
  launch_die "PVC venv differs from the qualified venv"
"$PVC_PYTHON" "$PROBE" --mode inspect --output-dir "$RUN_DIR/provenance/active-stack" \
  >"$RUN_DIR/logs/active-stack.out" 2>"$RUN_DIR/logs/active-stack.err"

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
{
  printf 'ccl_allreduce\t%s\n' "$CCL_ALLREDUCE"
  printf 'ccl_allgatherv\t%s\n' "$CCL_ALLGATHERV"
  printf 'ccl_barrier\t%s\n' "$CCL_BARRIER"
  printf 'ccl_bcast\t%s\n' "$CCL_BCAST"
  printf 'ccl_reduce_scatter\t%s\n' "$CCL_REDUCE_SCATTER"
  printf 'fi_provider\t%s\n' "$FI_PROVIDER"
  printf 'fi_tcp_iface\t%s\n' "$FI_TCP_IFACE"
} >>"$RUN_DIR/run_metadata.tsv"

join_hosts() {
  local count="$1"
  local joined="${hosts[0]}"
  local index
  for ((index = 1; index < count; index++)); do
    joined+=",${hosts[index]}"
  done
  printf '%s\n' "$joined"
}

# A successful inter-node XPU CCL collective is mandatory before any scale data.
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

step_index=0
run_scale_condition() {
  local nodes="$1"
  local condition="$2"
  shift 2
  local tasks=$((nodes * 8))
  step_index=$((step_index + 1))
  export MASTER_PORT=$((32400 + (SLURM_JOB_ID % 200) * 50 + step_index))
  srun --exclusive --nodes="$nodes" --ntasks="$tasks" --ntasks-per-node=8 --cpus-per-task=4 \
    --cpu-bind=none \
    --nodelist="$(join_hosts "$nodes")" --distribution=block:block \
    --kill-on-bad-exit=1 --wait=0 \
    --output="$RUN_DIR/logs/${condition}.%t.out" --error="$RUN_DIR/logs/${condition}.%t.err" \
    bash "$SCRIPT_DIR/pvc_rank_launch.sh" "$PVC_PYTHON" "$BENCHMARK" "$@"
}

common_args=(--accelerator xpu --backend ccl --output-dir "$RUN_DIR/results" --seed "$PAPER_SEED" --campaign-id "$CAMPAIGN_ID")
for nodes in 1 2 3 4; do
  world_size=$((nodes * 8))
  if [[ "$PVC_CAMPAIGN_SCOPE" == smoke ]]; then
    run_scale_condition "$nodes" "smoke-transformer-native-ws${world_size}-seed${PAPER_SEED}" \
      --profile qualification --model transformer --comm-mode native --sync-every 1 \
      --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration \
      --run-id "pvc-smoke-transformer-native-ws${world_size}-seed${PAPER_SEED}" "${common_args[@]}"
    continue
  fi

  calibration_reference="pvc-paper-resnet50-native-s1-ws${world_size}-seed${PAPER_SEED}"
  reuse_calibration_args=(
    --skip-calibration
    --calibration-reference "$calibration_reference"
  )

  for model in resnet50 transformer; do
    for sync_every in 1 2 4; do
      native_calibration_args=("${reuse_calibration_args[@]}")
      if [[ "$model" == resnet50 && "$sync_every" -eq 1 ]]; then
        # This first condition is the alpha/beta calibration source for every
        # other exact paper condition at the same world size.
        native_calibration_args=()
      fi
      run_scale_condition "$nodes" "paper-${model}-native-s${sync_every}-ws${world_size}-seed${PAPER_SEED}" \
        --profile paper --model "$model" --comm-mode native --sync-every "$sync_every" \
        --batch-size 32 --iters 40 --warmup-iters 10 \
        --run-id "pvc-paper-${model}-native-s${sync_every}-ws${world_size}-seed${PAPER_SEED}" \
        "${native_calibration_args[@]}" "${common_args[@]}"
      run_scale_condition "$nodes" "paper-${model}-topk-s${sync_every}-ws${world_size}-seed${PAPER_SEED}" \
        --profile paper --model "$model" --comm-mode gpu-topk --sync-every "$sync_every" \
        --batch-size 32 --iters 40 --warmup-iters 10 \
        --run-id "pvc-paper-${model}-topk-s${sync_every}-ws${world_size}-seed${PAPER_SEED}" \
        "${reuse_calibration_args[@]}" "${common_args[@]}"
    done
    for mode in gpu-quant cpu-quant powersgd; do
      run_scale_condition "$nodes" "paper-${model}-${mode}-ws${world_size}-seed${PAPER_SEED}" \
        --profile paper --model "$model" --comm-mode "$mode" --sync-every 1 \
        --batch-size 32 --iters 40 --warmup-iters 10 \
        --run-id "pvc-paper-${model}-${mode}-ws${world_size}-seed${PAPER_SEED}" \
        "${reuse_calibration_args[@]}" "${common_args[@]}"
    done
    run_scale_condition "$nodes" "validation-${model}-ddp-overlap-ws${world_size}-seed${PAPER_SEED}" \
      --profile paper --model "$model" --comm-mode ddp-overlap --sync-every 1 \
      --batch-size 32 --iters 40 --warmup-iters 10 \
      --run-id "pvc-validation-${model}-ddp-overlap-ws${world_size}-seed${PAPER_SEED}" \
      "${reuse_calibration_args[@]}" "${common_args[@]}"
  done
done

printf 'completed_conditions\t%s\n' "$step_index" >>"$RUN_DIR/run_metadata.tsv"
