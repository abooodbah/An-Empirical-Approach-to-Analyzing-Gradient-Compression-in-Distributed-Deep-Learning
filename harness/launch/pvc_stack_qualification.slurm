#!/usr/bin/env bash
#SBATCH --job-name=paper-pvc-qual
#SBATCH --account=OTH21057
#SBATCH --partition=pvc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --export=ALL

# One-node fail-closed qualification. It writes PVC_STACK_QUALIFIED.tsv only after
# the stack, all eight tiles, exact CCL all-reduce, and an eight-rank workload pass.

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
CAMPAIGN_ID="${CAMPAIGN_ID:-tpds-2026-01-0043-tacc-20260717}"

type module >/dev/null 2>&1 || launch_die "TACC module command is unavailable"
[[ -z "$PVC_STACK_SETUP" || -r "$PVC_STACK_SETUP" ]] || launch_die "PVC_STACK_SETUP is unreadable: $PVC_STACK_SETUP"
[[ -x "$PVC_PYTHON" ]] || launch_die "PVC Python is not executable: $PVC_PYTHON"
[[ -f "$BENCHMARK" ]] || launch_die "missing benchmark: $BENCHMARK"
[[ -f "$PROBE" ]] || launch_die "missing CCL probe: $PROBE"

init_run_dir pvc-qualification
install_exit_receipt_trap
printf 'RUN_DIR=%s\n' "$RUN_DIR"
collect_base_provenance "$PROJECT_ROOT"
record_and_enforce_cost 1 3 0.75
hash_launch_inputs "$SCRIPT_DIR/common.sh" "$SCRIPT_DIR/pvc_rank_launch.sh" "$PROBE" "${BASH_SOURCE[0]}" "$BENCHMARK"

# The intel/23.1 module and its +gpu environment are the cost-free ABI probe's
# only passing stack. Intel 24 GPU fails on an undefined SYCL symbol; its other
# oneCCL environment is CPU-only. Temporarily relax nounset for vendor scripts.
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
  printf 'qualification_scope\tactive XPU PyTorch + IPEX + XPU oneCCL; eight tiles; exact two-rank CCL; eight-rank workload\n'
  printf 'campaign_id\t%s\n' "$CAMPAIGN_ID"
  printf 'pvc_module\t%s\n' "$PVC_MODULE"
  printf 'pvc_venv\t%s\n' "$PVC_VENV"
  printf 'nodes\t1\n'
  printf 'maximum_ranks\t8\n'
  printf 'script_default_walltime_minutes\t15\n'
  printf 'script_default_charge_upper_bound_su\t0.75\n'
  printf 'pvc_python\t%s\n' "$PVC_PYTHON"
  printf 'pvc_site_packages\t%s\n' "$PVC_SITE_PACKAGES"
  printf 'intel_collective_path\toneccl_host_staged_ofi_tcp\n'
  printf 'device_direct_collectives\tfalse\n'
  printf 'affinity_discovery_cpu_bind\tcores\n'
  printf 'qualification_collective_cpu_bind\tnone\n'
  printf 'worker_affinity_strategy\tallocation-derived unique NUMA-0 CPUs\n'
} >>"$RUN_DIR/run_metadata.tsv"

export CCL_LOG_LEVEL="${CCL_LOG_LEVEL:-info}"
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-ofi}"
# The Intel23 oneCCL device kernel is ABI-incompatible with the installed PVC
# runtime. These documented non-topology algorithms stage XPU buffers through
# host memory and avoid the incompatible device-side kernels.spv module.
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

host="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR="$host"

{
  hostname
  command -v sycl-ls >/dev/null && sycl-ls || true
  command -v xpu-smi >/dev/null && xpu-smi discovery || true
  command -v xpumcli >/dev/null && xpumcli discovery -l || true
  command -v fi_info >/dev/null && fi_info || true
  lspci | grep -Ei 'display|vga|intel' || true
} >"$RUN_DIR/provenance/pvc_node.txt" 2>&1

"$PVC_PYTHON" "$PROBE" --mode inspect --output-dir "$RUN_DIR/provenance/stack" \
  >"$RUN_DIR/logs/stack-inspect.out" 2>"$RUN_DIR/logs/stack-inspect.err"

"$PVC_PYTHON" -c 'import pathlib, intel_extension_for_pytorch as ipex, oneccl_bindings_for_pytorch as ccl, torchvision; roots={pathlib.Path(ipex.__file__).resolve().parent,pathlib.Path(ccl.__file__).resolve().parent,pathlib.Path(torchvision.__file__).resolve().parent}; files=sorted({p for root in roots for p in root.rglob("*.so")}); print("\n".join(str(p) for p in files[:64]))' \
  >"$RUN_DIR/provenance/xpu_extension_files.txt"
[[ -s "$RUN_DIR/provenance/xpu_extension_files.txt" ]] || launch_die "no XPU extension shared libraries were found"
while IFS= read -r extension; do
  [[ -n "$extension" ]] || continue
  printf '\n### %s\n' "$extension"
  ldd "$extension"
done <"$RUN_DIR/provenance/xpu_extension_files.txt" >"$RUN_DIR/provenance/xpu_extension_ldd.txt" 2>&1
if grep -F 'not found' "$RUN_DIR/provenance/xpu_extension_ldd.txt"; then
  launch_die "unresolved XPU shared-library dependency detected"
fi

export MASTER_PORT=$((30400 + SLURM_JOB_ID % 500))
srun --exclusive --nodes=1 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=4 \
  --cpu-bind=cores \
  --kill-on-bad-exit=1 --wait=0 \
  --output="$RUN_DIR/logs/tile-check.%t.out" --error="$RUN_DIR/logs/tile-check.%t.err" \
  bash "$SCRIPT_DIR/pvc_rank_launch.sh" "$PVC_PYTHON" "$PROBE" --mode tile \
  --output-dir "$RUN_DIR/provenance/tiles"
[[ "$(find "$RUN_DIR/provenance/tiles" -name 'tile_*.json' -type f | wc -l)" -eq 8 ]] || \
  launch_die "eight-tile probe did not produce eight receipts"
export CCL_WORKER_AFFINITY="$("$PVC_PYTHON" "$PROBE" --mode affinity --output-dir "$RUN_DIR/provenance/tiles")"
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

export MASTER_PORT=$((30900 + SLURM_JOB_ID % 500))
srun --exclusive --nodes=1 --ntasks=2 --ntasks-per-node=2 --cpus-per-task=4 \
  --cpu-bind=none \
  --kill-on-bad-exit=1 --wait=0 \
  --output="$RUN_DIR/logs/ccl-exact.%t.out" --error="$RUN_DIR/logs/ccl-exact.%t.err" \
  bash "$SCRIPT_DIR/pvc_rank_launch.sh" "$PVC_PYTHON" "$PROBE" --mode ccl \
  --output-dir "$RUN_DIR/provenance/ccl-exact"
[[ "$(find "$RUN_DIR/provenance/ccl-exact" -name 'ccl_rank_*.json' -type f | wc -l)" -eq 2 ]] || \
  launch_die "exact CCL probe did not produce two rank receipts"

export MASTER_PORT=$((31400 + SLURM_JOB_ID % 500))
srun --exclusive --nodes=1 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=4 \
  --cpu-bind=none \
  --kill-on-bad-exit=1 --wait=0 \
  --output="$RUN_DIR/logs/workload-qualification.%t.out" \
  --error="$RUN_DIR/logs/workload-qualification.%t.err" \
  bash "$SCRIPT_DIR/pvc_rank_launch.sh" "$PVC_PYTHON" "$BENCHMARK" \
  --profile qualification --accelerator xpu --backend ccl \
  --model transformer --comm-mode native --sync-every 1 \
  --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration --seed 2026010043 \
  --output-dir "$RUN_DIR/results" --campaign-id "$CAMPAIGN_ID" \
  --run-id pvc-q-transformer-native-ws8

qualification_step=0
for model in resnet50; do
  qualification_step=$((qualification_step + 1))
  export MASTER_PORT=$((31500 + (SLURM_JOB_ID % 100) * 10 + qualification_step))
  srun --exclusive --nodes=1 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=4 \
    --cpu-bind=none \
    --kill-on-bad-exit=1 --wait=0 \
    --output="$RUN_DIR/logs/workload-${model}-native.%t.out" \
    --error="$RUN_DIR/logs/workload-${model}-native.%t.err" \
    bash "$SCRIPT_DIR/pvc_rank_launch.sh" "$PVC_PYTHON" "$BENCHMARK" \
    --profile qualification --accelerator xpu --backend ccl \
    --model "$model" --comm-mode native --sync-every 1 \
    --batch-size 2 --iters 2 --warmup-iters 1 --skip-calibration \
    --seed 2026010043 --output-dir "$RUN_DIR/results" --campaign-id "$CAMPAIGN_ID" \
    --run-id "pvc-q-${model}-native-ws8"
done

for mode in gpu-topk gpu-quant cpu-quant powersgd ddp-overlap; do
  qualification_step=$((qualification_step + 1))
  export MASTER_PORT=$((31500 + (SLURM_JOB_ID % 100) * 10 + qualification_step))
  srun --exclusive --nodes=1 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=4 \
    --cpu-bind=none \
    --kill-on-bad-exit=1 --wait=0 \
    --output="$RUN_DIR/logs/workload-transformer-${mode}.%t.out" \
    --error="$RUN_DIR/logs/workload-transformer-${mode}.%t.err" \
    bash "$SCRIPT_DIR/pvc_rank_launch.sh" "$PVC_PYTHON" "$BENCHMARK" \
    --profile qualification --accelerator xpu --backend ccl \
    --model transformer --comm-mode "$mode" --sync-every 1 \
    --batch-size 2 --iters 2 --warmup-iters 1 --skip-calibration \
    --seed 2026010043 --output-dir "$RUN_DIR/results" --campaign-id "$CAMPAIGN_ID" \
    --run-id "pvc-q-transformer-${mode}-ws8"
done

benchmark_sha="$(sha256sum "$BENCHMARK" | awk '{print $1}')"
{
  printf 'status\tPASS\n'
  printf 'platform\tpvc-one-node-eight-tile\n'
  printf 'job_id\t%s\n' "$SLURM_JOB_ID"
  printf 'benchmark_sha256\t%s\n' "$benchmark_sha"
  printf 'probe_sha256\t%s\n' "$(sha256sum "$PROBE" | awk '{print $1}')"
  printf 'stack_inspect_sha256\t%s\n' "$(sha256sum "$RUN_DIR/provenance/stack/stack_inspect.json" | awk '{print $1}')"
  printf 'pvc_module\t%s\n' "$PVC_MODULE"
  printf 'pvc_venv\t%s\n' "$PVC_VENV"
  printf 'receipt_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$RUN_DIR/PVC_STACK_QUALIFIED.tsv"
