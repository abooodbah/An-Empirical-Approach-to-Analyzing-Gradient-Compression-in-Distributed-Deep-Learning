#!/bin/bash
#SBATCH --job-name=redctl-e12a
#SBATCH --account=OTH21057
#SBATCH --partition=gpu-a100-dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --export=ALL

# E12 (diagnostic/E12_PROTOCOL.md), set e12a, scored allocation a: 15 case types (12, and the mirrors M01,
# M04, M12) on two workloads, all traced, with the BF16 production baseline and the
# matched-encoding control.
# Development queue, 2 nodes, 6 ranks. Written by make_ls6_e12.py from
# ls6_e11b.slurm; do not edit by hand. Environment selection is as in ls6_e8.slurm.
# Rate 3 SU/node-hour.

set -Eeuo pipefail
RED_ROOT="${RED_ROOT:?set RED_ROOT}"
RED_SET="e12a"
BENCH="$RED_ROOT/control_workload_benchmark_v5_5.py"
COLLECTIVE_PROBE="$RED_ROOT/collective_probe.py"
VENV="${RED_LS6_VENV:-/scratch/08447/abd2000/redctl-ls6/venv}"
SEED="${PAPER_SEED:-2026010043}"
CAMPAIGN_ID="${CAMPAIGN_ID:-redctl-e12-2026-10}"
NODES="${SLURM_JOB_NUM_NODES:?}"
want="$(cut -d' ' -f1 "$RED_ROOT/CODE_v55.sha256")"
[[ "$(sha256sum "$BENCH" | cut -d' ' -f1)" == "$want" ]] || { echo "FATAL: harness v5.5 differs from CODE_v55.sha256" >&2; exit 1; }
# E12 freeze (diagnostic/E12_PROTOCOL.md section 9): every frozen file is
# staged in $RED_ROOT or $RED_ROOT/prereg with its frozen hash, and this
# script is one of the frozen job scripts.
FREEZE="$RED_ROOT/prereg/E12_FREEZE.sha256"
[[ -f "$FREEZE" ]] || { echo "FATAL: prereg/E12_FREEZE.sha256 is not staged" >&2; exit 1; }
slurm_digests=" "
while read -r digest path; do
  [[ -n "$digest" ]] || continue
  path="${path%$'\r'}"
  base="${path##*/}"
  if [[ "$base" == *.slurm ]]; then slurm_digests+="$digest "; continue; fi
  staged="$RED_ROOT/$base"; [[ -f "$staged" ]] || staged="$RED_ROOT/prereg/$base"
  [[ -f "$staged" ]] || { echo "FATAL: frozen file $base is not staged" >&2; exit 1; }
  [[ "$(sha256sum "$staged" | cut -d' ' -f1)" == "$digest" ]] || { echo "FATAL: staged $base differs from E12_FREEZE.sha256" >&2; exit 1; }
done <"$FREEZE"
[[ "$slurm_digests" == *" $(sha256sum "$0" | cut -d' ' -f1) "* ]] || { echo "FATAL: this job script is not a frozen E12 script" >&2; exit 1; }
RPN="${RED_RPN:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[[ "$RPN" =~ ^[1-9]$ ]] || { echo "FATAL: could not detect GPUs per node (got '$RPN')" >&2; exit 1; }
WORLD=$((NODES * RPN))
export RPN

# Select a CUDA-capable environment on this node image: the cu121 wheel with
# and without the cuda module, then the MOSAIC venv. The choice is recorded.
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cuda_ok() { "$1" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count()>0 else 1)' >/dev/null 2>&1; }
ENV_CHOICE=""
module load cuda/12.8 2>/dev/null; module load python3/3.9.7 2>/dev/null
if cuda_ok "$VENV/bin/python"; then ENV_CHOICE="redctl-venv+cuda12.8-module"; fi
if [[ -z "$ENV_CHOICE" ]]; then module unload cuda/12.8 2>/dev/null; if cuda_ok "$VENV/bin/python"; then ENV_CHOICE="redctl-venv-no-cuda-module"; fi; fi
if [[ -z "$ENV_CHOICE" ]]; then module purge 2>/dev/null; unset LD_LIBRARY_PATH; if cuda_ok "$VENV/bin/python"; then ENV_CHOICE="redctl-venv-purged"; fi; fi
if [[ -z "$ENV_CHOICE" ]]; then module load cuda/12.8 2>/dev/null; VENV=/scratch/08447/abd2000/mosaic-scale/venv; if cuda_ok "$VENV/bin/python"; then ENV_CHOICE="mosaic-venv+cuda12.8-module"; fi; fi
[[ -n "$ENV_CHOICE" ]] || { echo "FATAL: no environment exposes CUDA on $(hostname)" >&2; exit 1; }
source "$VENV/bin/activate"
echo "env_choice=$ENV_CHOICE venv=$VENV"

RUN_DIR="$RED_ROOT/runs/e12-${RED_SET}-${NODES}n-${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/logs" "$RUN_DIR/provenance" "$RUN_DIR/traces" "$RUN_DIR/probes"
scontrol show job "$SLURM_JOB_ID" >"$RUN_DIR/provenance/slurm_job.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" >"$RUN_DIR/provenance/hosts.txt"
HEAD_NODE=$(head -n 1 "$RUN_DIR/provenance/hosts.txt")
sha256sum "$BENCH" "$COLLECTIVE_PROBE" "$RED_ROOT/e12_sets.sh" "$0" "$RED_ROOT"/prereg/* >"$RUN_DIR/provenance/benchmark.sha256"
python -c 'import torch,json; print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"nccl":torch.cuda.nccl.version()}))' \
  >"$RUN_DIR/provenance/torch.json"

srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --output="$RUN_DIR/provenance/node.%t.out" \
  bash -c 'hostname; nvidia-smi -L; lsmod | grep -iE "peer|gdr" || echo "no-peermem-module"; command -v ibv_devinfo >/dev/null && ibv_devinfo -l || true; ip -br link | grep -iE "ib|mlx" || true'

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export NCCL_NET_GDR_LEVEL="${RED_GDR_LEVEL:-SYS}"
echo "nccl_net_gdr_level=$NCCL_NET_GDR_LEVEL" >>"$RUN_DIR/provenance/settings.txt"

step=0
run_script() {  # run_script <condition> <script> <args...>
  local condition="$1" script="$2"; shift 2
  step=$((step + 1))
  local port=$((41000 + (10#$SLURM_JOB_ID % 20000) + step))
  export CONDITION_ID="$condition" MASTER_PORT="$port" HEAD_NODE RUN_DIR SCRIPT_PATH="$script"
  local rc=0
  echo "$(date -u +%FT%TZ) start ${condition}" >>"$RUN_DIR/provenance/timeline.txt"
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --cpus-per-task=32 \
    --kill-on-bad-exit=1 --wait=0 \
    --output="$RUN_DIR/logs/${condition}.node%t.out" --error="$RUN_DIR/logs/${condition}.node%t.err" \
    bash -c 'exec env NCCL_DEBUG_FILE="$RUN_DIR/logs/nccl.${CONDITION_ID}.%h.%p.log" \
      torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node="$RPN" --node_rank="$SLURM_PROCID" \
      --rdzv_backend=c10d --rdzv_endpoint="${HEAD_NODE}:${MASTER_PORT}" --rdzv_id="${SLURM_JOB_ID}-${CONDITION_ID}" \
      --max_restarts=0 "$SCRIPT_PATH" "$@"' _ "$@" || rc=$?
  echo "$(date -u +%FT%TZ) done ${condition} rc=${rc}" >>"$RUN_DIR/provenance/timeline.txt"
  echo "${condition} rc=${rc}" >>"$RUN_DIR/provenance/conditions.txt"
  return "$rc"
}

common=(--accelerator cuda --backend nccl --output-dir "$RUN_DIR/results" --seed "$SEED" --campaign-id "$CAMPAIGN_ID")
rid="redE12-ws${WORLD}"
cal_ref="${rid}-resnet50-native"
reuse=(--skip-calibration --calibration-reference "$cal_ref")
node_facts() {
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --kill-on-bad-exit=0 \
    --output="$RUN_DIR/provenance/node_facts.%t.txt" \
    bash -c 'hostname; echo "== gpu"; nvidia-smi --query-gpu=index,name,memory.total,driver_version,vbios_version,pci.bus_id --format=csv
      echo "== topology"; nvidia-smi topo -m 2>&1 | head -40
      echo "== cpu"; lscpu | grep -E "Model name|Socket|Core|Thread|NUMA"
      echo "== numa"; numactl -H 2>&1 | head -20
      echo "== adapter"; ibv_devinfo 2>&1 | grep -E "hca_id|fw_ver|board_id|active_mtu|rate|link_layer"
      for d in /sys/class/infiniband/*; do echo "$d numa_node=$(cat "$d/device/numa_node" 2>/dev/null)"; done
      echo "== pci"; lspci 2>/dev/null | grep -iE "nvidia|mellanox" | head -20; echo "== kernel"; uname -r' || true
}
preflight() {
  run_script "preflight-$1" "$BENCH" --profile qualification --model "$1" --comm-mode native --sync-every 1 \
    --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration --distributed-timeout-s 120 \
    --run-id "${rid}-preflight-$1" "${common[@]}" \
    || { echo "FATAL: preflight failed; no outcome run was attempted" >&2; exit 1; }
}
dtype_check() {
  run_script "dtype-check" "$COLLECTIVE_PROBE" --label dtype --output "$RUN_DIR/probes/dtype.json" \
    --sizes 1024,4096 --dtype-check --accelerator cuda --backend nccl || true
}
probe() {  # probe <label> [GDR level]; label ends in default, ring, or tree
  local label="$1" level="${2:-SYS}" collectives="allreduce,allgather"
  local saved_level="$NCCL_NET_GDR_LEVEL" saved_subsys="$NCCL_DEBUG_SUBSYS"
  export NCCL_NET_GDR_LEVEL="$level" NCCL_DEBUG_SUBSYS=INIT,NET,ENV,TUNING
  case "$label" in *ring) export NCCL_ALGO=Ring ;; *tree) export NCCL_ALGO=Tree; collectives=allreduce ;; esac
  run_script "probe-$label" "$COLLECTIVE_PROBE" --label "$label" --output "$RUN_DIR/probes/$label.json" \
    --collectives "$collectives" --sizes 4,1024,1048576,16777216,67108864,268435456,1073741824 \
    --max-rank-bytes 8589934592 --accelerator cuda --backend nccl || true
  unset NCCL_ALGO
  export NCCL_NET_GDR_LEVEL="$saved_level" NCCL_DEBUG_SUBSYS="$saved_subsys"
}
hs_probe() { probe "hs-$1" LOC; }
calib() {
  run_script "paper-resnet50-native" "$BENCH" --profile paper --model resnet50 --comm-mode native --sync-every 1 \
    --batch-size 32 --iters 40 --warmup-iters 10 --run-id "$cal_ref" "${common[@]}" \
    || { echo "FATAL: calibration failed; no outcome run was attempted" >&2; exit 1; }
}
equivalence() {
  run_script "equiv-$1-$2" "$BENCH" --profile qualification --model "$1" --comm-mode "$2" --batch-size 32 \
    --iters 3 --warmup-iters 1 --skip-calibration --verify-equivalence \
    --run-id "${rid}-$1-$2-equiv" "${common[@]}" || true
}
e9_fidelity() {
  run_script "fid-$1-$2" "$BENCH" --profile qualification --model "$1" --comm-mode "$2" --batch-size 32 \
    --iters 8 --warmup-iters 2 --skip-calibration --verify-fidelity \
    --run-id "${rid}-$1-$2-fid" "${common[@]}" || true
}
e9_phase() {
  run_script "phase-$1-$2" "$BENCH" --profile extension --model "$1" --comm-mode "$2" --sync-every 1 \
    --batch-size 32 --iters 40 --warmup-iters 10 "${reuse[@]}" \
    --run-id "${rid}-$1-$2-phase" "${common[@]}" || true
}
e9_block() {
  run_script "block-$1-$2-p$3" "$BENCH" --profile validation --model "$1" --comm-mode "$2" \
    --timing-mode block --sync-every 1 --batch-size 32 --iters 100 --warmup-iters 20 --input-pool 4 \
    "${reuse[@]}" --run-id "${rid}-$1-$2-block-p$3" "${common[@]}" || true
}
# E12 conditions (diagnostic/E12_PROTOCOL.md section 3). A tag is a harness
# mode, with "-bf16" for --dtype bf16; the int8 hook always runs synchronously.
tag_args() {  # tag_args <tag>: sets MODE and DTYPE_ARGS, and the int8 hook's form
  local tag="$1"
  MODE="${tag%-bf16}"
  DTYPE_ARGS=()
  if [[ "$tag" == *-bf16 ]]; then DTYPE_ARGS=(--dtype bf16); fi
  if [[ "$MODE" == ddp-int8-hook ]]; then export RED_HOOK_SYNC=1; else unset RED_HOOK_SYNC; fi
}
e12_equiv() {  # e12_equiv <model> <tag>: the matched control's arms cross-check each other
  tag_args "$2"
  run_script "equiv-$1-$2" "$BENCH" --profile qualification --model "$1" --comm-mode "$MODE" \
    ${DTYPE_ARGS[@]+"${DTYPE_ARGS[@]}"} --batch-size "$E12_BATCH" --iters 3 --warmup-iters 1 --skip-calibration \
    --verify-equivalence --run-id "${rid}-$1-$2-b${E12_BATCH}-equiv" "${common[@]}" || true
  unset RED_HOOK_SYNC
}
e12_phase() {  # e12_phase <model> <tag>: M1, the rules' input (extension profile, phase timing)
  tag_args "$2"
  run_script "phase-$1-$2" "$BENCH" --profile extension --model "$1" --comm-mode "$MODE" \
    ${DTYPE_ARGS[@]+"${DTYPE_ARGS[@]}"} --sync-every 1 --batch-size "$E12_BATCH" --iters 40 --warmup-iters 10 \
    "${reuse[@]}" --run-id "${rid}-$1-$2-b${E12_BATCH}-phase" "${common[@]}" || true
  unset RED_HOOK_SYNC
}
e12_trace() {  # e12_trace <model> <tag>: the replay's input, as E8's trace runs (ranks 0 and 3)
  tag_args "$2"
  run_script "trace-$1-$2" "$BENCH" --profile qualification --model "$1" --comm-mode "$MODE" \
    ${DTYPE_ARGS[@]+"${DTYPE_ARGS[@]}"} --timing-mode wall --batch-size "$E12_BATCH" --iters 8 --warmup-iters 20 \
    --input-pool 4 --trace-dir "$RUN_DIR/traces" --trace-ranks 0,3 --skip-calibration \
    --run-id "${rid}-trace-$1-$2-b${E12_BATCH}" "${common[@]}" || true
  unset RED_HOOK_SYNC
}
e12_block() {  # e12_block <model> <tag> <pass>: the outcome (validation profile, block timing)
  tag_args "$2"
  run_script "block-$1-$2-p$3" "$BENCH" --profile validation --model "$1" --comm-mode "$MODE" \
    ${DTYPE_ARGS[@]+"${DTYPE_ARGS[@]}"} --timing-mode block --sync-every 1 --batch-size "$E12_BATCH" --iters 100 \
    --warmup-iters 20 --input-pool 4 "${reuse[@]}" --run-id "${rid}-$1-$2-b${E12_BATCH}-block-p$3" "${common[@]}" || true
  unset RED_HOOK_SYNC
}
e12_fid() {  # e12_fid <model> <tag>: per-exchange error against the dense average (qualification job)
  tag_args "$2"
  run_script "fid-$1-$2" "$BENCH" --profile qualification --model "$1" --comm-mode "$MODE" \
    ${DTYPE_ARGS[@]+"${DTYPE_ARGS[@]}"} --batch-size "$E12_BATCH" --iters 8 --warmup-iters 2 --skip-calibration \
    --verify-fidelity --run-id "${rid}-$1-$2-b${E12_BATCH}-fid" "${common[@]}" || true
  unset RED_HOOK_SYNC
}
# shellcheck source=/dev/null
source "$RED_ROOT/e12_sets.sh"

set +e
{
  echo "nccl_net_gdr_level=$NCCL_NET_GDR_LEVEL"
  echo "gdr_lines=$(grep -h 'GDRDMA' "$RUN_DIR"/logs/nccl.*.log 2>/dev/null | wc -l)"
  echo "netib_lines=$(grep -h 'via NET/IB' "$RUN_DIR"/logs/nccl.*.log 2>/dev/null | wc -l)"
  echo "completed_conditions=$step"
} >"$RUN_DIR/provenance/transport_verdict.txt"
cat "$RUN_DIR/provenance/transport_verdict.txt"
exit 0
