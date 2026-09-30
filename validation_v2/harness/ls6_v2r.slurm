#!/bin/bash
#SBATCH --job-name=tpds-v2
#SBATCH --partition=gpu-a100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --export=ALL
# allocation: pass it at submit time with sbatch -A <account>

# TPDS v2 validation campaign (2026-09-23) on Lonestar6 A100 over GPUDirect
# RDMA (NCCL_NET_GDR_LEVEL=SYS, as in the round-4 jobs). One job runs one
# manifest (make_manifest_v2.py) in order: V_JOB=SMOKE|V1|V2|V3|L1. Node
# count and time come from sbatch -N and -t. A failed condition is recorded
# and the job continues; nothing is rerun.

set -Eeuo pipefail
RED_ROOT="${RED_ROOT:?set RED_ROOT}"
V_JOB="${V_JOB:?set V_JOB}"
BENCHMARK="$RED_ROOT/control_workload_benchmark_v5_1.py"
RUN_LIST="$RED_ROOT/v2/run_list_${V_JOB}.sh"
MANIFEST="$RED_ROOT/v2/manifest_${V_JOB}.json"
VENV="${RED_LS6_VENV:-$RED_ROOT/../venv}"
NODES="${SLURM_JOB_NUM_NODES:?}"
RPN="${RED_RPN:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[[ "$RPN" =~ ^[1-9]$ ]] || { echo "FATAL: could not detect GPUs per node (got '$RPN')" >&2; exit 1; }
WORLD=$((NODES * RPN))
export RPN
[[ -f "$RUN_LIST" && -f "$MANIFEST" ]] || { echo "FATAL: missing $RUN_LIST or $MANIFEST" >&2; exit 1; }
SEED="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "$MANIFEST")"

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
module load cuda/12.8 2>/dev/null || true
module load python3/3.9.7 2>/dev/null || true
source "$VENV/bin/activate"
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
  || { echo "FATAL: the venv does not expose CUDA on $(hostname)" >&2; exit 1; }

RUN_DIR="$RED_ROOT/runs/v2-${V_JOB}-${NODES}n-${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/logs" "$RUN_DIR/provenance" "$RUN_DIR/traces"
cp "$MANIFEST" "$RUN_LIST" "$RUN_DIR/provenance/"
scontrol show job "$SLURM_JOB_ID" >"$RUN_DIR/provenance/slurm_job.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" >"$RUN_DIR/provenance/hosts.txt"
HEAD_NODE=$(head -n 1 "$RUN_DIR/provenance/hosts.txt")
sha256sum "$BENCHMARK" "$MANIFEST" "$RUN_LIST" >"$RUN_DIR/provenance/sha256.txt"
python -c 'import torch,json; print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"nccl":torch.cuda.nccl.version()}))' \
  >"$RUN_DIR/provenance/torch.json"
srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --output="$RUN_DIR/provenance/node.%t.out" \
  bash -c 'hostname; nvidia-smi -L; nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu --format=csv'

export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=8
export NCCL_NET_GDR_LEVEL="${RED_GDR_LEVEL:-SYS}"
echo "nccl_net_gdr_level=$NCCL_NET_GDR_LEVEL world=$WORLD seed=$SEED" >"$RUN_DIR/provenance/settings.txt"

step=0
run_condition() {   # run_condition <id> <cap_minutes> <harness args...>
  local id="$1" cap="$2"; shift 2
  step=$((step + 1))
  local port=$((41000 + (10#$SLURM_JOB_ID % 20000) + step))
  local args=()
  for a in "$@"; do args+=("${a//__TRACE_DIR__/$RUN_DIR/traces}"); done
  export CONDITION_ID="$id" MASTER_PORT="$port" HEAD_NODE RUN_DIR BENCHMARK
  echo "$(date -u +%FT%TZ) start $id" >>"$RUN_DIR/provenance/timeline.txt"
  if srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --cpus-per-task=32 \
      --kill-on-bad-exit=1 --wait=0 --time="$cap" \
      --output="$RUN_DIR/logs/${id}.node%t.out" --error="$RUN_DIR/logs/${id}.node%t.err" \
      bash -c 'exec env NCCL_DEBUG_FILE="$RUN_DIR/logs/nccl.${CONDITION_ID}.%h.%p.log" \
        torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node="$RPN" --node_rank="$SLURM_PROCID" \
        --rdzv_backend=c10d --rdzv_endpoint="${HEAD_NODE}:${MASTER_PORT}" --rdzv_id="${SLURM_JOB_ID}-${CONDITION_ID}" \
        --max_restarts=0 "$BENCHMARK" "$@"' _ \
        "${args[@]}" --accelerator cuda --backend nccl --output-dir "$RUN_DIR/results" \
        --seed "$SEED" --campaign-id "tpds-v2-2026-09-23" --run-id "v2-${V_JOB}-${id}"; then
    echo "$(date -u +%FT%TZ) done $id" >>"$RUN_DIR/provenance/timeline.txt"
  else
    echo "$(date -u +%FT%TZ) FAILED $id" >>"$RUN_DIR/provenance/timeline.txt"
  fi
}

# shellcheck source=/dev/null
source "$RUN_LIST"

set +e
{
  echo "nccl_net_gdr_level=$NCCL_NET_GDR_LEVEL"
  echo "gdr_lines=$(grep -h 'GDRDMA' "$RUN_DIR"/logs/nccl.*.log 2>/dev/null | wc -l)"
  echo "netib_lines=$(grep -h 'via NET/IB' "$RUN_DIR"/logs/nccl.*.log 2>/dev/null | wc -l)"
  echo "conditions_started=$step"
  echo "conditions_failed=$(grep -c FAILED "$RUN_DIR/provenance/timeline.txt")"
} >"$RUN_DIR/provenance/transport_verdict.txt"
cat "$RUN_DIR/provenance/transport_verdict.txt"
exit 0
