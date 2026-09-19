#!/bin/bash
#SBATCH --job-name=redctl-ls6-gdr
#SBATCH --account=OTH21057
#SBATCH --partition=gpu-a100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --export=ALL

# Lonestar6 A100/HDR transport probe (2026-09-19): does NCCL take the
# GPUDirect RDMA path on this node image, and does the collective-shape
# triad reproduce over it? Node count and set come from sbatch -N and
# RED_SET=probe|triad4|triad8. Rate 3 SU/node-hour, 15-minute floor.

set -Eeuo pipefail
RED_ROOT="${RED_ROOT:?set RED_ROOT}"
RED_SET="${RED_SET:-probe}"
BENCHMARK="$RED_ROOT/control_workload_benchmark_v3.py"
VENV="${RED_LS6_VENV:-/scratch/08447/abd2000/redctl-ls6/venv}"
SEED="${PAPER_SEED:-2026010043}"
CAMPAIGN_ID="${CAMPAIGN_ID:-redctl-ls6-2026-09-19}"
NODES="${SLURM_JOB_NUM_NODES:?}"
# GPUs per node differ between LS6 partitions (3 on gpu-a100, 2 on the dev
# nodes); detect on the batch host, which is an allocated GPU node.
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

RUN_DIR="$RED_ROOT/runs/ls6-${RED_SET}-${NODES}n-${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/logs" "$RUN_DIR/provenance"
scontrol show job "$SLURM_JOB_ID" >"$RUN_DIR/provenance/slurm_job.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" >"$RUN_DIR/provenance/hosts.txt"
HEAD_NODE=$(head -n 1 "$RUN_DIR/provenance/hosts.txt")
sha256sum "$BENCHMARK" >"$RUN_DIR/provenance/benchmark.sha256"
python -c 'import torch,json; print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"nccl":torch.cuda.nccl.version()}))' \
  >"$RUN_DIR/provenance/torch.json"

# Node inventory and the kernel-module evidence for GPUDirect.
srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --output="$RUN_DIR/provenance/node.%t.out" \
  bash -c 'hostname; nvidia-smi -L; lsmod | grep -iE "peer|gdr" || echo "no-peermem-module"; command -v ibv_devinfo >/dev/null && ibv_devinfo -l || true; ip -br link | grep -iE "ib|mlx" || true'

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

step=0
run_condition() {
  local condition="$1"; shift
  step=$((step + 1))
  local port=$((41000 + (10#$SLURM_JOB_ID % 20000) + step))
  export CONDITION_ID="$condition" MASTER_PORT="$port" HEAD_NODE RUN_DIR BENCHMARK
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --cpus-per-task=32 \
    --kill-on-bad-exit=1 --wait=0 \
    --output="$RUN_DIR/logs/${condition}.node%t.out" --error="$RUN_DIR/logs/${condition}.node%t.err" \
    bash -c 'exec env NCCL_DEBUG_FILE="$RUN_DIR/logs/nccl.${CONDITION_ID}.%h.%p.log" \
      torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node="$RPN" --node_rank="$SLURM_PROCID" \
      --rdzv_backend=c10d --rdzv_endpoint="${HEAD_NODE}:${MASTER_PORT}" --rdzv_id="${SLURM_JOB_ID}-${CONDITION_ID}" \
      --max_restarts=0 "$BENCHMARK" "$@"' _ "$@"
}

common=(--accelerator cuda --backend nccl --output-dir "$RUN_DIR/results" --seed "$SEED" --campaign-id "$CAMPAIGN_ID")
rid="redL-ws${WORLD}"

case "$RED_SET" in
  probe)
    run_condition "smoke-tf-native" \
      --profile qualification --model transformer --comm-mode native --sync-every 1 \
      --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration --distributed-timeout-s 120 \
      --run-id "${rid}-smoke-tf-native" "${common[@]}"
    run_condition "smoke-tf-reducible" \
      --profile qualification --model transformer --comm-mode gpu-quant-reducible --sync-every 1 \
      --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration --distributed-timeout-s 120 \
      --run-id "${rid}-smoke-tf-reducible" "${common[@]}"
    run_condition "paper-resnet50-native" \
      --profile paper --model resnet50 --comm-mode native --sync-every 1 \
      --batch-size 32 --iters 40 --warmup-iters 10 \
      --run-id "${rid}-resnet50-native" "${common[@]}"
    for model in transformer resnet50; do
      for mode in native gpu-quant gpu-quant-reducible; do
        [[ "$model" == resnet50 && "$mode" == native ]] && continue
        run_condition "paper-${model}-${mode}" \
          --profile paper --model "$model" --comm-mode "$mode" --sync-every 1 \
          --batch-size 32 --iters 40 --warmup-iters 10 \
          --skip-calibration --calibration-reference "${rid}-resnet50-native" \
          --run-id "${rid}-${model}-${mode}" "${common[@]}"
      done
    done
    ;;
  triad4|triad8)
    run_condition "paper-resnet50-native" \
      --profile paper --model resnet50 --comm-mode native --sync-every 1 \
      --batch-size 32 --iters 40 --warmup-iters 10 \
      --run-id "${rid}-resnet50-native" "${common[@]}"
    for model in transformer resnet50; do
      for mode in native gpu-quant gpu-quant-reducible ddp-int8-hook ddp-fp16-hook; do
        [[ "$model" == resnet50 && "$mode" == native ]] && continue
        run_condition "paper-${model}-${mode}" \
          --profile paper --model "$model" --comm-mode "$mode" --sync-every 1 \
          --batch-size 32 --iters 40 --warmup-iters 10 \
          --skip-calibration --calibration-reference "${rid}-resnet50-native" \
          --run-id "${rid}-${model}-${mode}" "${common[@]}"
      done
    done
    ;;
esac

# Transport verdict from the NCCL logs: GDRDMA on the NET/IB path or not.
{
  echo "gdr_lines=$(grep -h 'GDRDMA' "$RUN_DIR"/logs/nccl.*.log 2>/dev/null | wc -l)"
  echo "netib_lines=$(grep -h 'via NET/IB' "$RUN_DIR"/logs/nccl.*.log 2>/dev/null | wc -l)"
  grep -h 'via NET/IB' "$RUN_DIR"/logs/nccl.*.log 2>/dev/null | sort | uniq -c | sort -rn | head -5
  grep -hE 'NET/IB : Using|GDR|GPU Direct RDMA' "$RUN_DIR"/logs/nccl.*.log 2>/dev/null | sort -u | head -8
} >"$RUN_DIR/provenance/transport_verdict.txt"
cat "$RUN_DIR/provenance/transport_verdict.txt"
echo "completed_conditions=$step" >>"$RUN_DIR/provenance/transport_verdict.txt"
