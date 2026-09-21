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
BENCHMARK="$RED_ROOT/control_workload_benchmark_v4.py"
VENV="${RED_LS6_VENV:-/scratch/08447/abd2000/redctl-ls6/venv}"
SEED="${PAPER_SEED:-2026010043}"
CAMPAIGN_ID="${CAMPAIGN_ID:-redctl-r4-2026-09-21}"
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

RUN_DIR="$RED_ROOT/runs/r4-${RED_SET}-${NODES}n-${SLURM_JOB_ID}"
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
# On these A100-PCIe nodes the HCA sits across the CPU root complex from the
# GPUs, and NCCL's default distance rule disables GPUDirect ("distance 6 > 4").
# SYS permits GDR across the system; the level used is recorded.
export NCCL_NET_GDR_LEVEL="${RED_GDR_LEVEL:-SYS}"
echo "nccl_net_gdr_level=$NCCL_NET_GDR_LEVEL" >>"$RUN_DIR/provenance/settings.txt"

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

cal_ref="${rid}-resnet50-native"
reuse=(--skip-calibration --calibration-reference "$cal_ref")
paper() {  # paper <model> <mode> [extra args...]
  local model="$1" mode="$2"; shift 2
  run_condition "paper-${model}-${mode}" \
    --profile paper --model "$model" --comm-mode "$mode" --sync-every 1 \
    --batch-size 32 --iters 40 --warmup-iters 10 "$@" "${reuse[@]}" \
    --run-id "${rid}-${model}-${mode}" "${common[@]}"
}
wall() {   # wall <model> <mode> [extra args...]
  local model="$1" mode="$2"; shift 2
  run_condition "wall-${model}-${mode}" \
    --profile paper --model "$model" --comm-mode "$mode" --timing-mode wall --sync-every 1 \
    --batch-size 32 --iters 40 --warmup-iters 10 "$@" "${reuse[@]}" \
    --run-id "${rid}-${model}-${mode}-wall" "${common[@]}"
}
tau() {    # tau <model> <sync_every> <barrier|nobarrier> <phase|wall>
  local model="$1" k="$2" bar="$3" timing="$4" extra=() label="tau${k}-${bar}"
  [[ "$bar" == nobarrier ]] && extra+=(--tau-no-barrier)
  [[ "$timing" == wall ]] && { extra+=(--timing-mode wall); label="${label}-wall"; }
  run_condition "${label}-${model}" \
    --profile paper --model "$model" --comm-mode native --sync-every "$k" \
    --batch-size 32 --iters 40 --warmup-iters 10 "${extra[@]}" "${reuse[@]}" \
    --run-id "${rid}-${model}-native-${label}" "${common[@]}"
}
case "$RED_SET" in
  pilot)
    for mode in gpu-quant-gathersum fp16-native; do
      run_condition "smoke-tf-${mode}" --profile qualification --model transformer --comm-mode "$mode" --sync-every 1 \
        --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration --distributed-timeout-s 120 \
        --run-id "${rid}-smoke-tf-${mode}" "${common[@]}"
    done
    for mode in gpu-quant-reducible gpu-quant-gathersum; do
      run_condition "verify-tf-${mode}" --profile qualification --model transformer --comm-mode "$mode" --sync-every 1 \
        --verify-equivalence --batch-size 4 --iters 5 --warmup-iters 1 --skip-calibration --distributed-timeout-s 120 \
        --run-id "${rid}-verify-tf-${mode}" "${common[@]}"
    done
    run_condition "smoke-tf-tau4-nobarrier" --profile qualification --model transformer --comm-mode native --sync-every 4 \
      --tau-no-barrier --batch-size 4 --iters 4 --warmup-iters 1 --skip-calibration --distributed-timeout-s 120 \
      --run-id "${rid}-smoke-tf-tau4nb" "${common[@]}"
    run_condition "smoke-tf-int8hook-wall" --profile qualification --model transformer --comm-mode ddp-int8-hook --timing-mode wall --sync-every 1 \
      --batch-size 4 --iters 3 --warmup-iters 1 --skip-calibration --distributed-timeout-s 120 \
      --run-id "${rid}-smoke-tf-int8hook-wall" "${common[@]}"
    ;;
  j2)
    run_condition "paper-resnet50-native" --profile paper --model resnet50 --comm-mode native --sync-every 1 \
      --batch-size 32 --iters 40 --warmup-iters 10 --run-id "$cal_ref" "${common[@]}"
    paper transformer native
    for model in transformer resnet50; do
      for mode in fp16-native gpu-quant gpu-quant-reducible gpu-quant-gathersum ddp-int8-hook ddp-fp16-hook gpu-topk; do
        paper "$model" "$mode"
      done
      for mode in native fp16-native gpu-quant gpu-quant-reducible gpu-quant-gathersum ddp-int8-hook ddp-fp16-hook gpu-topk; do
        wall "$model" "$mode"
      done
    done
    paper resnet50 cpu-quant
    wall resnet50 cpu-quant
    ;;
  j3)
    run_condition "paper-resnet50-native" --profile paper --model resnet50 --comm-mode native --sync-every 1 \
      --batch-size 32 --iters 40 --warmup-iters 10 --run-id "$cal_ref" "${common[@]}"
    paper transformer native
    for model in transformer resnet50; do
      for mode in fp16-native gpu-quant gpu-quant-reducible gpu-quant-gathersum ddp-int8-hook ddp-fp16-hook gpu-topk; do
        paper "$model" "$mode"
      done
      for mode in native fp16-native gpu-quant gpu-quant-reducible gpu-quant-gathersum ddp-int8-hook ddp-fp16-hook gpu-topk; do
        wall "$model" "$mode"
      done
    done
    ;;
  j4)
    run_condition "paper-resnet50-native" --profile paper --model resnet50 --comm-mode native --sync-every 1 \
      --batch-size 32 --iters 40 --warmup-iters 10 --run-id "$cal_ref" "${common[@]}"
    for model in transformer resnet50; do
      for k in 1 4; do for bar in barrier nobarrier; do for timing in phase wall; do
        [[ "$k" == 1 && "$bar" == nobarrier ]] && continue
        tau "$model" "$k" "$bar" "$timing"
      done; done; done
    done
    for model in transformer resnet50; do
      for mode in native gpu-quant; do
        run_condition "tail-${model}-${mode}" --profile extension --model "$model" --comm-mode "$mode" --sync-every 1 \
          --batch-size 32 --iters 400 --warmup-iters 10 "${reuse[@]}" \
          --run-id "${rid}-${model}-${mode}-tail400" "${common[@]}"
      done
    done
    ;;
  j5)
    # E6 pairing job: decisive isolated-shape pairs (cases 3, 12) and the
    # hook pair (case 8) re-run as M2/M3 inside ONE allocation, plus the
    # tau=1 wall reference for cases 9 and 10, so every scored delta is a
    # same-allocation pair.
    run_condition "paper-resnet50-native" --profile paper --model resnet50 --comm-mode native --sync-every 1 \
      --batch-size 32 --iters 40 --warmup-iters 10 --run-id "$cal_ref" "${common[@]}"
    for model in transformer resnet50; do
      wall "$model" native
      wall "$model" gpu-quant-reducible
      wall "$model" gpu-quant-gathersum
      wall "$model" ddp-int8-hook
      wall "$model" ddp-fp16-hook
    done
    wall resnet50 cpu-quant
    tau transformer 4 barrier wall
    tau transformer 4 nobarrier wall
    tau resnet50 4 barrier wall
    tau resnet50 4 nobarrier wall
    ;;
esac

# Transport verdict from the NCCL logs: GDRDMA on the NET/IB path or not.
set +e
{
  echo "nccl_net_gdr_level=$NCCL_NET_GDR_LEVEL"
  echo "gdr_lines=$(grep -h 'GDRDMA' "$RUN_DIR"/logs/nccl.*.log 2>/dev/null | wc -l)"
  echo "netib_lines=$(grep -h 'via NET/IB' "$RUN_DIR"/logs/nccl.*.log 2>/dev/null | wc -l)"
  grep -h 'via NET/IB' "$RUN_DIR"/logs/nccl.*.log 2>/dev/null | sed 's/^[^ ]* //' | sort | uniq -c | sort -rn | head -5
  grep -hE 'NET/IB : Using|GDR|GPU Direct RDMA' "$RUN_DIR"/logs/nccl.*.log 2>/dev/null | sed 's/^[^ ]* //' | sort -u | head -8
  echo "completed_conditions=$step"
} >"$RUN_DIR/provenance/transport_verdict.txt"
cat "$RUN_DIR/provenance/transport_verdict.txt"
exit 0
