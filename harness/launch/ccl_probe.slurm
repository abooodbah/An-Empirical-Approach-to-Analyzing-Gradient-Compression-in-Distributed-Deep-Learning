#!/usr/bin/env bash
#SBATCH --job-name=redctl-cclprobe
#SBATCH --account=OTH21057
#SBATCH --partition=pvc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:12:00
#SBATCH --export=ALL
set -Eeuo pipefail
R=/scratch/08447/abd2000/redctl2026
L="$R/launch"
PROBE="$L/pvc_ccl_exact_probe.py"
PVC_VENV=/scratch/projects/compilers/intel23.1/intelpython/python3.9/envs/pytorch-gpu
PY="$PVC_VENV/bin/python"
set +u; module purge; module load intel/23.1; set -u
export LD_LIBRARY_PATH="$PVC_VENV/lib/python3.9/site-packages/torch/lib:$PVC_VENV/lib/python3.9/site-packages/intel_extension_for_pytorch/lib:$PVC_VENV/lib:${LD_LIBRARY_PATH:-}"
export CCL_LOG_LEVEL=warn CCL_ATL_TRANSPORT=ofi ZE_FLAT_DEVICE_HIERARCHY=FLAT FI_PROVIDER=tcp FI_TCP_IFACE=ibp56s0 FI_LOG_LEVEL=warn
export CCL_ALLGATHERV=naive CCL_BARRIER=ring CCL_BCAST=naive CCL_REDUCE_SCATTER=ring
OUT="$R/runs_probe/cclprobe-$SLURM_JOB_ID"; mkdir -p "$OUT/tiles"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((31700 + SLURM_JOB_ID % 500))
srun --exclusive --nodes=1 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=4 --cpu-bind=cores \
  --kill-on-bad-exit=1 --wait=0 --output="$OUT/affinity.%t.out" --error="$OUT/affinity.%t.err" \
  bash "$L/pvc_rank_launch.sh" "$PY" "$PROBE" --mode tile --output-dir "$OUT/tiles"
export CCL_WORKER_AFFINITY="$("$PY" "$PROBE" --mode affinity --output-dir "$OUT/tiles")"
echo "affinity=$CCL_WORKER_AFFINITY" >"$OUT/summary.txt"
step=0
for alg in ring direct rabenseifner unset; do
  step=$((step+1)); export MASTER_PORT=$((33000 + SLURM_JOB_ID % 500 + step))
  if [ "$alg" = unset ]; then unset CCL_ALLREDUCE; else export CCL_ALLREDUCE="$alg"; fi
  srun --exclusive --nodes=1 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=4 --cpu-bind=none --time=00:03:00 \
    --kill-on-bad-exit=1 --wait=0 --output="$OUT/probe-$alg.%t.out" --error="$OUT/probe-$alg.%t.err" \
    bash "$L/pvc_rank_launch.sh" "$PY" "$R/ccl_dtype_probe.py" || echo "alg=$alg step failed" >>"$OUT/summary.txt"
  grep -h PROBE_RESULT "$OUT"/probe-$alg.0.out >>"$OUT/summary.txt" 2>/dev/null || true
done
cat "$OUT/summary.txt"
