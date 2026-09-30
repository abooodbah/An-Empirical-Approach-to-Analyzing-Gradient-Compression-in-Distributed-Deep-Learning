#!/bin/bash
#SBATCH --job-name=redctl-sens
#SBATCH --account=OTH21057
#SBATCH --partition=gpu-a100-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
set -Eeuo pipefail
RED_ROOT="${RED_ROOT:?set RED_ROOT}"
VENV="${RED_LS6_VENV:-/scratch/08447/abd2000/redctl-ls6/venv}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cuda_ok() { "$1" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count()>0 else 1)' >/dev/null 2>&1; }
ENV_CHOICE=""
module load cuda/12.8 2>/dev/null; module load python3/3.9.7 2>/dev/null
if cuda_ok "$VENV/bin/python"; then ENV_CHOICE="redctl-venv+cuda12.8-module"; fi
if [[ -z "$ENV_CHOICE" ]]; then module unload cuda/12.8 2>/dev/null; if cuda_ok "$VENV/bin/python"; then ENV_CHOICE="redctl-venv-no-cuda-module"; fi; fi
[[ -n "$ENV_CHOICE" ]] || { echo "FATAL: no environment exposes CUDA on $(hostname)" >&2; exit 1; }
source "$VENV/bin/activate"
OUT="$RED_ROOT/runs/rev-sens-1n-${SLURM_JOB_ID}"
mkdir -p "$OUT"
{ hostname; nvidia-smi -L; nproc; echo "env_choice=$ENV_CHOICE";
  python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.get_num_threads())'; } >"$OUT/node.txt" 2>&1
sha256sum "$RED_ROOT/control_workload_benchmark_v5_1.py" "$RED_ROOT/codec_sensitivity.py" >"$OUT/sha256.txt"
python "$RED_ROOT/codec_sensitivity.py" run "$RED_ROOT/control_workload_benchmark_v5_1.py" "$OUT/sens.csv" cuda
python "$RED_ROOT/codec_sensitivity.py" analyze "$OUT/sens.csv" >"$OUT/sens_report.txt"
