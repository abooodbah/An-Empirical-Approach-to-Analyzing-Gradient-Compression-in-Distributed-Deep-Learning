#!/usr/bin/env bash
# One Slurm controller task per H100 node; torchrun creates four local workers.

set -Eeuo pipefail

: "${SLURM_PROCID:?SLURM_PROCID is required}"
: "${SLURM_JOB_NUM_NODES:?SLURM_JOB_NUM_NODES is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"
: "${CONDITION_ID:?CONDITION_ID is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
: "${H100_VENV:?H100_VENV is required}"
: "${PAPER_RDMA_ROOT:?PAPER_RDMA_ROOT is required}"
: "${RDMA_RECEIPT_DIR:?RDMA_RECEIPT_DIR is required}"
: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"

runtime="${CONTAINER_RUNTIME:-}"
if [[ -z "$runtime" ]]; then
  runtime="$(command -v apptainer || command -v singularity || true)"
fi
[[ -x "$runtime" ]] || { printf 'FATAL: apptainer/singularity is unavailable\n' >&2; exit 1; }
[[ -r "$CONTAINER_IMAGE" ]] || { printf 'FATAL: missing container: %s\n' "$CONTAINER_IMAGE" >&2; exit 1; }
[[ -L "$H100_VENV/bin/python" || -x "$H100_VENV/bin/python" ]] || {
  printf 'FATAL: missing venv Python link/executable: %s\n' "$H100_VENV/bin/python" >&2
  exit 1
}
[[ -d "$RDMA_RECEIPT_DIR" ]] || {
  printf 'FATAL: missing condition-local RDMA receipt directory: %s\n' "$RDMA_RECEIPT_DIR" >&2
  exit 1
}
[[ -r "$PAPER_RDMA_ROOT/launch/rdma_sidecar.sh" ]] || {
  printf 'FATAL: missing vendored RDMA sidecar helper: %s\n' "$PAPER_RDMA_ROOT" >&2
  exit 1
}

# rdma_sidecar.sh deliberately resolves every bridge dependency relative to
# SOURCE_ROOT.  Point it only at the immutable, self-contained paper release.
SOURCE_ROOT="$PAPER_RDMA_ROOT"
# shellcheck source=../rdma_runtime/launch/rdma_sidecar.sh
source "$PAPER_RDMA_ROOT/launch/rdma_sidecar.sh"
prepare_acm_rdma_sidecar "$RDMA_RECEIPT_DIR"

binds="$ACM_RDMA_CONTAINER_BINDS"
for required_path in "$PROJECT_ROOT" "$RUN_ROOT"; do
  case "$required_path/" in
    "${WORK:?WORK is required}/"*|"${SCRATCH:?SCRATCH is required}/"*) ;;
    *) binds+=",${required_path}:${required_path}" ;;
  esac
done

rdma_env=(
  "ACM_RDMA_SIDECAR_ROOT=${ACM_RDMA_SIDECAR_ROOT}"
  "ACM_RDMA_PROVIDER_LAYOUT=${ACM_RDMA_PROVIDER_LAYOUT}"
  "ACM_RDMA_PROVIDER_LAYOUT_SHA256=${ACM_RDMA_PROVIDER_LAYOUT_SHA256}"
  "ACM_RDMA_LAYOUT_DIGEST_SHA256=${ACM_RDMA_LAYOUT_DIGEST_SHA256}"
  "IBV_DRIVERS=${ACM_RDMA_DRIVER_PATH}"
  "IBV_SHOW_WARNINGS=1"
  "LD_LIBRARY_PATH=${ACM_RDMA_LIBRARY_PATH}"
  "NCCL_IB_DISABLE=0"
  "NCCL_IB_HCA==mlx5_0:1"
  "NCCL_SOCKET_IFNAME==ibp92s0"
  "NCCL_NET_GDR_LEVEL=LOC"
  "PYTHONNOUSERSITE=1"
)

# Every node must authenticate and open mlx5_0 inside the exact container
# before this condition is allowed to launch any benchmark rank.
"$runtime" exec --cleanenv --nv --bind "$binds" "$CONTAINER_IMAGE" \
  env "${rdma_env[@]}" \
  "$H100_VENV/bin/python" "$PAPER_RDMA_ROOT/rdma_container_preflight.py" \
  --output-dir "$RDMA_RECEIPT_DIR" \
  --bridge-manifest "$ACM_RDMA_BRIDGE_MANIFEST" \
  --expected-hca mlx5_0

exec "$runtime" exec --cleanenv --nv --bind "$binds" "$CONTAINER_IMAGE" \
  env "${rdma_env[@]}" \
  NCCL_DEBUG=INFO \
  NCCL_DEBUG_SUBSYS=INIT,NET,ENV \
  NCCL_DEBUG_FILE="$RUN_DIR/logs/nccl.${CONDITION_ID}.%h.%p.log" \
  TORCH_DISTRIBUTED_DEBUG=DETAIL \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  OMP_NUM_THREADS=4 \
  "$H100_VENV/bin/python" -m torch.distributed.run \
  --nnodes="$SLURM_JOB_NUM_NODES" \
  --nproc-per-node=4 \
  --node-rank="$SLURM_PROCID" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv-id="${SLURM_JOB_ID}-${CONDITION_ID}" \
  --max-restarts=0 \
  "$@"
