# Software identity per track

Every per-rank provenance JSON in the raw archives embeds the same
information for its own run; this file is a convenience summary.

## H100 2x4 track (July campaign and September control)

- Python 3.11.9, PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121,
  numpy 1.26.4, CUDA runtime 12.1 (NCCL as bundled by PyTorch 2.4.1).
- Container: pytorch_2.4.1-cuda12.1-cudnn9-runtime.sif (sha256 recorded
  in each run's provenance/container.sha256).
- Host kernel: Linux 5.14.0-611.24.1.el9_7 (Stampede3 H100 nodes).
- Transport: NCCL over NDR InfiniBand (mlx5_0), host-staged
  (NCCL_NET_GDR_LEVEL=LOC); NCCL_SOCKET_IFNAME=ibp92s0. GPUDirect RDMA
  faulted on this node image; the per-condition transport validations
  in each run directory prove the host-staged NET/IB path.

## Intel Max 1550 track (July campaign and September control)

- TACC system stack: module intel/23.1 with the system conda
  environment /scratch/projects/compilers/intel23.1/intelpython/
  python3.9/envs/pytorch-gpu (Python 3.9; torch, intel-extension-for-
  pytorch, and oneccl_bind_pt versions embedded in every provenance
  JSON in the raw archives).
- Transport: oneCCL over OFI, FI_PROVIDER=tcp on the InfiniBand
  interface (ibp56s0); host-staged collective algorithms pinned
  (CCL_ALLREDUCE=ring, CCL_ALLGATHERV=naive).

## Single-node anchors (archived study)

- 8x RTX 3080 (PCIe 3.0, Xeon E5-2682 v4) and 8x H100 SXM (NVLink,
  EPYC 9534); versions in the archived study's own records.
