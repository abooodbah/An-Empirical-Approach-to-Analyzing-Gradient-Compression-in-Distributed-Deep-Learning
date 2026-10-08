# Platform facts from the archived run provenance

Collected by `environment/platform_facts.py`. A key with several values varied
between runs. Hostnames and per-rank identifiers are left out.

## H100 2x4 (20 records)

Read from the run directory `h100-3328658`.

| key | values |
|---|---|
| `accelerator.backend` | nccl |
| `accelerator.ccl_binding` | None |
| `accelerator.device` | cuda:0 |
| `accelerator.device_name` | NVIDIA H100 |
| `accelerator.kind` | cuda |
| `accelerator.visible_device_count` | 4 |
| `allowlisted_environment.NCCL_DEBUG` | INFO |
| `allowlisted_environment.NCCL_IB_DISABLE` | 0 |
| `allowlisted_environment.NCCL_IB_HCA` | =mlx5_0:1 |
| `allowlisted_environment.NCCL_NET_GDR_LEVEL` | LOC |
| `allowlisted_environment.NCCL_SOCKET_IFNAME` | =ibp92s0 |
| `collective_preflight.expected_float32_sum` | 36.0 |
| `collective_preflight.float32_allreduce` | pass |
| `collective_preflight.int64_allgather` | pass |
| `collective_preflight.uint8_allgather` | pass |
| `host_layout.nodes` | 2 |
| `host_layout.ranks_per_node` | 4 |
| `libraries.cuda_runtime_reported_by_torch` | 12.1 |
| `libraries.intel_extension_for_pytorch` | None |
| `libraries.numpy` | 1.26.4 |
| `libraries.oneccl_bind_pt` | None |
| `libraries.python` | 3.11.9 |
| `libraries.torch` | 2.4.1+cu121 |
| `libraries.torchvision` | 0.19.1+cu121 |
| `platform` | Linux-5.14.0-611.24.1.el9_7.x86_64-x86_64-with-glibc2.35 |

## Lonestar6 A100 (10 records)

Read from the run directory `ls6-triad4-4n-3457305`.

| key | values |
|---|---|
| `accelerator.backend` | nccl |
| `accelerator.ccl_binding` | None |
| `accelerator.device` | cuda:0 |
| `accelerator.device_name` | NVIDIA A100-PCIE-40GB |
| `accelerator.kind` | cuda |
| `accelerator.visible_device_count` | 3 |
| `allowlisted_environment.FI_PROVIDER` | mlx |
| `allowlisted_environment.NCCL_DEBUG` | INFO |
| `allowlisted_environment.NCCL_NET_GDR_LEVEL` | SYS |
| `collective_preflight.expected_float32_sum` | 78.0 |
| `collective_preflight.float32_allreduce` | pass |
| `collective_preflight.int64_allgather` | pass |
| `collective_preflight.uint8_allgather` | pass |
| `host_layout.nodes` | 4 |
| `host_layout.ranks_per_node` | 3 |
| `libraries.cuda_runtime_reported_by_torch` | 12.1 |
| `libraries.intel_extension_for_pytorch` | None |
| `libraries.numpy` | 2.0.2 |
| `libraries.oneccl_bind_pt` | None |
| `libraries.python` | 3.9.7 |
| `libraries.torch` | 2.4.1+cu121 |
| `libraries.torchvision` | 0.19.1+cu121 |
| `platform` | Linux-4.18.0-553.150.1.el8_10.x86_64-x86_64-with-glibc2.28 |

## Intel Max 1550 (25 records)

Read from the run directory `redctl-A-1n-3514993`.

| key | values |
|---|---|
| `accelerator.backend` | ccl |
| `accelerator.ccl_binding.module_file` | /scratch/projects/compilers/intel23.1/intelpython/python3.9/envs/pytorch-gpu/lib/python3.9/site-packages/oneccl_bindings_for_pytorch/__init__.py |
| `accelerator.ccl_binding.version` | 1.13.100+gpu |
| `accelerator.device` | xpu:0 |
| `accelerator.device_name` | Intel(R) Data Center GPU Max 1550 |
| `accelerator.kind` | xpu |
| `accelerator.visible_device_count` | 8 |
| `allowlisted_environment.CCL_ATL_TRANSPORT` | ofi |
| `allowlisted_environment.CCL_LOG_LEVEL` | info |
| `allowlisted_environment.CCL_WORKER_COUNT` | 1 |
| `allowlisted_environment.FI_LOG_LEVEL` | warn |
| `allowlisted_environment.FI_PROVIDER` | tcp |
| `collective_preflight.expected_float32_sum` | 36.0 |
| `collective_preflight.float32_allreduce` | pass |
| `collective_preflight.int64_allgather` | pass |
| `collective_preflight.uint8_allgather` | pass |
| `host_layout.nodes` | 1 |
| `host_layout.ranks_per_node` | 8 |
| `libraries.cuda_runtime_reported_by_torch` | None |
| `libraries.intel_extension_for_pytorch` | 1.13.10+xpu |
| `libraries.numpy` | 1.22.3 |
| `libraries.oneccl_bind_pt` | 1.13.100+gpu |
| `libraries.python` | 3.9.15 |
| `libraries.torch` | 1.13.0a0+gitb1dde16 |
| `libraries.torchvision` | 0.14.1a0+0504df5 |
| `platform` | Linux-5.14.0-611.24.1.el9_7.x86_64-x86_64-with-glibc2.34 |

## Lonestar6 A100, validation (16 records)

Read from the run directory `v2-L1-2n-3465529`.

| key | values |
|---|---|
| `accelerator.backend` | nccl |
| `accelerator.ccl_binding` | None |
| `accelerator.device` | cuda:0 |
| `accelerator.device_name` | NVIDIA A100-PCIE-40GB |
| `accelerator.kind` | cuda |
| `accelerator.visible_device_count` | 3 |
| `allowlisted_environment.FI_PROVIDER` | mlx |
| `allowlisted_environment.NCCL_DEBUG` | INFO |
| `allowlisted_environment.NCCL_NET_GDR_LEVEL` | SYS |
| `collective_preflight.expected_float32_sum` | 21.0 |
| `collective_preflight.float32_allreduce` | pass |
| `collective_preflight.int64_allgather` | pass |
| `collective_preflight.uint8_allgather` | pass |
| `host_layout.nodes` | 2 |
| `host_layout.ranks_per_node` | 3 |
| `libraries.cuda_runtime_reported_by_torch` | 12.1 |
| `libraries.intel_extension_for_pytorch` | None |
| `libraries.numpy` | 2.0.2 |
| `libraries.oneccl_bind_pt` | None |
| `libraries.python` | 3.9.7 |
| `libraries.torch` | 2.4.1+cu121 |
| `libraries.torchvision` | 0.19.1+cu121 |
| `platform` | Linux-4.18.0-553.150.1.el8_10.x86_64-x86_64-with-glibc2.28 |

## Not recorded before October 2026

- GPU driver version
- GPU and adapter firmware
- PCIe and NUMA placement of GPUs and adapters
- CPU model and core count where the platform block lacks them

The Lonestar6 jobs of Revision 4 (E9) record these in `provenance/node_facts.*.txt`.
The Stampede3 jobs (E10) do not, so they stay unrecorded for the Intel system.
