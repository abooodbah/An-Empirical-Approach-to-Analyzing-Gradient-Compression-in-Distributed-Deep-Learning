# TACC paper-workload experiment contract

Contract version: `2.0.0`  
Amendment: `2026-07-17-a`  
Campaign identifier: `tpds-legacy-continuity-tacc-2026-07`  
Original freeze: 2026-07-17 (America/Chicago)  
Amended: 2026-07-17 (America/Chicago)

## Purpose and claim boundary

This campaign extends the measurements reported in the rejected TPDS
manuscript without silently changing the original synthetic workloads. The
executing matrix has two independently reported platform tracks:

1. **H100 world-size-matched continuity:** compare the archived paper's
   single-node `1 x 8` H100 observation with a new TACC `2 x 4` H100
   observation at the same world size and per-rank work.
2. **Intel PVC weak scaling:** run the same ResNet-50 and Transformer workload
   matrix on `1 -> 2 -> 3 -> 4` PVC nodes (`8 -> 16 -> 24 -> 32` ranks) using
   the qualified Intel host-staged oneCCL/OFI/TCP path.

These tracks answer different questions and must never be pooled. In
particular, `causal_fabric_contrast=false`: the archived and TACC H100 systems
differ in host platform, accelerator topology/form factor, software, clocks,
and system state, while PVC uses a different accelerator and communication
stack. The phrase "fabric-only effect" is prohibited.

The archived `1 x 8` H100 data are descriptive-only and noncausal. They contain
rank-0 measurements only and do not contain the complete seed, software,
topology, and per-rank provenance required by this contract. They are not an
independent replicate of any new TACC allocation and must not be used in pooled
confidence intervals or significance tests.

A real-dataset validation track (for example Food-101/ResNet-50 or
WikiText-103/GPT-2 Medium with overlapped DDP) remains useful future work but is
outside this executing amendment and its SU envelope. Such results, if later
authorized, must not be pooled with this legacy-continuity matrix.

## Frozen legacy workload

The workload definition is inherited from `5.9/workspace/benchmark.py` and
`5.9/workspace/run_all_experiments.sh`, with the corrected, provenance-complete
implementation in `experiments/tacc_paper_workloads/paper_workload_benchmark.py`.
Two track-specific frozen sources are authorized:

- Intel PVC uses schema `tacc-paper-workloads/v2`, SHA-256
  `26f884cc32d366441b248386d5afaf873adf9c289ff0c1f1091689514df1be29`,
  including hash-bound calibration references.
- The already queued H100 qualification and three full jobs use schema
  `tacc-paper-workloads/v1`, SHA-256
  `27a81ace30e103da19d905a684a30744ccc9ad1df9497e5f0ba021f68a1674ee`.
  V1 is accepted only for CUDA/NCCL H100 evidence and every paper condition
  must retain finite direct all-reduce and all-gather alpha/beta calibration;
  it has no calibration-reference semantics.

Schemas or hashes may not be mixed within a platform/seed stratum. V1 is not
valid for PVC, and v2 calibration references may not be inferred for v1.

| Field | ResNet-50 | Transformer |
|---|---|---|
| Model | uninitialized torchvision ResNet-50, 1,000 outputs | custom 2-layer encoder, `d_model=512`, 8 heads, vocabulary 10,000, binary classifier |
| Input | random FP32 `[32,3,224,224]` and random 1,000-class labels | random integer `[32,64]` tokens and random binary labels |
| Local batch | 32 per rank | 32 per rank |
| Precision | FP32 | FP32 |
| Optimizer | SGD, learning rate 0.01 | SGD, learning rate 0.01 |
| Warmup/measured | 10 warmup and exactly 40 measured iterations | 10 warmup and exactly 40 measured iterations |

Initial model parameters and random inputs are deterministic per seed. Rank 0
broadcasts initialized parameters before measurement. Missing ResNet,
torchvision, IPEX, XPU, CUDA, or collective support is a hard failure; model,
device, or backend substitution is forbidden.

The frozen mode matrix is:

- `native`: manual flattened blocking all-reduce, `sync_every={1,2,4}`.
- `gpu_topk`: 1% Top-k, `sync_every={1,2,4}`, preserving the legacy residual
  update against the global averaged reconstruction. This is labeled
  `legacy_global_reconstruction_residual`, not standard local error feedback.
- `gpu_quant`: legacy 8-bit accelerator-resident quantization,
  `sync_every=1`.
- `cpu_quant`: legacy CPU conversion/serialization path, `sync_every=1`.
- `powersgd`: official PyTorch DDP PowerSGD hook with legacy rank 1 and hook
  settings, `sync_every=1`.
- `ddp_overlap`: separately reported normal DDP baseline with no explicit
  isolation barrier. It is not included in legacy isolated-timing ratios.

The legacy TP and PP scripts are synthetic communication proxies, not model
parallel implementations. If run separately, they must be labeled
`tp_traffic_proxy` and `pp_traffic_proxy` and shown only as supplemental data.

## Executing hardware and replication matrix

### H100 continuity test

- Archived reference: one node, eight H100 ranks, local batch 32, world size 8;
  descriptive-only, rank-0-only, incomplete provenance, and non-independent.
- New TACC observation: two `gh` nodes with four H100 ranks per node, local
  batch 32, world size 8, NCCL backend.
- Three separately submitted full jobs use seeds `2026010043`, `2026011043`,
  and `2026012043`.
- Each full job has a hard ceiling of 15 minutes and 2 SUs. The three-job full
  H100 phase therefore has a maximum authorized exposure of 6 SUs.
- Primary reported quantities are distributed critical-path latency, phase
  fractions, and variability at identical world size/per-rank work.
- `causal_fabric_contrast=false`; the archived baseline is never pooled with
  the three new jobs.

### Intel PVC weak-scaling test

- A single four-node allocation runs ordered nested steps at `1,2,3,4` nodes,
  with eight ranks per node and world sizes `8,16,24,32`.
- Local batch remains 32, so global batches are `256,512,768,1024`. This is
  weak scaling and must not be described as fixed-global-batch strong scaling.
- The executing descriptive campaign uses seed `2026010043` in one allocation.
  Its nested node-count steps are not independent allocations; no inferential
  claim or p-value is permitted.
- The full PVC allocation has a hard ceiling of 120 minutes and 24 SUs.
- Required stack: XPU-enabled PyTorch 1.13, matching IPEX/torchvision, and
  `oneccl_bind_pt-1.13.100+gpu` under TACC Intel 23.1.
- Qualified communication path: `oneccl_host_staged_ofi_tcp`, with
  `FI_PROVIDER=tcp` and `FI_TCP_IFACE=ibp56s0`.
- This path is explicitly host-staged. It provides no evidence for PVC
  device-direct collectives, native GPU-fabric transport, PSM2, RDMA, or an
  InfiniBand device-direct data path. Those claims are prohibited.
- PVC-versus-H100 comparisons are cross-stack descriptive comparisons only.
  `causal_fabric_contrast=false`.

## Network calibration-reference contract

The paper profile performs exactly one alpha/beta network calibration source
per PVC world size, using the `resnet50/native/sync_every=1` cell for seed
`2026010043`. Each other cell at that same world size must carry the source run
ID in `calibration_reference` and may then use `--skip-calibration`.

A referenced cell is valid only if all of the following hold:

1. Its referenced source completed successfully and retained raw calibration
   samples and validation metadata.
2. Source and consumer match campaign, platform, backend, provider, interface,
   seed, world size, node/rank layout, benchmark SHA-256, and software stack.
3. References never cross world sizes or platforms.
4. Exactly one source exists for each of world sizes `8,16,24,32`; a missing,
   duplicate, failed, or ambiguous source invalidates every dependent cell.
5. `--skip-calibration` without a nonempty valid reference is forbidden for a
   paper-profile cell. Qualification-profile probes are the only exception.

Calibration reuse reduces duplicate setup work only. It does not change the
models, modes, synchronization cadences, warmups, 40 measured iterations, or
per-rank metric collection.

## Qualification gates

No full campaign may start until its relevant platform gate passes.

H100 qualification (maximum 15 minutes, 2 SUs):

1. Four distinct H100 devices per host and eight distinct rank/device mappings.
2. Two-host exact-value NCCL all-reduce.
3. NCCL logs identify the selected inter-node transport and HCA/interface; any
   observed fallback must be retained and disclosed.
4. One deterministic distributed optimizer step produces matching parameter
   hashes.

PVC qualification (maximum 15 minutes on one node, 0.75 SU):

1. TACC Intel 23.1 `pytorch-gpu` stack loads torch, IPEX, torchvision, and
   `oneccl_bind_pt-1.13.100+gpu` with no unresolved shared libraries.
2. Eight flat PVC tiles are uniquely mapped and each completes synchronized
   matrix multiplication, Top-k, byte packing, ResNet forward/backward, and
   Transformer forward/backward.
3. Exact-value CCL collective probes and one deterministic distributed
   optimizer step pass on the qualified host-staged stack.

Separate four-node PVC smoke gate (maximum 15 minutes, 3 SUs):

1. World sizes `8,16,24,32` execute in order across the allocated nodes.
2. Rank/device and CPU-affinity mappings are unique and recorded.
3. Collectives use `FI_PROVIDER=tcp`, `FI_TCP_IFACE=ibp56s0`, and the labeled
   `oneccl_host_staged_ofi_tcp` path.
4. No result from this gate may be represented as device-direct, RDMA, or
   native-fabric evidence.

## Measurement and statistics

Accelerator synchronization brackets every isolated timing interval. Each rank
records all 40 measured samples for compute, serialization/compression,
collective/network, synchronization/wait, reconstruction, and total iteration
time where applicable. The primary configuration statistic is the maximum rank
time for each iteration (the distributed critical path), summarized with mean,
median, sample standard deviation, coefficient of variation, P50, P90, P95,
P99, minimum, and maximum. Also report global examples/s, communication and
serialization fractions, speedup, and PVC weak-scaling efficiency.

The three new H100 jobs permit across-job descriptive variability summaries.
Any inferential test must treat allocations, not within-job iterations or
ranks, as independent units. The single-allocation PVC series supports
descriptive scaling estimates only. No p-value or population-level confidence
claim is allowed for PVC.

Measured and inferred times must be distinguished. In particular, a PowerSGD
communication estimate obtained by subtraction is labeled inferred and never
plotted as a directly timed collective.

## Provenance and invalidation

Each run retains Slurm job/allocation data, timestamps, hostnames,
rank-to-device maps, hardware topology, software/package versions, loaded
modules, transport environment, backend debug logs, command line, benchmark
and launcher SHA-256 values, seeds, raw per-rank samples, calibration source or
reference, return codes, and a machine-readable validation report.

A cell is invalid if any expected rank or measured sample is absent; any value
is non-finite; ranks disagree on model, seed, or world size; a dependency,
model, device, or backend fallback occurs; required transport evidence is
missing; initial or measured-start full-state hashes differ; the cell is
outside the frozen matrix; benchmark schema or SHA-256 differs from the
track-authorized source; calibration validation/reference validation fails;
or finalization is incomplete.
Final full-state hashes are retained but are not a universal equality gate:
ResNet-50 `state_dict()` includes rank-local BatchNorm running buffers, and
`sync_every={2,4}` deliberately permits local parameter updates between
communication steps. Any final inequality must therefore be disclosed and
interpreted by mode rather than silently treated as transport failure. Failed
and partial runs are retained as diagnostic evidence but never enter plots or
aggregate tables.

## SU budget and stop rules

- The active user-authorized cumulative ceiling for this extension is **100 SUs**.
  It includes successful, failed, canceled, retried, qualification,
  smoke, and full jobs.
- This 100-SU operational ceiling is stricter than 15% of the original
  854-SU allocation (`128.1` SUs). The lower 100-SU value governs.
- Reserved phase ceilings are:

  | Phase | Maximum exposure |
  |---|---:|
  | H100 qualification | 2 SUs |
  | H100 full jobs: 3 x 2 SUs | 6 SUs |
  | PVC one-node qualification | 0.75 SU |
  | PVC four-node smoke | 3 SUs |
  | PVC full `1 -> 2 -> 3 -> 4` nested allocation | 24 SUs |
  | **Current planned phase maximum** | **35.75 SUs** |

- Phase ceilings are maxima, not claims of actual accounting. Previously
  charged retries remain part of the cumulative 100-SU ledger and cannot be
  erased or reclassified.
- Before every submission, the launcher or operator ledger must calculate the
  new maximum exposure. Submission fails closed if the job would exceed either
  its phase ceiling or the remaining cumulative 100-SU authorization.
- Qualification precedes full measurement. A failed gate stops dependent jobs;
  a retry requires a diagnosed code/environment change and available budget.
- No additional PVC seed/allocation or real-workload campaign is implicitly
  authorized by unused headroom.
