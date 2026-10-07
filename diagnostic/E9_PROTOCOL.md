# E9: reducible quantizers and a matched DDP baseline (pre-registered)

Fixed on 2026-10-07, before any E9 job was submitted. The commit and the
push time are in provenance/PROVENANCE.md.

## 1. Questions
Q1. What does a summable quantizer that could be trained with cost in step
time, against dense FP32, dense FP16, and the rank-local int8 gather?
Q2. How do PyTorch's two compression hooks compare with dense DDP that uses
the same reducer, bucket size, and synchronization in the same allocation?
Q3. How large is each encoder's gradient error?
Q4 (the probes, exploratory). Does the two-level traffic model explain the
all-gather to all-reduce slope ratio, and does a fit to 64 MiB predict
larger payloads?

## 2. Platform and protocol
Lonestar6 gpu-a100, three A100 per node, NCCL_NET_GDR_LEVEL=SYS. Harness
v5.2 (validation_v2/CODE_v52.sha256). Four allocations: q12a and q12b
(4 nodes, 12 ranks), q24a and q24b (8 nodes, 24 ranks). Workloads
transformer and resnet50, batch 32.

Outcome runs: validation profile, block timing, 100 measured and 20 warm-up
iterations, input pool 4, two passes, the second in reverse order. The
outcome of a condition is the mean of block_step_equiv_s over its passes.
Modes: native, fp16-native, ddp-overlap, ddp-fp16-hook, ddp-int8-hook,
gpu-quant, gpu-quant-reducible, gpu-quant-sr-int8, gpu-quant-sr-fp16.

In the "a" allocation of each scale only: fidelity runs (qualification
profile, 8 measured iterations, --verify-fidelity) for fp16-native,
gpu-quant, gpu-quant-reducible, gpu-quant-sr-int8, gpu-quant-sr-fp16;
phase-timed runs (extension profile, 40 measured and 10 warm-up iterations)
for the six modes that are not DDP modes; the host-staged probes.

In every allocation: a wire-type check, three probes (default algorithm;
NCCL_ALGO=Ring; NCCL_ALGO=Tree for all-reduce only), and an equivalence run
of each stochastic quantizer.

## 3. Outcomes and how they are reported
R(m, b) = T(m) / T(b) for each baseline b in native, fp16-native,
ddp-overlap, per allocation. Nothing is pooled across scales or workloads.
Both allocations' values are always printed. A comparison is called
"faster" if R <= 0.95 in both allocations of the scale, "slower" if
R >= 1.05 in both, "within 5%" if 0.95 < R < 1.05 in both, and
"allocation-dependent" otherwise.
Fidelity: mean relative L2 error and mean cosine over the measured
iterations of rank 0. Exchange time: mean network_s pooled over ranks and
iterations of the phase-timed run.

## 4. Validity
An outcome run counts if its status is PASS and final_param_hashes_equal is
True. An equivalence run must record equivalence_ok True. A condition with
one valid pass is reported from that pass and marked. A condition with none
is reported as missing. Nothing is rerun. A job that completes no outcome
run may be resubmitted once.
Q2 is a matched comparison only if the DDP bucket structure recorded in the
provenance is identical for ddp-overlap and both hooks in the allocation.
Otherwise the ratios are reported and labeled unmatched.

## 5. Expectations, stated now and not used in scoring
X1. gpu-quant-sr-int8 is within 10% of gpu-quant-reducible in step time.
X2. gpu-quant-sr-fp16 is not faster than fp16-native.
X3. ddp-int8-hook is slower than ddp-overlap at both scales, by more at 24
ranks.
X4. ddp-fp16-hook is within 5% of ddp-overlap or faster.
X5. The relative error of gpu-quant-sr-fp16 is below that of
gpu-quant-sr-int8.

## 6. Probes (exploratory, reported whatever they show)
Payloads from 4 B to 1 GiB; a collective is skipped where its buffers
would exceed 8 GiB on a rank. The slope of a collective is the
least-squares line through its points from 1 MiB to 64 MiB.
P1. With all-reduce forced to Ring, slope(all-gather) / slope(all-reduce)
is within a factor 1.5 of p/2.
P2. The default all-reduce slope is not more than 5% above the Ring slope.
P3. The 1 MiB to 64 MiB line predicts the 256 MiB and 1 GiB times within
15%.
If P1 fails, the paper says that the traffic model does not explain the
slope ratio on this system.

## 7. Frozen components
validation_v2/analysis/score_e9.py and rev4_records.py, harness v5.2,
harness/launch/ls6_e9.slurm and e9_sets.sh. Hashes: diagnostic/E9_FREEZE.sha256.
