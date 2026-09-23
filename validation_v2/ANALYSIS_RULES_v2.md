# TPDS v2 validation campaign: analysis rules (frozen before any job is submitted)

Scope: jobs V1, V2, V3 (Experiments A and B; 12 ranks on Lonestar6 A100 over GPUDirect RDMA,
NCCL_NET_GDR_LEVEL=SYS), the optional L1 (6 ranks, 1.05B model), and the SMOKE job. This file's
SHA-256 is committed before any of these jobs is submitted. The E6 held-out evaluation of
2026-09-21 is neither rescored nor revised; this campaign is a new, prospectively specified
follow-up to the limitation it found.

## Execution
- All five jobs are submitted at the same time, at the author's request, each with
  --nice=10000 so that MOSAIC jobs keep priority. They may run concurrently and share the
  fabric; every job records its hosts and the start and end time of every run.
- Settings are fixed now: local batch 32 (Experiment A) and 32 and 128 (Experiment B); 20
  warm-up and 100 measured iterations; input pool 4; two counterbalanced passes (pass 2 is
  pass 1 reversed); L1 at sequence 512, local batch 1. Seeds: V1 2026092301, V2 2026092302,
  V3 2026092303, L1 2026092311, SMOKE 2026092399.
- The SMOKE job is a verification record that runs concurrently, not a gate. Contingencies
  (each reported as such, with its cost):
  - a batch-128 run that exhausts memory: that workload is rerun at batch 64 for every
    method in three further allocations, and batch 64 replaces 128 for that workload;
  - a synchronizing call found inside a B block other than the Top-k size exchange (which the
    algorithm needs): the affected B runs are excluded, the harness is fixed, and those
    conditions are rerun in further allocations;
  - L1 failing for memory at sequence 512: L1 is rerun once at sequence 256, else dropped.

## Quantities
- Replication unit: the allocation (job). Ranks and iterations are not replications.
- Run value T: maximum over ranks of the rank-local elapsed time of the measured block of 100
  iterations, divided by 100 (block_step_equiv_s), in every convention.
- Condition value in a job: the mean of its two passes. A pass gap above 0.05 of the mean is
  reported beside the value. If one pass is missing (failure or the job's time limit), the
  remaining pass is the value and is marked as a single pass.
- Conventions: E = Tier 1 (phase), S = Tier 2 (wall), B = block reference.
- Perturbation: rho = T^E / T^B and T^S / T^B, per path, workload, and job.
- Matched gain under convention q: g^q = 1 - T_reduce^q / T_gather^q (shared-scale int8 arms).
- Bundled gain: h^q = 1 - T_reduce^q / T_topk^q (Top-k to the shared-scale int8 all-reduce;
  encoder, payload, and collective change together; read as an implementation substitution).
- Hook ratio (Experiment B, convention B): R_h(b) = T_h(b) / T_dense(b), dense = FP32 DDP
  (ddp-overlap), h in {16-bit hook, int8 hook}, b in {32, 128}; change D_h = R_h(128) - R_h(32).

## Margin
0.05 throughout. Magnitude discrepancies are reported in percentage points.

## Verdicts
Experiment A, per workload, for q in {E, S} against B, applied to g and to h:
- Direction, per job, in four categories:
  - same practical direction: same sign, both |g^q| and |g^B| at least 0.05;
  - reversed: opposite signs, both at least 0.05;
  - crosses the 5% margin: exactly one of |g^q|, |g^B| is at least 0.05 (whatever the signs);
  - no practical difference under either: both below 0.05.
  Only "reversed" is described as a change of conclusion. A crossing of the margin is
  described as the conventions disagreeing about whether the difference is practically
  large, not as a reversal.
- Magnitude, per job: preserved if |g^q - g^B| <= 0.05.
- Ranking of the three paths (ties within 0.02 relative) identical to B's.
- Workload-level words need the same outcome in all three jobs; otherwise the result is
  "allocation-dependent", with the per-job outcomes listed.
- rho "differs between paths" when, in every job, max rho - min rho over the paths exceeds
  0.05 within the workload.

Experiment B, per workload and hook:
- "Stable with computation" if |D_h| < 0.05 in all three jobs; "changes with computation" if
  |D_h| >= 0.05 with one sign in all three; otherwise "inconsistent across allocations".
- Faster (slower) than dense DDP at a batch if R_h <= 0.95 (>= 1.05) in all three jobs;
  "within 5%" if |R_h - 1| < 0.05 in all three; otherwise inconsistent.
- The larger batch also changes kernels and gradient values: this is a compute-to-communication
  sensitivity test, not an isolated manipulation of overlap.

## Checks that do not decide verdicts
- Drift: dense anchor at the start and end of each job; |T_end / T_start - 1| > 0.05 flags
  the job. Flagged jobs are reported, not excluded.
- Warm-up adequacy: in every timed E and S run, the mean per-iteration step (maximum over
  ranks) of measured iterations 81-100 against 1-20; a difference above 0.05 flags the run.
- Calibration at the submitted payload sizes (start and end of each job) is reported beside
  the E exchange times; it characterizes collectives only and is never added into step times.
- Traces (V1 only) are read for overlap and completion waits on ranks 0 and 3; their times are
  never used as performance numbers.
- Missing measurements stay missing; no exchange phase is inferred for an overlapped hook.

## Exclusions
Only runs failing the harness's own validation (status not PASS, a rank missing, parameter
hashes differing, non-finite loss, a measurement synchronization counted inside a B block) are
excluded, and each is listed. No condition is rerun inside a job; no job is repeated to replace
an unfavorable one. A job that stops early is reported as partial.

## L1
One allocation. Reported descriptively (T under S and B for dense DDP and both hooks, the hook
ratios, and T^S / T^B) with the words "in this allocation"; no replication claim.

## Not claimed
No tail percentiles from block timing; nothing about accuracy or convergence; no generality
beyond these workloads, this transport, and these rank counts; no significance tests or
confidence intervals from three allocations; the old diagnostic rules are not presented as
validated by this campaign.
