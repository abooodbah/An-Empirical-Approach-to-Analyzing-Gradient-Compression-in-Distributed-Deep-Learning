# E8: the frozen rules against an overlap-aware reference (pre-registered)

This file, the launch set `harness/launch/e8_sets.sh`, and
`harness/launch/ls6_e8.slurm` were written and pushed before any E8 job was
submitted. The reference (`diagnostic/e8/replay_reference.py`), the scorer
(`diagnostic/e8/score_e8.py`), their tests, and the development report
(`diagnostic/e8/DEV_REPORT.md`) are committed and pushed before any E8
record is copied from Lonestar6; `diagnostic/e8/E8_FREEZE.md` then lists
their hashes, and `provenance/PROVENANCE.md` records the push time beside
the E8 jobs' end times. Nothing in these files changes after that push; a
later change is a new version, and E8 is then reported as retrospective.

The original plan ran E8 in a second submission window, after the
reference had been developed on the E7 traces. The author's deadline
(2026-09-30) allows about two windows of two to three days each, so the E8
jobs are submitted in the first window, beside E7. The measurement does
not depend on the reference; only the analysis does, and the analysis is
frozen before the records are read.

## 1. Question
Which analysis recommends a change that pays, how accurately does it
estimate the benefit, and how much measurement does it need? The methods
are the frozen rule sets (full procedure, A1, A2, A3; `diagnostic/rules.py`)
and an overlap-aware replay of the original's trace (the reference). The
reference adapts the dependency-graph replay of dPRO to the harness's
Kineto traces; it is not dPRO, which reads only its own traces.

## 2. Platform and measurements
Lonestar6 gpu-a100, A100-PCIe, three per node, NCCL with
NCCL_NET_GDR_LEVEL=SYS, as in E6 and E7. Two regimes, neither measured on
Lonestar6 before (E7 is p = 6 at batch 128; the 24-rank replication is
p = 24 at batch 32):

| Set | Nodes | p | Local batch |
|---|---|---|---|
| e8s | 2 | 6 | 32 |
| e8l | 8 | 24 | 128 |

Two allocations per regime (two jobs each). In each allocation, in this
order (26 runs):
1. preflight: harness v4, qualification profile, transformer native;
2. calibration: harness v4, paper profile, resnet50 native, batch 32, as
   in E7; every later run records it as its calibration reference;
3. M1, the inputs of the rules: the four originals (section 3), harness
   v4, extension profile, phase timing, the regime's batch, 40 iterations
   after 10 warm-up;
4. traces, the input of the reference: the four originals, harness v5.1,
   qualification profile, wall timing, the regime's batch, 8 iterations
   after 20 warm-up, input pool 4, ranks 0 and 3;
5. outcomes: the eight modes (the originals and the candidates), harness
   v5.1, validation profile, block timing, the regime's batch, 100
   iterations after 20 warm-up, input pool 4, two passes, pass 2 being
   pass 1 reversed.
Every method's input is measured before any outcome, and no method sees a
candidate's measurement.

## 3. Cases
Per regime and model (transformer, resnet50), eight cases in all, each
measured in two allocations:

| Family | Original | Candidate | Mechanism tested |
|---|---|---|---|
| G | gpu-quant-gathersum | gpu-quant-reducible | shape, identical encoding |
| H | ddp-int8-hook | ddp-fp16-hook | shape under overlap |

## 4. Outcome
Run value T: the largest over ranks of the measured block's elapsed time,
divided by 100 (`block_step_equiv_s`), as in the v2 analysis rules. The
value of a mode in an allocation is the mean of its two passes; if one pass
is missing, the other is used and marked single-pass. A pass gap above 0.05
of the mean is reported. The measured gain of a case in an allocation is
g = 1 - T(candidate) / T(original). A run failing the harness's own
validation (status not PASS, a rank missing, non-finite loss, a
synchronization inside the block) is missing. No condition is rerun.

## 5. Predictions
- Rule sets: the frozen rules on the allocation's M1 record of the original,
  read by the frozen extractor with the allocation's calibration, trend
  None, as in E7. A verdict of shape is a recommendation of the case's
  candidate, with effect e for the full procedure and none (e = 0) for the
  ablations. Placement and coordination verdicts are "not tested"; none,
  mixed, and insufficient are abstentions. Readings: frozen (primary) and
  described (C1 and C2 of `diagnostic/v1_1/rescore.py`), as in E7.
- Reference: g_pred = 1 - replay(candidate) / replay(original) from the
  original's trace on ranks 0 and 3 and the allocation's calibration, with
  the substitutions and scheduling of section 7. It recommends the
  candidate when g_pred >= 0.05, with effect g_pred.

## 6. Scoring
E6's rule with E7's labels, per case and allocation (each counts 1/2):
correct if g >= max(0.05, 0.5 e); short if 0.05 <= g < 0.5 e; null if
|g| < 0.05; harmful if g <= -0.05; an abstention is correct when g < 0.05
and missed otherwise; missing when a needed run failed. Reported per
method: recommendations, recommendations that cleared 5%, null or harmful
recommendations, abstentions (correct, missed), missing, the median
magnitude error |e - g| over the method's recommendations (full procedure
and reference only), and the measurement seconds the method needs (the
M1 run for the rules, plus the calibration run for A2 and the full
procedure; the trace run plus the calibration run plus the replay time for
the reference).

## 7. Reference algorithm (fixed now; the code is frozen before any E8 record is read)
1. From the original's Kineto trace on a rank, take the GPU kernels
   (stream, start, duration), the CPU-launch-to-kernel flows, and the
   cudaEventRecord / cudaStreamWaitEvent pairs that give the cross-stream
   dependencies.
2. For each traced iteration build a graph: kernels in stream order, plus
   the event edges, with each kernel released no earlier than its launch.
3. Substitute the candidate:
   - G: the all-gather kernel becomes an all-reduce of the same int8
     payload, costed alpha + beta x bytes from the allocation's
     calibration; the p-way local int32 sum becomes one decode, costed at
     the elementwise throughput measured in the same trace.
   - H: each bucket's int8 encode, all-gather, and decode become an fp16
     cast, an all-reduce of 2 x the bucket's int8 bytes, and a cast back,
     costed the same way.
4. Replay by list scheduling and take the makespan of each iteration; the
   prediction uses the median iteration and the larger of the two ranks.

## 8. Development data and gate G2 (a reporting gate)
Development data are configurations E8 does not measure: the V1 hook
traces (12 ranks, batch 32 and 128), with the V1 to V3 block-timed hook
ratios as ground truth, and the e7a traces (6 ranks, batch 128), with E7's
wall-timed pairs, if e7a completes before the E8 records are read. G2
passes for a family when the median absolute error of the predicted
candidate step is at most 15% on that family's development cases. Because
E8 is submitted before G2 can be applied, G2 no longer decides whether E8
runs. It decides how the reference is reported for a family: a family that
fails G2, or has no development case, is reported with the reference
marked "not validated on development data". The development errors are in
`DEV_REPORT.md`, frozen with the reference; errors on the e7a traces,
computed afterward with the frozen code, go to `DEV_REPORT_E7A.md` before
any E8 record is read.

## 9. Expectations (stated now; not used in scoring)
At p = 6 the shape rule and A1 cannot fire (r_V = 0.75), as in E7. At
p = 24, r_V = 3 and both can. The family-G candidate should pay at p = 24
on both models. The hook family's gain should be smaller at batch 128 than
at batch 32, because more of the exchange is hidden behind the backward
pass.

## 10. Budget and stop rules
Expected about 19 SU (e8s about 13 minutes, e8l about 21 minutes); time
limits of 30 and 40 minutes cap it at 38 SU. MOSAIC jobs are never
cancelled or changed. A job that completes no timed run (preflight or
calibration failure) may be resubmitted once, in the second window.

## 11. Frozen components for the E8 jobs
- harness v4, sha256 3aecaae1c1021269ae25a9e0468a7c7a6bcae7c27a7f64c8f275531e10207179 (git 35cf9be, the file round 4 and E7 run)
- harness v5.1, sha256 b920424cfa25bdd751491e9dd80fcb30898662e8c600a81331d2f75e20352f6b
- sha256 71329c336cc3e9646dbb6e6dd4ce20e12305d666c663f2a3e4b35195b535a822 `diagnostic/rules.py`
- sha256 95e240d86e79301e6740276f851a30bf6c9289c093cdc5147de92117bed1c7dd `diagnostic/features.py`
- sha256 246d3f929b2b26978fe5af910177a054d58c4ec7876ea30c61a6ddadf3423dea `diagnostic/v1_1/rescore.py`
- `harness/launch/e8_sets.sh` and `harness/launch/ls6_e8.slurm` at the commit that adds this file; each job records their hashes.
