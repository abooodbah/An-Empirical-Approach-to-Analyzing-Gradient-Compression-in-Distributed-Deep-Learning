# E12 (A-plus): a held-out set in which every case is traced (pre-registered)

Written on 2026-10-18, before any E12 job is submitted. This file, the scorer
`diagnostic/e12/score_e12.py`, the qualification check
`diagnostic/e12/qual_check.py`, the reverse substitutions of the mirror
cases `diagnostic/e12/replay_reverse.py` with their development report,
their tests, harness v5.5, the condition list
`harness/launch/e12_sets.sh` and the three job scripts are frozen by
`diagnostic/E12_FREEZE.sha256` (section 9). Nothing listed there changes
after the first submission; a later change is a new version, and E12 is then
reported as retrospective.

## 1. Questions

Q1. On a new set of cases in which about half of the proposed changes are
expected not to pay, which methods recommend the changes that pay and avoid
the ones that do not? Methods: the frozen rules, the benchmark check, the
frozen E8 replay, an always-recommend baseline and a calibration-only cost
model (section 5).
Q2. How far is each method's predicted gain from the measured one?
Q3. Do the answers change when a change counts as a fix only if it does not
increase the per-exchange gradient error?
Q4. Against a production baseline (dense DDP with bucketing and overlap, BF16
compute), do the 16-bit and int8 hooks pay?
Q5. How large is the phase-timing bias rho in each allocation, measured again
on the matched-encoding control?

## 2. Cases

Workloads: Transformer (T) and ResNet-50 (R). Each case below is measured on
both, so the set has 30 cases: the 24 of cases 01 to 12 and 6 mirror cases
(M01, M04, M12), which reverse the three case types the replay models so that
the replay is also scored on changes expected to hurt. A tag is a harness mode; "-bf16" marks a run
with `--dtype bf16` (forward pass and loss under bfloat16 autocast; parameters,
gradients, DDP buckets and the optimizer stay float32; the 16-bit hook then
sends bfloat16). `ddp-int8-hook` always runs its synchronous form
(`RED_HOOK_SYNC=1`), the form E7 and E8 measured and the frozen replay parses;
on NCCL at 6 ranks it was the faster of the two forms (E11, 1.01 to 1.14 times
faster). `ddp-fp16-hook` without "-bf16" sends float16.

| Id | Original | Intervention | Tests | Replay | T expected | R expected | Group |
|---|---|---|---|---|---|---|---|
| 01 | gpu-quant-gathersum | gpu-quant-reducible | shape | G | help | help | held-out |
| 02 | gpu-quant | gpu-quant-reducible | shape | none | help | help | held-out |
| 03 | cpu-quant | gpu-quant | placement | none | help | help | held-out |
| 04 | ddp-int8-hook | ddp-fp16-hook | shape | H | help | help | held-out |
| 05 | ddp-overlap | ddp-int8-hook | none | none | harm | harm | held-out |
| 06 | ddp-overlap | gpu-quant | none | none | harm | harm | held-out |
| 07 | ddp-overlap | cpu-quant | none | none | harm | harm | held-out |
| 08 | ddp-overlap | ddp-fp16-hook | none | none | help | neutral | held-out |
| 09 | gpu-quant | gpu-quant-gathersum | none | none | neutral | neutral | held-out |
| 10 | ddp-overlap-bf16 | ddp-fp16-hook-bf16 | none | none | help | neutral | production |
| 11 | ddp-overlap-bf16 | ddp-int8-hook-bf16 | none | none | harm | harm | production |
| 12 | ddp-int8-hook-bf16 | ddp-fp16-hook-bf16 | shape | H | help | help | production |
| M01 | gpu-quant-reducible | gpu-quant-gathersum | none | G-rev | harm | harm | mirror of 01 |
| M04 | ddp-fp16-hook | ddp-int8-hook | none | H-rev | harm | harm | mirror of 04 |
| M12 | ddp-fp16-hook-bf16 | ddp-int8-hook-bf16 | none | H-rev | harm | harm | mirror of 12, production |

"Tests" is the mechanism whose verdict counts as a recommendation of the
intervention by the rules (section 5). The rules have no verdict that
recommends adding compression to a dense exchange, or turning a reduce into
a gather, so for cases 05 to 11 and the mirrors they can only decline.
"Replay" is the substitution family: G and H are the frozen E8 reference's,
G-rev and H-rev their reverses (section 5); the replay has none for the
other cases and abstains on them. The mirrors need no new outcome runs: both
tags of each pair are already block-timed and traced in every scored
allocation. The scorer's group label of a mirror is that of its forward
case (held-out for M01 and M04, production for M12).

Expectations (stated now, never used in scoring), with their basis. Help
means a block-timed gain of at least 5%, harm a loss of at least 5%, neutral
in between. Batch 64 has not been measured on Lonestar6 at 6 ranks; the
values in brackets are measured at batch 32 (E8 e8s, E11b) and 128 (E7 e7a).
- 01: the matched gather-to-reduce substitution paid 0.34 (T) and 0.21 (R) at
  batch 32 and 0.17 to 0.20 (T) and 0.07 (R) at batch 128.
- 02: the rank-local int8 gather replaced by the reducible quantizer paid 0.20
  to 0.22 (T) and 0.08 (R) at batch 128. R may fall near 5%.
- 03: moving the int8 encoder from the CPU to the GPU paid 0.81 (T) and 0.61 to
  0.62 (R) at batch 128.
- 04: the int8 hook replaced by the 16-bit hook paid 0.28 to 0.29 (T) and 0.18
  to 0.19 (R) at batch 32 and 0.16 (T) and 0.08 to 0.09 (R) at batch 128.
- 05: the int8 hook was 1.19 to 1.21 (T) and 1.24 to 1.25 (R) times dense DDP
  at batch 32 (E11b, synchronous form).
- 06, 07: a gather without overlap, and the CPU-staged gather, against
  overlapped dense DDP. gpu-quant took 36 ms (T) and 173 ms (R) at batch 128
  in E7 against 28 ms and 161 ms for the overlapped 16-bit hook; cpu-quant took
  177 ms (T) and 447 ms (R).
- 08: the 16-bit hook was 0.84 to 0.86 (T) and 1.005 (R) times dense DDP at
  batch 32. ResNet-50 is the workload where communication is a small share of
  an overlapped step.
- 09: same gather shape, shared scale instead of per-rank scale: 4.1% (T)
  and 1.3% (R) apart at batch 128 (E7 e7a wall runs).
- 10: no measurement exists. BF16 compute shortens the backward pass, so the
  exchange is a larger share and the 16-bit hook should pay on T as in 08. R
  is expected within 5%, with low confidence.
- 11: as 05, with less compute to hide the gathers behind.
- 12: as 04, under BF16 compute.
- M01, M04, M12: the reverse of a change that pays loses: if 01 gains g, M01
  gains -g / (1 - g). The batch-128 values above give M01 -0.21 to -0.25 (T)
  and -0.075 (R), M04 -0.18 to -0.19 (T) and -0.09 to -0.10 (R) (E7 wall pairs,
  DEV_REPORT_REVERSE.md). M12 is expected to hurt with low confidence on R,
  as 12 is expected to help.
Tally: 12 expected to help, 14 to hurt, 4 neutral.

## 3. Platform and measurements

Lonestar6 development queue (`gpu-a100-dev`), 2 nodes, 3 A100 per node
(p = 6), NCCL with `NCCL_NET_GDR_LEVEL=SYS` (GPUDirect), environment
selection and transport settings as E11b (`harness/launch/ls6_e11b.slurm`).
Harness v5.5 for every run (`validation_v2/harness/build_v55.py`, hash in
`validation_v2/CODE_v55.sha256`; with default arguments it is v5.4). Local
batch 64 everywhere except the calibration run (batch 32, as in E8). Seed
2026010043, campaign id `redctl-e12-2026-10`, run-id prefix `redE12-ws6`.

Three jobs, in this order:
- e12q, qualification (`ls6_e12q.slurm`, 15-minute limit): preflight,
  calibration, the 16 fidelity runs (section 7.3), traces of
  `ddp-int8-hook-bf16` and `ddp-fp16-hook-bf16` on both workloads (parse
  checks for case 12 and mirror M12), and one block run each of R
  `cpu-quant` and T `ddp-overlap-bf16`. 24 conditions. Nothing in it is a
  scored outcome. Section 8 gives the go/no-go rule.
- e12a and e12b, the scored allocations (`ls6_e12a.slurm`, `ls6_e12b.slurm`,
  40-minute limit, or 45 under section 8). The same 86 conditions, in this
  order:
  1. preflight (Transformer native, qualification profile, batch 4);
  2. calibration (paper profile, ResNet-50 native, batch 32, 40 iterations
     after 10 warm-up; run id `redE12-ws6-resnet50-native`); every later run
     names it as its calibration reference;
  3. equivalence runs of the matched control, `gpu-quant-gathersum` and
     `gpu-quant-reducible` on both workloads (qualification profile,
     `--verify-equivalence`, 3 iterations): each arm checks outside the timed
     region that the other arm's exchange gives the same integer sum;
  4. M1, the rules' input: phase timing, extension profile, 40 iterations
     after 10 warm-up, for the ten originals (the seven of cases 01 to 12,
     and `gpu-quant-reducible`, `ddp-fp16-hook`, `ddp-fp16-hook-bf16` of the
     mirrors; `gpu-quant-reducible` is also the matched control's arm)
     (20 runs);
  5. traces, the replay's input: all ten tags on both workloads (20 runs),
     as E8's trace runs: qualification profile, wall timing, 8 iterations
     after 20 warm-up, input pool 4, Kineto traces of ranks 0 and 3;
  6. outcomes: all ten tags on both workloads, validation profile, block
     timing, 100 iterations after 20 warm-up, input pool 4, two passes, pass
     2 being pass 1 reversed (40 runs).
  Every method's input is measured before any outcome. No method sees a
  measurement of an intervention: interventions are traced so that a case
  whose intervention is another case's original has its input, and for the
  secondary replay-fidelity reading (section 7.5).

`harness/launch/test_e12_launch.py` checks that every command these scripts
emit is accepted by harness v5.5 and carries the run id the scorer reads.

## 4. Outcome and validity

Run value: the block-timed step `block_step_equiv_s` (the largest rank-local
block time over the 100 iterations, divided by 100), in ms. The value of a
tag in an allocation is the mean of its valid passes; a single valid pass is
used and marked. A pass is valid if every one of the 6 ranks wrote records,
every record has status PASS, `final_param_hashes_equal` True and a finite
loss, the ranks agree on the block value, the provenance counts no
synchronization inside the block, and the rank-0 provenance records the
tag's precision (`condition.compute_dtype`), for the 16-bit hook its wire type
(`ddp.dtype` float16, or bfloat16 under "-bf16"), and for the int8 hook the
synchronous form (`ddp.hook_state.mode` = sync). A tag with no valid pass is
missing, and so is every case that needs it. Nothing is rerun. A scored job
that completes no outcome run may be resubmitted once.

Measured gain of a case in an allocation: g = 1 - T(intervention) / T(original).
The intervention pays if g >= 0.05.

## 5. Methods

All predictions come from the original's M1 record, its traces and the
allocation's calibration, never from an intervention's run.

- Frozen rules (`diagnostic/rules.py`, 71329c33): the full procedure
  (`diagnose`, trend None as in E7 and E8), the traffic-ratio check (A1) and
  the phase-share check (A3), applied to the original's M1 record through
  the frozen extractor (`diagnostic/features.py`, 95e240d8) with the
  allocation's calibration. The scorer copies one run's CSVs and the
  calibration CSVs into a scratch directory of the allocation's name before
  calling the extractor, so the fp32 and bf16 forms of one harness mode,
  which the extractor would merge, stay apart. A rule set recommends a case's
  intervention only when its verdict is the mechanism the case tests
  (section 2), and never with a shape verdict on a reduce-shaped exchange.
  Other mechanism verdicts are recorded as "not tested" and are not
  recommendations; none, mixed and insufficient are abstentions. The effect
  e of a recommendation is the full procedure's; A1 and A3 give none.
- Benchmark check (A2, `rules.ablation_calibration_only`): shape when the
  original's exchange takes more than twice the calibrated all-reduce of the
  same payload. Scored as the rules are; it gives no effect.
- Replay (`diagnostic/e8/replay_reference.py`, frozen in E8, 35754666):
  g_pred from the original's traces on ranks 0 and 3 and the allocation's
  all-reduce calibration, family G for case 01 and H for 04 and 12. For the
  mirrors, the same replay with the reverse substitutions of
  `diagnostic/e12/replay_reverse.py` (the frozen reference is imported and
  not changed), from the original's traces and the allocation's calibration
  of both collectives:
  - G-rev (M01): the gradient all-reduce of the int8 bytes becomes an
    all-gather of the same bytes per rank, followed by p device copies of n
    bytes on the collective stream; the decode gains an int32 zero-fill and,
    per source, a cast and an add on the compute stream after the copies; the
    final cast reads int32 (8n bytes) instead of int8 (5n).
  - H-rev (M04, M12): each bucket's 16-bit hook (cast, divide, all-reduce,
    copy back) becomes the synchronous int8 hook on the compute stream: its
    statistics, quantization and per-source dequantization ops with their
    byte counts, and its two all-gathers (8 bytes and n bytes per rank), each
    followed by p copies that the compute stream waits for.
  An all-gather takes the allocation's fitted all-gather line (the one the
  cost model uses); elementwise ops take their bytes over the elementwise
  bandwidth measured in the same trace, scalar ops 1.5 us; every new op costs
  the median CPU time of the original's own codec or hook ops. Developed only
  on the e7a traces (p = 6, batch 128), with E7's wall-timed pairs as truth
  (`diagnostic/e12/DEV_REPORT_REVERSE.md`, written by
  `dev_reverse_e7a.py`): median absolute error of the predicted candidate step
  4.2% for G-rev and 4.2% for H-rev, 4 points each, largest 7.3% and 5.0%;
  the recommendation matches the measured outcome on all 8 points. The
  forward families on the same traces gave 5.0% and 5.7%
  (`diagnostic/e8/DEV_REPORT_E7A.md`). H-rev has not been run on a BF16
  trace; the qualification job checks that it parses one (section 8).
  The replay recommends when g_pred >= 0.05. On the other nine cases it has
  no substitution and abstains; its coverage is therefore 6 of 15 case types
  per workload (3 expected to help, 3 to hurt) and is reported beside its
  scores, with its metrics over the covered rows alone.
- Always-recommend: recommends every intervention; no effect. Its precision
  is the share of interventions that pay.
- Calibration-only cost model (cm): g_pred = (X(original) - X(intervention)) /
  T_M1(original), where T_M1 is the original's M1 step (the step the rules
  read) and X is the time of the collectives a mode issues per step, from
  the allocation's fitted lines with the harness's own alpha and beta undone:
  all-reduce of b bytes per rank 2(p-1) alpha_AR + beta_AR 2(p-1)/p b,
  all-gather (p-1) alpha_AG + beta_AG (p-1)/p b. With n gradient elements
  (from the M1 record):
  dense DDP AR(4n); 16-bit hook AR(2n); reducible AR(4) + AR(n); gather-sum
  AR(4) + AG(n); gpu-quant and cpu-quant AG(4) + AG(n); int8 hook
  AG(8) + AG(n). It recommends when g_pred >= 0.05. It ignores overlap, codec
  time and where the encoder runs: it is the bytes-on-the-wire argument the
  paper argues against, stated as a predictor. It is not A2, which
  `rules.py` also calls calibration-only.

## 6. Matched-encoding control and rho

Per allocation and workload, from runs already in section 3:
T^E (M1 step of `gpu-quant-gathersum` and of `gpu-quant-reducible`, the
extractor's mean over iterations of the largest rank's step) and T^B (their
block values). rho = T^E / T^B per arm; g^E = 1 - T^E(red) / T^E(gs);
g^B = 1 - T^B(red) / T^B(gs); bias g^E - g^B. "Verified equal" when both
equivalence runs record `equivalence_ok` True on every rank. Reported with
both allocations' values; no pooling.

## 7. Scoring

### 7.1 Weights and missing data
Each case and allocation is one row of weight 1/2, so a case measured in
both allocations has weight 1 and the set 30. A row whose outcome is missing,
or whose method input is missing (M1, trace, calibration, a replay that
cannot find its substitution), is left out of that method's scores and its
weight is reported as missing.

### 7.2 Metrics, per method
Over the method's scored rows, with the truth "pays" (g >= 0.05):
precision = TP / recommended; recall = TP / paying; false-positive rate =
FP / not paying; abstention = abstained / scored. Each with a 95% Wilson
score interval computed on the weighted counts. Gain-prediction error for
the methods that give a magnitude (full procedure, replay, cm): the median
of |e - g| over their recommendations and, for replay and cm, also over every
row they predict, each with a 95% percentile bootstrap interval that
resamples cases (both allocations of a case together; 2000 resamples, seed
20261018). The cases share originals and workloads, so the intervals
describe these 30 cases; they are not a population claim. The replay's
metrics are also reported over the 12 rows of its covered case types (01,
04, 12 and the three mirrors on both workloads), where about half the
changes are expected not to pay. The E7 and E8
labels (correct, short, null, harmful, abstain-correct, abstain-miss, not
tested, inapplicable) are printed beside each rules row for continuity.

### 7.3 Quality-constrained variant
A change counts as a fix only if it pays and its per-exchange error does not
exceed the original's by more than 1% of the original's: err(intervention)
<= 1.01 err(original). err is the mean, over 8 measured iterations of rank 0,
of the relative L2 distance between the exchanged gradient and the dense
float32 average of the same local gradients. Where it is recorded:
- non-DDP modes (gpu-quant, cpu-quant, gpu-quant-gathersum,
  gpu-quant-reducible): `--verify-fidelity`, harness v5.2 onward; the
  sidecar `<run>__<condition>__rank0000.v52.json`, key `fidelity`, one entry
  per iteration with `relative_l2_error` and `cosine`;
- the two hooks (ddp-fp16-hook, ddp-int8-hook, fp32 and bf16): new in v5.5,
  the same sidecar and key, measured per bucket inside the hook against an
  all-reduce of a float32 copy of the bucket, summed over buckets per
  iteration; also copied into the provenance at `ddp.hook_state.fidelity`;
- dense DDP (ddp-overlap, fp32 and bf16): sends the dense average itself, so
  err = 0 by definition; it is not run.
Fidelity runs are qualification-profile runs in the qualification job (run
ids `redE12-ws6-<model>-<tag>-b64-fid`), never in a timed run, because the
reference all-reduce would change the timing. A run with fewer than 8 finite
entries is missing, and so is the quality label of every case that needs it.
Consequence, visible before any run: a change from dense DDP to any
compressor (cases 05 to 08, 10, 11) cannot count as a fix under this variant;
M01 keeps the error of its original (the two arms send the same integers),
and M04 and M12 increase it.
All metrics of 7.2 are reported again with this truth.

### 7.4 Readings
Primary: frozen. Secondary: described (C1 and C2 of
`diagnostic/v1_1/rescore.py`), as in E7 and E8, for the rule sets only.

### 7.5 Reported but never used to score a method
The replay of each tag's own trace with no substitution, against that tag's
block value (simulator fidelity, separate from substitution error); the
expectation tally of section 2; pass gaps above 0.05; the E12q fidelity
values themselves.

## 8. Budget and the qualification rule

Lonestar6 charges 3 SU per node-hour on the A100 queues; a 2-node job costs
0.1 SU per minute of elapsed time. Per-condition fixed cost (launch,
rendezvous, model build, hash checks, records): 14 to 19 s, from the
timelines of the 2-node jobs 3480007 (E8 e8s, 26 conditions, 14 to 24 s each)
and 3509336 (E11b, 18 conditions, 17 to 40 s, the 40 s being the first
preflight); the 4-node job 3495229 (E9 q12a, 69 conditions) took 12 to 33 s
for all but two. Step times at batch 64
are interpolated from batch 32 and 128 (section 2), in ms, T / R:
ddp-overlap 18 / 85, ddp-fp16-hook 16 / 85, ddp-int8-hook 21 / 100,
gpu-quant 22 / 100, cpu-quant 170 / 380, gathersum 21 / 100,
reducible 17 / 90, ddp-overlap-bf16 13 / 55, ddp-fp16-hook-bf16 11 / 55,
ddp-int8-hook-bf16 17 / 70 (sums 326 / 1120).

Scored job (e12a or e12b), 86 conditions:
- block runs, 40: per pass 10 x 15 s + 120 x 0.326 s = 189 s (T) and
  10 x 18 s + 120 x 1.120 s = 314 s (R); two passes 1007 s;
- traces, 20: 10 x 18 s + 28 x 0.326 s x 1.5 = 194 s (T) and
  10 x 21 s + 28 x 1.120 s x 1.5 = 257 s (R); 451 s (1.5 for profiler
  overhead, 3 s for the trace export);
- M1, 20 (all ten tags): 10 x 15 s + 50 x 0.326 s x 1.2 = 170 s (T) and
  10 x 18 s + 50 x 1.120 s x 1.2 = 247 s (R); 417 s (1.2 for phase
  synchronization); the 4 runs the mirrors add (`ddp-fp16-hook` and
  `ddp-fp16-hook-bf16`) are 76 s of it;
- equivalence 4 x 17 s = 68 s; preflight 30 s; calibration 30 s;
- conditions 2002 s, plus 90 s for environment selection, node facts and
  srun start-up: 2092 s = 34.9 min, 3.49 SU.
Qualification job (e12q), 24 conditions: preflight 30 s, calibration 30 s,
16 fidelity runs 297 s (fixed cost plus 10 iterations at 1.3 times the step),
4 traces 85 s (`ddp-int8-hook-bf16` 43 s, `ddp-fp16-hook-bf16` 42 s), R
cpu-quant block 64 s, T ddp-overlap-bf16 block 17 s; conditions 522 s, plus
90 s: 612 s = 10.2 min, 1.02 SU.

Expected total 1.02 + 2 x 3.49 = 8.0 SU (7.7 before the mirrors). If every
fixed cost were 20 s (the high end seen) and every step 25% longer, a scored
job would take about 40.5 minutes, above its 40-minute limit; the
qualification rule would then move both scored jobs to 45 minutes, and the
total would be about 9.2 SU (1.1 + 2 x 4.05). Caps set by the time limits:
1.5 + 4.0 + 4.0 = 9.5 SU, or 10.5 SU under go-45. MOSAIC jobs are never
cancelled or changed.

Qualification rule (`diagnostic/e12/qual_check.py`), applied to the e12q run
directory before e12a and e12b are submitted. Checks: all 24 conditions
returned 0; every compressed tag has a complete fidelity record on both
workloads; both `ddp-int8-hook-bf16` traces exist on ranks 0 and 3 and the
frozen replay finds at least one hook bucket in each, and both
`ddp-fp16-hook-bf16` traces exist and the reverse substitution finds at least
one 16-bit-hook bucket in each (parses, not predictions); both block runs are
valid under section 4. Projection of a scored job: 90 s + 2002 s x max(e12q
condition span / 522 s, R cpu-quant block / 64 s). At the plan the projection
is 34.9 minutes, so an allocation 3.4% slower than planned already moves to
go-45. Decision: stop if a check fails or the projection exceeds 43
minutes; go-40 (submit the two scripts as written) if it is at most 36;
go-45 (submit both with `sbatch --time=00:45:00`, caps 4.5 SU each and 10.5
SU in all) otherwise. A stop is reported to the author with the e12q
records, and nothing else is submitted under this protocol.

## 9. Freeze procedure

1. After the last edit: `python validation_v2/harness/build_v55.py` (if the
   harness changed), `python harness/launch/make_ls6_e12.py`, then
   `python diagnostic/e12/make_freeze.py`, which writes
   `diagnostic/E12_FREEZE.sha256` (sha256 and path of the 24 files it
   lists, this protocol included).
2. Run the tests (`diagnostic`, `diagnostic/e8`, `diagnostic/e12`,
   `harness/launch`, `validation_v2/analysis`); `diagnostic/e12/test_e12_freeze.py`
   fails if any listed file differs from the freeze.
3. Commit the freeze file and push. The push time goes into
   `provenance/PROVENANCE.md` beside the E12 job times when the records are
   copied back.
4. Stage on Lonestar6 in RED_ROOT: `control_workload_benchmark_v5_5.py`,
   `CODE_v55.sha256`, `collective_probe.py`, `e12_sets.sh`; in
   `RED_ROOT/prereg/`: `E12_FREEZE.sha256` and every other listed file. Each
   job script checks every staged file against the freeze and checks that it
   is itself one of the three frozen scripts; it stops before any run
   otherwise.
5. Submit e12q; apply section 8; submit e12a and e12b, each from its own
   frozen script.
6. Copy the three run directories back, score with
   `python diagnostic/e12/score_e12.py --qual <e12q> <e12a> <e12b> --json
   diagnostic/E12_RESULT.json`, and the described reading with
   `--reading described`.

## 10. Expectations for the methods (stated now; not used in scoring)

- Full procedure: shape cannot fire at p = 6 (the traffic ratio is 0.75, as in
  E7 and E8). It should recommend case 03 (placement on cpu-quant) on both
  workloads and nothing else; on gpu-quant it may say placement, which is
  "not tested" for cases 02 and 09. Expected: high precision, low recall,
  false-positive rate near 0.
- Benchmark check: shape on the gather originals with a slow exchange, so
  cases 01 and 02 on both workloads; insufficient on the DDP originals.
- Replay: recommends 01, 04 and 12 on both workloads, with errors of about
  5% of the step on 01 and 04 as on the e7a traces (DEV_REPORT_E7A.md); 12
  is its first BF16 case. It declines M01, M04 and M12 on both workloads,
  with errors of about 4% of the step as on the e7a traces
  (DEV_REPORT_REVERSE.md); M12 is the reverse direction's first BF16 case.
  On its 12 covered rows: precision and recall near 1, false-positive rate
  near 0. The R side of M01 (-0.075 at batch 128, predicted -0.03 on the
  e7a traces) is the closest to the 5% line.
- Rules and benchmark check: decline every mirror (no verdict recommends a
  gather over a reduce or the int8 hook over the 16-bit hook).
- Always-recommend: precision near 0.4 (12 of 30 expected to pay),
  false-positive rate 1.
- Cost model: recommends most changes that send fewer bytes, including the
  harmful 05, 06, 07 and 11; expected false-positive rate near 1 on those.
  On the mirrors it should decline: with the e7a calibration an all-gather
  of n bytes per rank takes longer than an all-reduce of n or 2n bytes.
- Quality variant: only 01, 03, 04 and 12 can count as fixes (02 and 09 move
  to a coarser shared scale; 05 to 08, 10 and 11 start from an exact
  exchange).
- rho: above 1 for both arms on both workloads, with g^E within 0.05 of g^B
  on the Transformer, as in the v2 campaign at 12 ranks.

## 11. Differences from E7 and E8, stated for the reader

- M1 runs use harness v5.5 at batch 64 (E7 used v4 at 128, E8 v4 at 32 and
  128; E9's phase runs used v5.2). v5.5 changes nothing in the phase path
  with default arguments; the v5 line added an outer block timer around the
  measured loop and options the M1 runs do not enable (input pool, traces).
  The extractor and the rules are unchanged.
- The int8 hook is the synchronous form throughout, the form E7 and E8 ran.
- The replay is also scored on three mirror case types, with reverse
  substitutions added in E12 (`replay_reverse.py`); E8 had only the forward
  ones, all expected to help. The frozen E8 code is unchanged.
- Interventions are traced as well as originals; no method reads them.
- The outcome is block-timed, as in E8 (E7 used wall timing).
- Two allocations on the development queue, so allocation-to-allocation
  variation is seen twice, not estimated.
