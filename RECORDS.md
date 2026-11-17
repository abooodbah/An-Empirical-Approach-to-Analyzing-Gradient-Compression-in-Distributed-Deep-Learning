# Records and reproduction details

Where each held-out set, job and result lives, and how to regenerate them.

## The four held-out sets

| Set | Protocol | Result | Records |
|---|---|---|---|
| First (E6) | `diagnostic/E6_PROTOCOL.md` | `diagnostic/E6_FINAL.txt`, corrections in `diagnostic/v1_1/` | `raw/round4/` |
| Second (E7) | `diagnostic/E7_PROTOCOL.md` | `diagnostic/E7_RESULT.txt` | `raw/rev/` |
| Third (E8) | `diagnostic/E8_PROTOCOL.md` | `diagnostic/E8_RESULT.json` | `raw/rev/` |
| Fourth (E12) | `diagnostic/E12_PROTOCOL.md` | `diagnostic/E12_RESULT.md`, `E12_RESULT.json` | `raw/e12/` |

Each protocol and its frozen files were pushed before the set's jobs ran;
`provenance/PROVENANCE.md` gives the order of pushes and submissions, and the
`*_FREEZE.sha256` files list the frozen files with their hashes.

## Reproducing the evaluation

From the repository root, in a shell with `bash`, `tar`, `sha256sum` and
Python 3 with `numpy`, `matplotlib` and `pytest`:

```
REPRO_WORK=<work folder> bash reproduce_diagnostic.sh
```

The work folder defaults to `/c/reprow`. The script extracts the archives
into it, re-runs every scorer and generator on the archived records, and
compares each output with the committed copy. It covers all four held-out
sets (E6, E7, E8 and E12), the E9 and E10 scorers, the selection metrics,
the replay validation, the memory account and the other Revision 4 files,
and the tests of those modules. It needs no cluster and about 6 GB of free
disk for the work folder, takes about twenty minutes, and prints
`ALL REGENERATED` when everything matches. A file that
differs is restored to the committed copy and the script stops.

Three helper scripts take their input folder from an argument or an
environment variable, with a folder under `work/` as the default:

```
python diagnostic/alloc_compare.py <folder with raw/round4 j2 and j5 extracted>   # or R4W
python diagnostic/peak_mem_1b.py <folder with the three redctl-V archives extracted>  # or REV4_MEM_ROOT
TPDS_RAW_SEEDS=<folder with h100-3328658, -59, -60 extracted> python figures/make_figures.py
```

`make_figures.py` reads `processed/` (or `TPDS_FIG_DATA`) and writes to
`figures/figures/`. `raw/july/h100-3328657.tar.gz` is a qualification job
and is not part of the phase figure; leave it out of `TPDS_RAW_SEEDS`.

## Revision 4 jobs (E9 and E10)

Pre-registered in `diagnostic/E9_PROTOCOL.md` and `diagnostic/E10_PROTOCOL.md`, with the frozen
components listed in `diagnostic/E9_FREEZE.sha256` and `E10_FREEZE.sha256`. Jobs are in
`provenance/rev4_jobs.tsv` and charges in `provenance/rev4_su.tsv`.

E10, a bucketed gather on the 1B workload (Stampede3): records in `raw/rev4/redctl-E10-*.tgz`,
scored by `validation_v2/analysis/score_e10.py` into `validation_v2/results/rev4/E10_RESULT.json`.
The 16- and 8-rank allocation completed every condition. The first 32-rank allocation (job 3579138) completed its two
check runs and no timed run. One of its nodes was drained with "GPU missing" 44 s after the job ended
(`provenance/stampede3_node_c552-003.txt`); no record establishes that this caused the stalled run.
It was resubmitted once (job 3579496), as section 4 of the protocol allows, and the resubmission
completed all three timed conditions with 20 measured iterations each. The scored result keeps both
32-rank allocations; the first has state `missing`. The three jobs were charged 20.17 SU, 0.17 SU
above the 20 SU of section 6; the resubmission was approved before it was submitted, with an estimate
of up to 21.84 SU (`provenance/PROVENANCE.md`).
The scorer's model check uses the allocation's probe, taken at 16 ranks, so its value for the
8-rank condition compares against a 16-rank line and is not a test of the model.

E9, reducible quantizers and a matched DDP baseline (Lonestar6): records in `raw/rev4/e9-*.tgz`, scored by
`validation_v2/analysis/score_e9.py` into `validation_v2/results/rev4/E9_RESULT.json`. All four jobs
completed. Every outcome run that is present passed the validity gates, both equivalence runs recorded
True in every allocation, and the DDP bucket structure matched in every allocation. Five runs failed at
start-up with a rendezvous or connection error: three conditions have one valid pass in one allocation,
and the fidelity values of resnet50 gpu-quant at 12 ranks and resnet50 gpu-quant-reducible at 24 ranks
are absent. Nothing was rerun. Charges: 26.467 SU on Lonestar6.

The protocol's five expectations and three probe checks are set against the scored result in
`validation_v2/results/rev4/E9_EXPECTATIONS.md`, written by `validation_v2/analysis/e9_expectations.py`.
X1, X2, X3 and X5 held. X4 did not: the 16-bit hook is slower than dense DDP on ResNet-50 at 24 ranks
(1.06 in both allocations). The three probe checks held at 12 ranks and over the host-staged transport.
They did not hold at 24 ranks over GPUDirect, in either allocation (slope ratio 30.4 and 33.4 against p/2 = 12).

A second reading of the probes, made after the records were read and labeled post hoc, is in
`validation_v2/analysis/e9_probes_posthoc.py` and `E9_PROBES_POSTHOC.json`. The protocol's line passes
through three points of the probe grid: 1, 16 and 64 MiB. At 24 ranks over GPUDirect the forced-ring
all-reduce of 1 MiB took 10.26 and 10.64 ms, longer than the 16 MiB one (3.80 and 4.01 ms), and that
point lowers the fitted ring slope, the denominator of P1 and P2. With the line through 16 and 64 MiB
the slope ratio is 11.75 and 12.12 at 24 ranks and 5.45 and 5.68 at 12. Over three lines that leave the
1 MiB point out (16 to 64, 16 to 256 and 64 to 256 MiB) it is 10.46 to 12.12 at 24 ranks, inside the
factor 1.5 of p/2 each time.

The other two checks do not pass cleanly under the second reading. P2 still fails at 24 ranks
(0.98 and 1.06 with the line through 16 and 64 MiB, up to 1.11 with the others). P3 is scored on the
default probe, whose own 1 MiB point is high, so the forced ring's point does not enter it. P3 holds
for the line through 16 and 64 MiB at 24 ranks (13.4% and 10.3%) and fails for the two longer lines at
both scales (15.7% to 25.4%). `E9_EXPECTATIONS.md` prints every line with a verdict for each check.

The same script reads the forced-tree probe, which the frozen scorer collects and does not read: at
1 GiB the tree all-reduce took 11.8 to 12.3 times the ring's time over GPUDirect, and 4.1 to 4.2
times host-staged. Under the protocol's own range the script repeats the frozen scorer's numbers,
and a test compares them. It replaces `e9_tree_posthoc.py` of commit 5573a1d, whose 24-rank ratio of
the tree to the ring carried the same 1 MiB point. The second reading replaces no scored number.

The two jobs of a scale ran on the same nodes, and the 24-rank jobs contain the four nodes of the
12-rank jobs (`environment/E9_NODE_FACTS.json`): 24 node records describe 8 nodes. One of the 8, present
in both 24-rank jobs and in neither 12-rank job, has a different adapter (a BlueField-2 integrated
ConnectX-6 Dx with firmware 24.33.1048, where the other seven have a ConnectX-6 with 20.32.2004). The
two allocations of a scale are therefore not independent in hardware, and every contrast between 12
and 24 ranks carries that node.

`validation_v2/analysis/e9_pass_spread_posthoc.py` (post hoc) keeps each pass of every condition in
`E9_PASS_SPREAD_POSTHOC.json`. The scored step is the mean of two passes, and the calls compare such
means at 5%. The two passes of a condition differ by a median of 0.8% at 12 ranks and 0.9% at 24, by more
than 5% in 2 of 34 conditions at 12 ranks and 8 of 35 at 24, and by 14.8% at most.

In the two jobs that probed both transports, collectives of 16 MiB and more ran faster over GPUDirect
than host-staged, by 1.7 to 2.9 times (for example 11.8 against 27.0 ms for a 64 MiB forced-ring
all-reduce at 12 ranks). The September triad (`raw/control/ls6-triad4-*`) had the order the other way
in its training loop, from two jobs on different nodes, one per transport.

The int8 hook in every harness version issues its two all-gathers synchronously on NCCL as well as on
oneCCL: the harness tests for `"ccl"` in the backend name, and `nccl` contains it. Every int8-hook
provenance record in `raw/` carries `hook_state.mode` = `sync`. The 16-bit hook is PyTorch's own and is
asynchronous. The four E9 jobs, on the frozen harness v5.2, record `sync` in every int8-hook record.

## Revision 4: retrospective analyses and how to regenerate them

Devised after the outcomes they analyze. Each file's docstring says so, apart from
`platform_facts.py`, `INPUT_SCHEMA.md` and `build_v52.py`, which analyze no outcome:

- `diagnostic/v1_2/metrics.py` writes `METRICS.json` and `METRICS.txt`: coverage, precision, recall and
  false-positive rate of every rule set on the three held-out sets, with intervals from resampling cases.
  A rate of exactly 0 or 1 carries its count and no interval.
- `diagnostic/v1_2/replay_validation.py` writes `REPLAY_VALIDATION.csv`: every prediction of the replay
  beside the measured gain.
- `diagnostic/v1_2/memory_account.py` writes `MEMORY_ACCOUNT.md`: peak device memory of the 1B conditions.
- `validation_v2/analysis/intervals.py` writes `results/rev4/INTERVALS.*`: allocation-level intervals.
- `validation_v2/analysis/traffic_models.py`: ring, tree, reduce-scatter and shard-wise traffic, checked
  by a link-load counter in its tests.
- `validation_v2/analysis/step_spread.py` writes `results/rev4/H100_STEP_SPREAD.json`: how far the three
  July jobs are apart on each bar of the phase-attribution figure.
- `validation_v2/analysis/e9_probes_posthoc.py` writes `results/rev4/E9_PROBES_POSTHOC.json`: the E9
  probes with three lines that leave the 1 MiB point out, the measured points, and the forced-tree
  probe.
- `validation_v2/analysis/e9_pass_spread_posthoc.py` writes `results/rev4/E9_PASS_SPREAD_POSTHOC.json`:
  each pass of every E9 condition and how far the two are apart.
- `validation_v2/analysis/e9_expectations.py` writes `results/rev4/E9_EXPECTATIONS.*`: the protocol's
  expectations and probe checks against the scored result. It reads the frozen scorer's output only,
  apart from one section that repeats the post hoc reading and is labeled so.
- `environment/platform_facts.py` writes `environment/PLATFORMS.md` from the archived provenance.
- `environment/node_facts.py` writes `environment/E9_NODE_FACTS.json` from the node records of the four
  E9 jobs: the hosts of each job, and per node the GPU driver and VBIOS, processors, adapter and
  firmware, and the NUMA placement of each. Values are counted per node, not per record.
- `diagnostic/INPUT_SCHEMA.md` lists the fields the scorers read.
- `validation_v2/harness/build_v52.py` builds harness v5.2 from v5.1 by insertion only.

## Notes for reading the records

- Phase timing is exposed cost: every phase boundary is device synchronized
  and a barrier separates iterations, which suppresses overlap. The
  block-timed and wall-mode runs measure the same conditions without it.
- Step time is the per-iteration maximum across ranks.
- Every E6 job ran harness v4 with sha256 3aecaae1 (git 35cf9be). The
  current `harness/control_workload_benchmark_v4.py` was edited afterward,
  so a rerun of E6 should check that file out from 35cf9be.
- The int8 hook issues its all-gathers synchronously in every harness
  version, on NCCL as well as on oneCCL; every int8-hook provenance record
  carries `hook_state.mode` = `sync`.
- The launch scripts expect the TACC Stampede3 and Lonestar6 environments
  described in `environment/`.
