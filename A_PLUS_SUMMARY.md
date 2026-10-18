# A-plus campaign (E12): one-page summary

Prepared on branch `a-plus-campaign` (local, not pushed). Nothing has been
submitted. The full rules are in `diagnostic/E12_PROTOCOL.md`.

## What runs

Three Lonestar6 development-queue jobs. Each uses 2 nodes (6 A100), GPUDirect, harness v5.5 and local batch 64.

1. **e12q, qualification** (`harness/launch/ls6_e12q.slurm`, 15-minute limit). It measures the per-exchange
   gradient error of the eight compressed modes on both workloads. It traces the BF16 int8 hook so we can check
   that the frozen replay can parse it. It also times the slowest outcome run once. `diagnostic/e12/qual_check.py`
   then decides stop, go-40 or go-45.
2. **e12a and e12b, the scored allocations** (`ls6_e12a.slurm`, `ls6_e12b.slurm`, 40-minute limit). Both jobs run
   the same 82 conditions:
   - preflight and calibration;
   - equivalence runs of the matched control;
   - 16 phase-timed runs (the rules' input);
   - 20 traces (every mode, originals and interventions);
   - 40 block-timed outcome runs (two passes, the second reversed).

Cases: 12 per workload, 24 in all (table in section 2 of the protocol). By expectation, 12 help, 8 hurt and 4 are
neutral. They include:
- the matched gather-to-reduce change;
- int8 hook to 16-bit hook;
- CPU to GPU encoding;
- three compressors substituted for dense overlapped DDP;
- the production baseline: dense DDP in BF16 against the BF16 16-bit hook and the int8 hook.

BF16 is new in harness v5.5. `--dtype bf16` runs autocast compute, and the 16-bit hook then sends bfloat16. v5.5
also adds per-exchange error measurement for the two DDP hooks.

## Estimated SU, with the arithmetic

The rate is 3 SU per node-hour, so a 2-node job costs 0.1 SU per elapsed minute. Each condition has a fixed cost of
14 to 19 s; this comes from the timelines of jobs 3480007 and 3509336. Step times at batch 64 are interpolated from
the batch-32 and batch-128 runs.

| Job | Conditions | Estimated time | Estimated SU | Cap (time limit) |
|---|---|---|---|---|
| e12q | 22 | 480 s + 90 s = 9.5 min | 0.95 | 1.5 |
| e12a | 82 | 1926 s + 90 s = 33.6 min | 3.36 | 4.0 |
| e12b | 82 | 33.6 min | 3.36 | 4.0 |
| Total | | | **7.7** | 9.5 |

Where the 1926 s for a scored job goes:

| Runs | Count | Time |
|---|---|---|
| Block | 40 | 1007 s |
| Trace | 20 | 451 s |
| Phase | 16 | 341 s |
| Equivalence | 4 | 68 s |
| Preflight and calibration | 2 | 60 s |

If every fixed cost were 20 s and every step 25% longer, a scored job would take about 40 minutes and the total
would be 8.9 SU.

The qualification rule moves the scored jobs to a 45-minute limit when the projection is between 36 and 43 minutes.
The caps then add up to 10.5 SU, above your 9. Above 43 minutes it stops.

## What each result would show

- **Precision, recall and false-positive rate**, for each method on a set where about half the changes do not pay.
  The methods are the frozen rules, the benchmark check (A2), the frozen E8 replay, always-recommend and a
  calibration-only cost model.
  - If the rules and the replay keep the false-positive rate near 0 while always-recommend and the cost model are
    near 1, that is the paper's specificity claim on a new set that includes the production baseline.
  - The rules' recall will be low by construction: no rule verdict recommends adding compression.
  - The replay covers only 3 of the 12 case types (G, and H twice) and abstains on the rest.
- **Gain-prediction error** of the replay at batch 64, plus its first BF16 case (12).
- **Quality-constrained variant.** Only cases 01, 03, 04 and 12 can count as fixes. Any move from exact dense DDP to
  a compressor fails it by definition.
- **BF16 production baseline**: whether the 16-bit hook still pays on the Transformer when compute is BF16, and
  whether the int8 hook still loses.
- **rho and the phase-timing bias** of the matched control, measured again in each allocation.

## Risks

- The batch-64 step times are estimates, not measurements, and so are the cost and some expectations.
- There are only two allocations, and the cases share originals. The Wilson and bootstrap intervals describe these
  24 cases; they are not a population claim.
- The int8 hook always runs its synchronous form, because that is what the frozen replay parses. On NCCL at 6 ranks
  this form was faster than the asynchronous one (E11), so it is the stronger int8 baseline. A reviewer may still
  ask for PyTorch's default form.
- The phase-timed runs use harness v5.5 at batch 64. E7 and E8 used v4.
- ResNet-50 under BF16 has not run anywhere yet. The qualification job's ResNet BF16 fidelity and trace runs are the
  first runs of it.
- The per-exchange error comes from the qualification allocation, not from the scored ones.
- The development queue may run one job at a time, which only delays the jobs.

## Before submission (protocol section 9)

1. Review the branch. After any edit, regenerate the scripts (`make_ls6_e12.py`) and the freeze file
   (`diagnostic/e12/make_freeze.py`), then rerun the tests.
2. Commit, push, and stage the files listed in `diagnostic/E12_FREEZE.sha256`.
3. Submit e12q, then run `qual_check.py`, then submit e12a and e12b.

The job scripts refuse to start if any staged file differs from the freeze.
