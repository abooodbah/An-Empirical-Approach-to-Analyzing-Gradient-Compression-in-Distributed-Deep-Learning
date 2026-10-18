# A-plus campaign (E12): one-page summary

Prepared on branch `a-plus-campaign` (local, not pushed). Nothing has been
submitted. The full rules are in `diagnostic/E12_PROTOCOL.md`.

## What runs

Three Lonestar6 development-queue jobs. Each uses 2 nodes (6 A100), GPUDirect, harness v5.5 and local batch 64.

1. **e12q, qualification** (`harness/launch/ls6_e12q.slurm`, 15-minute limit, 24 conditions). It measures the
   per-exchange gradient error of the eight compressed modes on both workloads. It traces the BF16 int8 hook and
   the BF16 16-bit hook, so we can check that the replay can parse both before anything is scored. It also times
   the slowest outcome run once. `diagnostic/e12/qual_check.py` then decides stop, go-40 or go-45.
2. **e12a and e12b, the scored allocations** (`ls6_e12a.slurm`, `ls6_e12b.slurm`, 40-minute limit). Both jobs run
   the same 86 conditions:
   - preflight and calibration;
   - equivalence runs of the matched control;
   - 20 phase-timed runs (the rules' input, all ten modes);
   - 20 traces (every mode, originals and interventions);
   - 40 block-timed outcome runs (two passes, the second reversed).

Cases: 15 per workload, 30 in all (table in section 2 of the protocol). By expectation, 12 help, 14 hurt and 4 are
neutral. They include:
- the matched gather-to-reduce change;
- int8 hook to 16-bit hook;
- CPU to GPU encoding;
- three compressors substituted for dense overlapped DDP;
- the production baseline: dense DDP in BF16 against the BF16 16-bit hook and the int8 hook;
- three mirror cases, M01, M04 and M12: cases 01, 04 and 12 run backwards (reduce to gather, 16-bit hook to int8
  hook, in fp32 and BF16). They need no new outcome runs, because both modes of each pair are already timed and
  traced. They let us check that the replay turns down a change that hurts, not only that it accepts one that
  helps.

BF16 is new in harness v5.5. `--dtype bf16` runs autocast compute, and the 16-bit hook then sends bfloat16. v5.5
also adds per-exchange error measurement for the two DDP hooks.

## Estimated SU, with the arithmetic

The rate is 3 SU per node-hour, so a 2-node job costs 0.1 SU per elapsed minute. Each condition has a fixed cost of
14 to 19 s; this comes from the timelines of jobs 3480007 and 3509336. Step times at batch 64 are interpolated from
the batch-32 and batch-128 runs.

| Job | Conditions | Estimated time | Estimated SU | Cap (time limit) |
|---|---|---|---|---|
| e12q | 24 | 522 s + 90 s = 10.2 min | 1.02 | 1.5 |
| e12a | 86 | 2002 s + 90 s = 34.9 min | 3.49 | 4.0 |
| e12b | 86 | 34.9 min | 3.49 | 4.0 |
| Total | | | **8.0** | 9.5 |

The mirror cases add 0.3 SU to the earlier 7.7: four phase-timed runs of the 16-bit hooks in each scored job (76 s),
so the rules and the cost model have an input for M04 and M12, and two traces in the qualification job (42 s).

Where the 2002 s for a scored job goes:

| Runs | Count | Time |
|---|---|---|
| Block | 40 | 1007 s |
| Trace | 20 | 451 s |
| Phase | 20 | 417 s |
| Equivalence | 4 | 68 s |
| Preflight and calibration | 2 | 60 s |

If every fixed cost were 20 s and every step 25% longer, a scored job would take about 40.5 minutes, just over its
limit. The qualification job would see this first and move both scored jobs to 45 minutes; the total would then be
about 9.2 SU.

The qualification rule moves the scored jobs to a 45-minute limit when the projection is between 36 and 43 minutes.
At the plan the projection is 34.9 minutes, so an allocation 3.4% slower than planned already moves to 45. The caps
then add up to 10.5 SU, above your 9. Above 43 minutes it stops.

## What each result would show

- **Precision, recall and false-positive rate**, for each method on a set where more than half the changes (18 of 30
  by expectation) do not pay.
  The methods are the frozen rules, the benchmark check (A2), the frozen E8 replay, always-recommend and a
  calibration-only cost model.
  - If the rules and the replay keep the false-positive rate near 0 while always-recommend and the cost model are
    near 1, that is the paper's specificity claim on a new set that includes the production baseline.
  - The rules' recall will be low by construction: no rule verdict recommends adding compression.
  - The replay covers 6 of the 15 case types per workload: 3 expected to help (01, 04, 12) and 3 expected to hurt
    (their mirrors). It abstains on the rest. Its scores are also given over the covered rows alone, where half the
    changes should not pay.
  - The reverse direction is new code (`diagnostic/e12/replay_reverse.py`); the frozen E8 replay is unchanged. On
    the e7a development traces its predicted step for the changed mode was off by a median 4.2% for both families
    (largest 7.3%), and
    it called all 8 development points the right way (`diagnostic/e12/DEV_REPORT_REVERSE.md`). The forward
    direction on the same traces was off by 5.0% and 5.7%.
- **Gain-prediction error** of the replay at batch 64, in both directions, plus its first BF16 cases (12 and M12).
- **Quality-constrained variant.** Only cases 01, 03, 04 and 12 can count as fixes. Any move from exact dense DDP to
  a compressor fails it by definition.
- **BF16 production baseline**: whether the 16-bit hook still pays on the Transformer when compute is BF16, and
  whether the int8 hook still loses.
- **rho and the phase-timing bias** of the matched control, measured again in each allocation.

## Risks

- The batch-64 step times are estimates, not measurements, and so are the cost and some expectations.
- There are only two allocations, and the cases share originals. The Wilson and bootstrap intervals describe these
  30 cases; they are not a population claim. A mirror is not independent of its forward case: both use the same two
  block-timed values, so a forward case and its mirror are right or wrong together for the outcome, though the
  replay predicts each from a different trace.
- The reverse substitutions were checked on 8 development points, all fp32 at batch 128. The ResNet side of M01
  is the closest to the 5% line (-0.075 measured at batch 128, -0.03 predicted).
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

Staging and submission, from the repository root (R is the campaign root on Lonestar6, as for E11; the jobs read
it from the environment):

```
R=<RED_ROOT on Lonestar6>
ssh ls6 "mkdir -p $R/prereg"
scp validation_v2/harness/control_workload_benchmark_v5_5.py validation_v2/CODE_v55.sha256 \
    validation_v2/harness/collective_probe.py harness/launch/e12_sets.sh ls6:$R/
scp diagnostic/E12_FREEZE.sha256 $(awk '{print $2}' diagnostic/E12_FREEZE.sha256 | grep -v -e _v5_5.py -e CODE_v55 \
    -e collective_probe -e e12_sets.sh) ls6:$R/prereg/
# on Lonestar6
export RED_ROOT=$R; cd $R/prereg
sbatch ls6_e12q.slurm
# when it ends: copy $R/runs/e12-q-2n-<job> back, then on the laptop
python diagnostic/e12/qual_check.py <local copy of e12-q-2n-<job>>
# go-40
sbatch ls6_e12a.slurm; sbatch ls6_e12b.slurm
# go-45
sbatch --time=00:45:00 ls6_e12a.slurm; sbatch --time=00:45:00 ls6_e12b.slurm
```
