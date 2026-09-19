# Beyond Compression Ratio: measurement artifact

This repository holds the measurement harness, raw per-rank records,
launch scripts, and processing data behind the paper "Beyond
Compression Ratio: Collective Shape, Execution Placement, and Scale in
Distributed Training." Nothing here is post-processed beyond the CSV
aggregations in `processed/`; every number in the paper can be traced
to a per-rank record in `raw/`.

The study asks one question: when gradient compression fails to reduce
wall-clock step time, where did the cost go? The harness answers it by
timing every phase of every training step under device synchronization,
on every rank, and by calibrating the collectives directly on the same
allocations.

## Layout

- `harness/` — the benchmark programs and launch scripts.
  `paper_workload_benchmark.py` is the frozen July harness that
  produced the main campaign. `control_workload_benchmark.py` is the
  September control variant, generated from the frozen file by
  `make_control_harness.py`; it adds a shared-scale summable int8 mode
  exchanged by all-reduce, a Top-k mode with standard local error
  feedback, and an uninstrumented wall-clock timing mode. The frozen
  file is unchanged; the generator asserts every edit.
- `contract/` — the experiment contract and campaign protocol the July
  runs were validated against.
- `raw/july/` — the July campaign archives: three independent H100 2x4
  seed jobs (3328658 to 3328660, host-staged NDR InfiniBand) and the
  four-node Intel weak-scaling job (3323561), with checksums. Each
  archive holds per-rank CSV, JSONL, and provenance records for every
  condition, plus transport validations.
- `raw/control/` — the September control campaign: the reducible int8
  and local-error-feedback conditions, uninstrumented timing repeats,
  and second-allocation Intel endpoint repeats.
- `processed/` — the aggregated CSVs the figures and tables read.
- `figures/` — the scripts that regenerate every figure from `raw/`
  and `processed/`.
- `environment/` — software identity per platform track.

## Measurement conventions

Two conventions matter when reading the raw records.

1. Phase timing is exposed cost. Every phase boundary is device
   synchronized and an explicit barrier separates iterations, so
   phases are disjoint and attributable. This deliberately suppresses
   overlap; the DDP control bounds what overlap recovers, and the
   wall-mode records in `raw/control/` measure the same conditions
   with no instrumentation beyond one synchronization per step.
2. Step time is the per-iteration maximum across ranks. Phase
   quantities pool rank-iterations. Percentiles are computed within a
   job and averaged across jobs where independent jobs exist.

## Reproducing

Each run directory under the raw archives is self-describing: the
provenance JSON records the command line, source hash, library
versions, Slurm layout, and calibration constants for that run. The
launch scripts under `harness/launch/` are the exact scripts the
campaigns used; they expect the TACC Stampede3 environment described
in `environment/versions.md`.

The collective cost model in the paper (Section IV) is fitted from
`processed/h100_fabric_collectives.csv`; the weak-scaling table from
`processed/intel_step_scaling.csv`; the phase attribution from the
per-rank records of the three H100 seeds.

## Boundary

These are execution-cost measurements on synthetic fixed-input
workloads. No accuracy or convergence claim is supported, and the
multi-node transports are host-staged; the records include the
transport validations that prove which path ran.
