# Artifact: An Empirical Approach to Analyzing Gradient Compression in Distributed Deep Learning

This repository is the artifact for the paper "An Empirical Approach to
Analyzing Gradient Compression in Distributed Deep Learning". Earlier drafts
carried the working title "Beyond Compression Ratio", which the repository
name keeps. It holds the measurement harness, the launch scripts, the raw
per-rank records of every job, the collective calibration data, the
processed tables, the diagnostic rules, the protocols and scorers of the
four held-out sets, the overlap-aware replay, and a script that regenerates
the evaluation from the archived records.

## What each folder holds

- `harness/`: the benchmark programs and the Slurm launch scripts
  (`harness/launch/`). `paper_workload_benchmark.py` is the frozen July
  harness. The control harnesses (v3, v4) are generated from it by the
  `make_control_harness*.py` scripts, which assert every edit.
- `validation_v2/`: the validation campaign and the later harness versions
  (v5 to v5.5 in `validation_v2/harness/`), its frozen analysis rules and
  code, the E9 and E10 scorers and post hoc readings
  (`validation_v2/analysis/`), and their results (`validation_v2/results/`).
- `diagnostic/`: the frozen diagnostic rules (`rules.py`, hash in
  `RULES_FROZEN.sha256`), the feature extractor, the protocols, scorers and
  results of the four held-out sets, the replay (`diagnostic/e8/`) and its
  reverse substitutions (`diagnostic/e12/`), and retrospective analyses
  (`diagnostic/v1_1/`, `diagnostic/v1_2/`).
- `raw/`: one archive per job, each with a sha256 file where one was made.
  `july/` (H100 and Intel campaign), `control/` (September controls),
  `round4/` (first held-out set), `v2/` (validation campaign), `rev/` (second
  and third held-out sets), `rev4/` (E9 and E10), `e12/` (fourth held-out
  set). Each run directory holds per-rank CSV, JSONL and provenance records.
- `processed/`: the aggregated CSVs the figures and tables read.
- `figures/`: `make_figures.py` (Figs. 1 to 4 and the Section V numbers) and
  `make_fig_replay.py` (predicted against measured speedup of the replay).
- `environment/`: software and node facts per platform.
- `provenance/`: job lists with submit times, SU records, push times of each
  protocol, and the freeze tests.
- `contract/`: the experiment contract and campaign protocol of the July runs.

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
and the tests of those modules. It needs no cluster, takes about twenty
minutes, and prints `ALL REGENERATED` when everything matches. A file that
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

These are execution-cost measurements on synthetic fixed-input workloads. No
accuracy or convergence claim is supported.

## License

MIT, see `LICENSE`.
