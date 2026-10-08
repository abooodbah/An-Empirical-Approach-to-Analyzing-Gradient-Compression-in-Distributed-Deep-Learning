# What the frozen scorers read

One row per measured condition, written by `features.py` (sha256 95e240d8
as the first set was scored; a3eef19d as frozen on 2026-09-21, see
`v1_1/EXTRACTOR_FREEZE.md`). The rule file reads these fields and nothing
else. Only records of the paper and extension profiles are read.

| Field | Meaning |
|---|---|
| `tag` | Campaign tag of the record, taken from the name of its run directory |
| `alloc` | Slurm job id of the allocation |
| `model` | Workload |
| `mode` | Exchange method (the harness `comm_mode`) |
| `p` | Number of ranks |
| `tau` | Synchronization interval |
| `barrier` | The record's `tau_barrier` label: `every_iteration`, or `none_on_local_iterations` when the barrier was removed |
| `timing` | `phase` (Tier 1) or `wall` (Tier 2) |
| `kind` | `gather` or `reduce`, from the method's collective |
| `step_ms` | Mean over iterations of the largest step time across ranks, in milliseconds |
| `exchange_share` | Exchange phase over the step; empty when the record has no separable exchange phase |
| `codec_share` | Encode plus decode over the step; 0, not empty, when the record has no encode or decode phase |
| `barrier_share` | Barrier wait over the step; empty when the record has no barrier phase |
| `shape_ratio` | Exchange time over the calibrated all-reduce time for the same submitted bytes; empty without a separable exchange phase or a calibration |
| `volume_ratio` | Received-traffic ratio of Eq. (1), from recorded bytes; 1 for a reduce-shaped method |

Scorers: `score_e6.py` (first set), `score_e7.py` (second), `e8/score_e8.py`
with `e8/replay_reference.py` (third). Hashes: `provenance/rev4_baseline.sha256`.
Regenerate every table with `reproduce_diagnostic.sh`.
