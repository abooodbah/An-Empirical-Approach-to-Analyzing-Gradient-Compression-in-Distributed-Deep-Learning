# E6 protocol, held-out manifest, and job ledger (frozen 2026-09-21)

Rules: rules.py v1-2026-09-21, sha256 71329c33... (RULES_FROZEN.sha256).
Developed from Stampede3 tags A, B, C, V only. Frozen before any run
below and before inspecting any E1 to E5 result.

## Measurements per E6 case (all paired inside one allocation)

| Measurement | Mode | Use |
|---|---|---|
| M1 original, instrumented (Tier 1) | phase | input to the frozen rules |
| M2 original, uninstrumented | wall | production baseline T_orig^U |
| M3 intervention, uninstrumented | wall | T_int^U |
| M4 intervention, instrumented | phase | evidence about the attributed mechanism; not scored |

delta = (T_orig^U - T_int^U) / T_orig^U, both from the same allocation,
same warm-up, same iteration count. M2 and M3 are the scored pair; M1
never enters the score except through the verdict it produced.

Scoring, from rules.py (fixed): correct if the named intervention gave
delta >= max(0.05, 0.5 e); null if |delta| < 0.05; harmful if delta <=
-0.05; abstentions scored as coverage and checked against whether any
tried intervention paid. Report the per-case table first; counts second;
no headline percentage without the table. Twelve cases; one case is 8.3
points, and the paper says so.

## Held-out manifest

Previously observed (development or confirmation only; never scored):
every Stampede3 condition (tags A, B, C, V) and the Lonestar6 12-rank
triad and hooks on both transports (jobs 3456440, 3457305). Those LS6
runs were inspected when the transport paragraph was written, so they
are not held out despite tag L being excluded from rule development.

Scored, held-out cases (new combinations or new scale; none inspected
before the rules froze):

| # | Case (LS6 A100, GDR, 3 GPUs/node) | p | Expected mechanism family | Intervention the rules would name |
|---|---|---|---|---|
| 1 | Transformer, gather int8 | 24 | shape (new scale) | reduce-shaped same encoding (arm A) |
| 2 | ResNet-50, gather int8 | 24 | shape or none (new scale) | arm A |
| 3 | Transformer, arm B gathersum | 12 | shape (new method) | arm A |
| 4 | ResNet-50, arm B gathersum | 24 | shape (new method, new scale) | arm A |
| 5 | Transformer, Top-k all-gather | 12 | placement (new platform) | cheaper selection: TopKC-style chunk all-reduce port (bundled intervention, labeled) |
| 6 | ResNet-50, Top-k all-gather | 24 | placement | same |
| 7 | Transformer, async int8 hook | 24 | shape under overlap (new: asynchronous) | 16-bit hook (reduce-shaped) |
| 8 | ResNet-50, async int8 hook | 12 | none or shape | 16-bit hook |
| 9 | Transformer, native FP32, tau = 4 with barrier | 12 | coordination (new family) | tau = 4 without barrier (E3) |
| 10 | ResNet-50, native FP32, tau = 4 with barrier | 12 | coordination or none | same |
| 11 | Transformer, FP16 dense | 24 | none (new baseline) | none expected; scored as abstention |
| 12 | ResNet-50, CPU-staged int8 | 12 | placement (new platform) | GPU-resident int8 (arm A) |

Cases 5, 6, and 12 use bundled interventions (they change selection or
placement work as well as the collective); the table labels them so and
they are reported separately from the isolated-shape cases 1 to 4.

Cases 9 and 10 change optimization semantics (tau) and are labeled
execution-cost only.

Ablations A1, A2, A3 are scored on the same twelve M1 records.

## Job ledger (Lonestar6 gpu-a100, 3 SU per node-hour, 15-minute floor)

Rate confirmed from the TACC accounting page cited by the reviewer.
Every job carries --nice=10000 and runs only when MOSAIC leaves nodes
idle. One job at a time.

| Job | Nodes | Wall | Contents | SU (floor-aware) |
|---|---|---|---|---|
| J1 pilot | 4 | 0:20 | v4 smokes (arm B, fp16, tau-no-barrier, verify-equivalence on arm A and arm B); async int8 hook smoke | 3.0 |
| J2 E1+E2a at 12 | 4 | 0:40 | FP32 and FP16 dense, arm A, arm B, async int8 hook, 16-bit hook, both workloads, phase + wall; Kineto trace on two conditions | 6.0 |
| J3 E1+E2a+E4 at 24 | 8 | 0:40 | same set at 24 ranks with peak-memory recording and a bounded-live-bucket gather variant | 12.0 |
| J4 E3+E5 at 12 | 4 | 0:40 | tau sweep with and without barrier, both workloads; 400-iteration tails for four conditions | 6.0 |
| J5 E6 interventions | 4 | 0:30 | the M2/M3/M4 runs for cases 3, 5, 8, 9, 10, 12 (M1 for these come from J2/J4 within the same allocation family; re-run M1 here when the allocation differs) | 4.5 |
| reserve | | | one repeat of J2 or a failed J3 rerun | 3.5 |
| total | | | | 35.0 |

Constraint: J3 is the only 8-node job and the only one that can exceed
the budget if it fails; it runs after J2 succeeds and its condition
list is trimmed to fit 40 minutes based on J2 timings. E6 cases at 24
ranks (1, 2, 4, 6, 7, 11) take their M1 to M4 from J3 in the same
allocation; the 12-rank cases from J2, J4, J5. If J3 cannot be
afforded, cases 1, 2, 4, 6, 7, 11 are dropped and the paper reports six
cases.

## What is deferred, per the review

int16 native arm (NCCL has no int16 reduction): dropped. E2b THC
saturating ring (non-associative, order-dependent): deferred; cited,
not implemented. 1B workload on LS6 (40 GB A100s): not attempted; its
Intel results stay labeled instrumented-only.
