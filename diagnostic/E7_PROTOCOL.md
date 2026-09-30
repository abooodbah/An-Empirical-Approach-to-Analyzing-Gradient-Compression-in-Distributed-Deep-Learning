# E7: specificity of the frozen diagnostic rules (pre-registered)

Written and pushed before any E7 job was submitted. Nothing in this file
or in score_e7.py changes after submission; a later change is a new
version and the E7 result is then reported as retrospective.

## 1. Question
Do the rule sets avoid recommending changes that do not pay? E6 offered
almost no neutral or harmful interventions.

## 2. Platform and protocol
Lonestar6 gpu-a100, 2 nodes x 3 A100-PCIe (p = 6), NCCL with
NCCL_NET_GDR_LEVEL=SYS, as in E6. Harness v4 (the round-4 harness).
Per model: one calibration run (resnet50 native, paper profile, batch 32,
40 iterations, 10 warm-up), then M1 (phase) for each original and M2/M3
(wall) for each original and intervention, extension profile, batch 128,
40 iterations, 10 warm-up, all inside one allocation. Two allocations
(jobs e7a and e7b). After every scored run, e7a also runs eight trace
conditions: both models, the two pairs of E8 modes, harness v5.1,
qualification profile, batch 128 (the E7 configuration). They are never scored in E7. They are the development
data for the reference of a later experiment (E8), which is evaluated
only on configurations E7 does not measure.

## 3. Cases (per model, transformer and resnet50)
| Suffix | Original | Intervention | Family |
|---|---|---|---|
| 1 | gpu-quant-gathersum | gpu-quant-reducible | isolated shape (identical encoding) |
| 2 | gpu-quant | gpu-quant-reducible | shape, rank-local substitution |
| 3 | gpu-topk | gpu-quant-reducible | bundled placement |
| 4 | cpu-quant | gpu-quant | isolated placement (positive control) |
| 5 | ddp-int8-hook | ddp-fp16-hook | shape under overlap |
| 6 | fp16-native | none | abstention expected |
None of these configurations has been measured on Lonestar6 at p = 6.

## 4. Scoring (fixed)
The frozen rules and the E6 outcome rule: correct if delta >= max(0.05,
0.5 e); short if 0.05 <= delta < 0.5 e; null if |delta| < 0.05; harmful
if delta <= -0.05; abstentions checked against whether the case's
intervention paid. Each allocation's verdict comes from its own M1; trend
is None. A case counts 1/2 in each allocation.

## 5. Labels added to E6's
inapplicable: a shape verdict on an exchange already reduce-shaped.
not tested: the verdict names a mechanism the case's intervention does not
test. missing (run failed) and missing (M1 failed): a condition failed;
it is reported, never rerun. A job that completes no timed run (preflight
or calibration failure) may be resubmitted once.

## 6. Readings reported
Primary: frozen. Secondary: described (diagnostic/v1_1/rescore.py C1+C2).

## 7. Expectations (stated now; not used in scoring)
The shape rule and A1 cannot fire at p = 6 (r_V = 0.75). A2 has no share
gate and may fire on the gather cases, where the step is compute-heavy.
Case 4 should clear 5%.

## 8. Frozen components
- sha256 71329c336cc3e9646dbb6e6dd4ce20e12305d666c663f2a3e4b35195b535a822 `diagnostic/rules.py`
- sha256 95e240d86e79301e6740276f851a30bf6c9289c093cdc5147de92117bed1c7dd `diagnostic/features.py`
- sha256 928550323d0a5f8e9f48edc66c0b71fbd17489b4ae827afc3cf5d55a5dc9fede `diagnostic/score_e6.py`
- sha256 bece6f877f38a19b3ab815b3e05b2be2816023af8d89a4f67dc950d96fdd125c `diagnostic/score_e7.py`
- sha256 246d3f929b2b26978fe5af910177a054d58c4ec7876ea30c61a6ddadf3423dea `diagnostic/v1_1/rescore.py`
- sha256 e81bc428d364caa3be7bc8789a5bfd6eb06897864fe4ed21b4a7e3b10a2c2c88 `harness/control_workload_benchmark_v4.py`
- sha256 b920424cfa25bdd751491e9dd80fcb30898662e8c600a81331d2f75e20352f6b `validation_v2/harness/control_workload_benchmark_v5_1.py`
