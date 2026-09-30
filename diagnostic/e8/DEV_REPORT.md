# E8 reference: development report (V1, hook family)

Frozen with the reference. Written by `diagnostic/e8/dev_v1.py` from the
artifact's archives only. Calibration: V1 cal-start, alpha 1.28e-05 s, beta 1.025e-10 s/byte.

## Replay of the traced originals and candidates

| trace | replayed step (ms) | traced step (ms) | ratio |
|---|---|---|---|
| ddp-int8-hook batch 32 rank 0 | 25.86 | 25.98 | 0.995 |
| ddp-int8-hook batch 32 rank 3 | 25.86 | 26.01 | 0.994 |
| ddp-fp16-hook batch 32 rank 0 | 12.25 | 11.98 | 1.023 |
| ddp-fp16-hook batch 32 rank 3 | 12.26 | 12.07 | 1.015 |
| ddp-int8-hook batch 128 rank 0 | 40.00 | 40.54 | 0.987 |
| ddp-int8-hook batch 128 rank 3 | 39.84 | 40.38 | 0.987 |
| ddp-fp16-hook batch 128 rank 0 | 27.00 | 27.20 | 0.993 |
| ddp-fp16-hook batch 128 rank 3 | 27.82 | 27.95 | 0.995 |

## Buckets found in the int8-hook traces

- batch 32, rank 0: 16 hook regions in 8 iterations, bucket sizes 1051138, 10374656 elements
- batch 128, rank 0: 16 hook regions in 8 iterations, bucket sizes 1051138, 10374656 elements

## Predictions against the block-timed ground truth

Predicted candidate step = measured original step x (1 - g_pred), per allocation.

| batch | g_pred | allocation | T int8 (ms) | T fp16 measured (ms) | T fp16 predicted (ms) | error |
|---|---|---|---|---|---|---|
| 32 | 0.579 | V1 | 25.31 | 12.10 | 10.66 | -11.9% |
| 32 | 0.579 | V2 | 24.32 | 11.99 | 10.24 | -14.6% |
| 32 | 0.579 | V3 | 25.62 | 12.18 | 10.79 | -11.4% |
| 128 | 0.359 | V1 | 44.31 | 30.48 | 28.39 | -6.8% |
| 128 | 0.359 | V2 | 44.71 | 30.08 | 28.65 | -4.8% |
| 128 | 0.359 | V3 | 45.43 | 29.79 | 29.11 | -2.3% |

## Gate G2

Hook family (H): median absolute error 9.1% over 6 development points; G2 passes (limit 15%).

Gather family (G): no development trace exists at the freeze. The reference is
reported for G as "not validated on development data" unless the e7a traces
arrive before any E8 record is read; the frozen code is then applied to them and
the errors go to DEV_REPORT_E7A.md.

## Known bias (reported, not corrected)

The substituted all-reduce is costed from the isolated calibration, as section 7
of the protocol fixes. The V1 fp16 traces, used here only as a diagnostic, show
the in-training all-reduce kernels running longer than the calibration predicts:

| batch | bucket elements | real all-reduce (us) | calibrated (us) | ratio |
|---|---|---|---|---|
| 32 | 1051138 | 541 | 228 | 2.37 |
| 32 | 10374656 | 3551 | 2140 | 1.66 |
| 128 | 1051138 | 946 | 228 | 4.14 |
| 128 | 10374656 | 4248 | 2140 | 1.99 |

The reference therefore predicts a cheaper candidate than the one measured, and
its gains are optimistic; the error table above includes this bias.
