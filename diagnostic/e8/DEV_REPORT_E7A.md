# E8 reference: development report on the e7a traces

The frozen reference (diagnostic/e8/E8_FREEZE.md) applied, before any E8 record
was read, to the e7a traces (p = 6, batch 128). Calibration: e7a, alpha 3e-05 s,
beta 9.534e-11 s/byte. Ground truth: E7's wall-timed pairs.

| family | model | g_pred | allocation | T original (ms) | T candidate measured (ms) | predicted (ms) | error |
|---|---|---|---|---|---|---|---|
| G | transformer | 0.222 | rev-e7a-2n-3479985 | 34.91 | 28.94 | 27.16 | -6.1% |
| G | transformer | 0.222 | rev-e7b-2n-3479986 | 36.30 | 28.92 | 28.24 | -2.4% |
| G | resnet50 | 0.117 | rev-e7a-2n-3479985 | 170.77 | 158.80 | 150.82 | -5.0% |
| G | resnet50 | 0.117 | rev-e7b-2n-3479986 | 170.50 | 158.61 | 150.58 | -5.1% |
| H | transformer | 0.193 | rev-e7a-2n-3479985 | 33.51 | 28.33 | 27.04 | -4.6% |
| H | transformer | 0.193 | rev-e7b-2n-3479986 | 35.31 | 29.58 | 28.48 | -3.7% |
| H | resnet50 | 0.152 | rev-e7a-2n-3479985 | 176.19 | 161.80 | 149.40 | -7.7% |
| H | resnet50 | 0.152 | rev-e7b-2n-3479986 | 175.76 | 159.95 | 149.04 | -6.8% |

## Gate G2

Gather family (G): median absolute error 5.0% over 4 points; G2 passes on these points.
Hook family (H): median absolute error 5.7% over 4 points; G2 passes on these points.
