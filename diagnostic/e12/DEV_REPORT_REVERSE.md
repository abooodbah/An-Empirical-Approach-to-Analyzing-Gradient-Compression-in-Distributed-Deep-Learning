# E12 reverse substitutions: development report on the e7a traces

diagnostic/e12/replay_reverse.py applied to the e7a traces (p = 6, batch 128),
before any E12 job exists. Calibration: e7a, all-gather alpha 0.000109 s, beta 5.864e-10 s/byte.
Ground truth: E7's wall-timed pairs. Error = (predicted - measured) / measured candidate step.

| family | model | g_pred | allocation | T original (ms) | T candidate measured (ms) | measured g | predicted (ms) | error |
|---|---|---|---|---|---|---|---|---|
| G-rev | transformer | -0.163 | rev-e7a-2n-3479985 | 28.94 | 34.91 | -0.206 | 33.67 | -3.6% |
| G-rev | transformer | -0.163 | rev-e7b-2n-3479986 | 28.92 | 36.30 | -0.255 | 33.65 | -7.3% |
| G-rev | resnet50 | -0.029 | rev-e7a-2n-3479985 | 158.80 | 170.77 | -0.075 | 163.49 | -4.3% |
| G-rev | resnet50 | -0.029 | rev-e7b-2n-3479986 | 158.61 | 170.50 | -0.075 | 163.29 | -4.2% |
| H-rev | transformer | -0.242 | rev-e7a-2n-3479985 | 28.33 | 33.51 | -0.183 | 35.18 | +5.0% |
| H-rev | transformer | -0.242 | rev-e7b-2n-3479986 | 29.58 | 35.31 | -0.194 | 36.73 | +4.0% |
| H-rev | resnet50 | -0.137 | rev-e7a-2n-3479985 | 161.80 | 176.19 | -0.089 | 183.91 | +4.4% |
| H-rev | resnet50 | -0.137 | rev-e7b-2n-3479986 | 159.95 | 175.76 | -0.099 | 181.80 | +3.4% |

## Summary

G-rev: median absolute error 4.2% over 4 points (largest 7.3%); the replay's recommendation agrees with the measured outcome on 4 of 4 points.
H-rev: median absolute error 4.2% over 4 points (largest 5.0%); the replay's recommendation agrees with the measured outcome on 4 of 4 points.
