# E12 result (scored 2026-10-18)

Jobs on Lonestar6 `gpu-a100-dev`, 2 nodes each: e12q 3518376 (qualification, decision go-40),
e12a 3518534 (36 min 34 s) and e12b 3518535 (32 min 21 s). Raw run directories:
`raw/e12/*.tgz` with their sha256 files. The freeze check (`sha256sum -c diagnostic/E12_FREEZE.sha256`)
was clean before scoring.

Scored as protocol section 9, step 6:

```
python diagnostic/e12/score_e12.py --qual e12-q-2n-3518376 e12-e12a-2n-3518534 e12-e12b-2n-3518535 \
    --json diagnostic/E12_RESULT.json
python diagnostic/e12/score_e12.py --qual e12-q-2n-3518376 e12-e12a-2n-3518534 e12-e12b-2n-3518535 \
    --reading described --json diagnostic/E12_RESULT_described.json
```

Printed output: `diagnostic/e12/E12_SCORE_frozen.txt` and `E12_SCORE_described.txt`. The two readings
give the same counts.

Truth "pays" (g >= 0.05), 30 cases of weight 1 (12 paying, 18 not):

| Method | Recommended | True pos. | False pos. | Precision | Recall | FPR |
|---|---|---|---|---|---|---|
| Full procedure | 2 | 2 | 0 | 1.00 | 0.17 | 0.00 |
| Traffic-ratio check (A1) | 0 | 0 | 0 | - | 0.00 | 0.00 |
| Benchmark check (A2) | 4 | 4 | 0 | 1.00 | 0.33 | 0.00 |
| Phase-share check (A3) | 2 | 2 | 0 | 1.00 | 0.17 | 0.00 |
| Replay | 6 | 6 | 0 | 1.00 | 0.50 | 0.00 |
| Always recommend | 30 | 12 | 18 | 0.40 | 1.00 | 1.00 |
| Cost model | 14.5 | 9.5 | 5 | 0.66 | 0.79 | 0.28 |

Replay on its 12 covered rows (01, 04, 12 and mirrors M01, M04, M12 on both workloads): it recommended
all 6 paying changes and none of the 6 harmful ones. Median |e - g| 0.065 over its 12 recommendations
(95% bootstrap interval 0.046 to 0.101) and 0.061 over its 24 predictions (0.040 to 0.080). It
overpredicted the gain of every paying change. Its largest errors (0.13) are TM04 and R12. On the
other 18 cases it has no substitution model and abstains.

Quality-constrained truth: the replay keeps precision 1.00 and false-positive rate 0.

rho (phase over block step, matched arms): 1.04 to 1.19; the phase-timed matched gain is 0.014 below
the block-timed gain on both workloads in allocation e12a.

Replay fidelity (no substitution, reported only): on the BF16 traces the replay predicts the step
1.47 to 1.60 times the block value; on most FP32 traces within 6% (ResNet-50 dense DDP is 1.31).

All 60 pre-stated expectation rows held.

Figure: `figures/make_fig_replay.py` draws predicted against measured speedup for the replay's
predictions on E8 (15) and E12 (24): `python figures/make_fig_replay.py . figures`.
