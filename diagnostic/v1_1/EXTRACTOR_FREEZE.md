# The feature extractor after the freeze (RETROSPECTIVE, 2026-09-29)

RULES_FROZEN.sha256 (frozen 2026-09-21 09:35:42 UTC, commit 35cf9be,
pushed 09:39:45 UTC) lists features.py a3eef19d. The extractor that
produced features_r4.json and E6_FINAL.txt is 95e240d8, changed in
6955df2 (pushed 19:01:27 UTC) and ef3ed2d (pushed 20:12:43 UTC). The first
scored held-out job, j2 (3459786), ran 17:25-17:37 UTC, before either
change. rules.py (71329c33) and the E6 case list and named interventions
did not change.

Changes (git diff 35cf9be HEAD -- diagnostic/features.py):
1. gpu-quant-gathersum added to GATHER_MODES. The arm is an all-gather;
   the rule input "kind" is defined by the method's collective.
2. fp16-native added to REDUCE_MODES (no effect: reduce is the default).
3. The round-4 tag R.
4. The per-run-directory calibration propagated to every row of the run
   (the frozen launch script, also in 35cf9be, calibrates once per
   allocation and reuses it).
5. Wall-timed, tau, barrier, and allocation keys. The frozen extractor
   keeps phase rows only (literal_frozen_extractor.py: 0 wall rows), so
   E6 cannot be scored with it.

With the frozen extractor's classification and calibration applied to the
M1 rows, and the wall rows supplied:

| Rule set | As scored (E6_FINAL) | Extractor as frozen |
|---|---|---|
| Full procedure | 4 correct, 1 short | 2 correct, 1 short |
| A1 traffic ratio | 5 correct | 3 correct |
| A2 exchange-to-reference | 7 correct | 0 correct |
| A3 phase shares | 4 correct, 1 short, 1 null | unchanged |

No threshold or rule changed. The ranking of the rule sets that the paper
reports depends on changes 1 and 4.
