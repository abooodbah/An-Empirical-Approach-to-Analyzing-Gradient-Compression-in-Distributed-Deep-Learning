# A-plus campaign (E12): preparation progress

Branch `a-plus-campaign`, local only. No cluster contact, no push.

## Done
- Branch created from main (cc6b488).
- Background read: E7, E8 protocols and results, E8 replay and scorer, rules.py, score_e7.py, E11 note,
  ls6_e11b.slurm and e11_sets.sh, SU tables, per-condition durations from the e8s, e9 q12a and e11b timelines.
- Baseline test suite on main before any change: 406 passed.
- Harness v5.5 (validation_v2/harness/build_v55.py): --dtype bf16 and hook fidelity; 19 tests; three
  mutations each fail a named test. Tests committed first.
- e12_sets.sh, make_ls6_e12.py -> ls6_e12q/a/b.slurm with a freeze check; 8 launch tests.
- Scorer diagnostic/e12/score_e12.py with 43 tests on synthetic records; 12 mutations each fail a test.
- Qualification go/no-go diagnostic/e12/qual_check.py with 7 tests.
- diagnostic/E12_PROTOCOL.md, diagnostic/e12/make_freeze.py, diagnostic/E12_FREEZE.sha256, 3 freeze tests.
- A_PLUS_SUMMARY.md.
- Full suite on the branch: 486 passed, 0 failed.

## In progress
- Nothing.

## Remaining (author)
- Review; after any edit regenerate scripts and freeze, rerun tests; push; stage; submit e12q, apply
  qual_check.py, submit e12a and e12b.
