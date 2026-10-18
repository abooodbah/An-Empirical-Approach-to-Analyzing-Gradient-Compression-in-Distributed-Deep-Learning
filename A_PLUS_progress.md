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

### Mirror cases (second session, 2026-10-18)
- Cases M01 (reducible -> gathersum), M04 (fp16 hook -> int8 hook), M12 (the same in BF16), both workloads,
  expected harm. Replay coverage now 6 case types per workload: 3 help, 3 harm.
- diagnostic/e12/replay_reverse.py: reverse substitutions G-rev and H-rev on top of the frozen E8 reference
  (imported, unchanged; diagnostic/e8/test_freeze.py still passes).
- Development on the e7a traces only (diagnostic/e12/dev_reverse_e7a.py -> DEV_REPORT_REVERSE.md): median
  absolute error 4.2% (G-rev) and 4.2% (H-rev), 4 points each, largest 7.3% and 5.0%; 8 of 8 outcome calls
  right. One development change before the first run (G-rev redirects only the all-reduce kernel); no change
  after seeing the errors.
- test_replay_reverse.py: 12 tests (hand-computed makespans for both families). Mutations, each caught:
  final cast 8/5 factor removed (test_reducible_to_gathersum_by_hand); decode no longer waits for the gather
  copies (that test and test_gathersum_decode_waits_for_the_copies); int8-hook ops no longer wait for the
  gather copies (test_fp16_to_int8_hook_by_hand, test_h_rev_gain_falls_as_the_gather_slows).
- Scorer: mirror rows, reverse predictor (full calibration), replay metrics over its covered rows.
- Run lists: every mirror tag was already block-timed and traced. Added: M1 of ddp-fp16-hook and
  ddp-fp16-hook-bf16 (4 runs per scored job, 76 s; the rules and the cost model need the original's M1), and
  traces of ddp-fp16-hook-bf16 in e12q (2 runs, 42 s; parse check of H-rev on BF16 by qual_check.py).
- SU: e12q 24 conditions 10.2 min 1.02 SU; e12a/b 86 conditions 34.9 min 3.49 SU each; total 8.0 SU (was 7.7).
- Scripts regenerated, freeze regenerated (24 files). Commits d6be4df (tests), 470f807 (code), then the
  summary and this log.
- Full suite: 504 passed, 0 failed (480 in the protocol's five directories, 24 in environment, provenance,
  validation_v2/sensitivity).

## In progress
- Nothing.

## Remaining (author)
- Review; after any edit regenerate scripts and freeze, rerun tests; push; stage (commands in
  A_PLUS_SUMMARY.md); submit e12q, apply qual_check.py, submit e12a and e12b.
