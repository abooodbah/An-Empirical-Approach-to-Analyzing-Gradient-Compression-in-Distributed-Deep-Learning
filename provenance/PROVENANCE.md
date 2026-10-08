# Provenance and chronology (compiled 2026-09-30)

All times UTC. Sources: provenance/github_push_events.json (GitHub push
events, retained 90 days by GitHub, captured here), provenance/
lonestar6_jobs_utc.tsv (Slurm accounting), and git.

## E6 held-out evaluation
| Time | Event |
|---|---|
| 2026-09-21 09:35:42 | rules v1 and E6 protocol frozen (RULES_FROZEN.sha256, commit 35cf9be) |
| 2026-09-21 09:39:15 | pilot job 3459209 submitted (qualification runs; never scored) |
| 2026-09-21 09:39:45 | 35cf9be pushed to GitHub |
| 2026-09-21 17:25:04 | j2 3459786 submitted (first scored held-out job) |
| 2026-09-21 18:53:34 | j3 3459984 submitted (24 ranks) |
| 2026-09-21 19:01:27 | extractor change 6955df2 pushed (see diagnostic/v1_1/EXTRACTOR_FREEZE.md) |
| 2026-09-21 19:34:37 | j4 3460161 submitted |
| 2026-09-21 19:47:56 | j5 3460182 submitted |
| 2026-09-21 20:12:43 | extractor change ef3ed2d pushed |
| 2026-09-21 20:12:43 | ef3ed2d also edits harness v4 (a PowerSGD-only guard and a schema bump); every round-4 job ran the earlier file, 3aecaae1c1021269 (git 35cf9be) |

## v2 validation campaign
| Time | Event |
|---|---|
| 2026-09-23 15:25 | analysis rules committed (218ddee, local) |
| 2026-09-23 15:26:30 | jobs 3465525-3465529 submitted; job-time checksums cover harness, manifests, run lists (not the analysis rules) |
| 2026-09-26 | jobs ran; analysis |
| 2026-09-26 | R1 3471881 submitted with harness v5.1 (retrospective fix, parameters-only hash) |
| 2026-09-30 01:21:37 | 218ddee and the records pushed to GitHub |

## Revision 3: E7, the 24-rank replication, and E8
| Time | Event |
|---|---|
| 2026-09-30 06:46:37 | E7 pre-registration (5980d80, harness pin 7ff88a0) and the audit commits pushed to GitHub (2b02a48..4ad1e1b) |
| 2026-09-30 06:47:43 | window 1 submitted: e7a 3479985, e7b 3479986, j3r 3479987 and 3479988, sensitivity 3479989 (provenance/rev_jobs.tsv) |
| 2026-09-30 07:01:02 | E8 pre-registration (f84cee6: diagnostic/E8_PROTOCOL.md, e8_sets.sh, ls6_e8.slurm) pushed to GitHub |
| 2026-09-30 07:01:41 | E8 submitted in window 1: e8s 3480007 and 3480008, e8l 3480009 and 3480010 |
| 2026-09-30 09:27:12 | E8 analysis frozen and pushed (27b8bb4: diagnostic/e8/E8_FREEZE.md); every E8 job still pending |
| 2026-09-30 21:41 to 2026-10-02 17:36 | all nine jobs ran and completed (provenance/rev_su.tsv: 20.3 SU) |
| 2026-10-03 05:48 | e7a development errors of the frozen E8 reference (DEV_REPORT_E7A.md) and the window-1 results pushed (fe2c2f5); E8 records fetched and scored afterward |

## Revision 4: E9 and E10
| Time | Event |
|---|---|
| 2026-10-08 00:14:19 | E9 and E10 pre-registration pushed to GitHub (fce4c1b..78b077e: harness v5.2, diagnostic/E9_PROTOCOL.md and E10_PROTOCOL.md, both scorers, the launch scripts, and the freeze hashes diagnostic/E9_FREEZE.sha256 and E10_FREEZE.sha256). The time is the repository's `pushed_at` and the push in its activity list (provenance/github_activity_rev4.json); the author approved the push at 00:13 |
| 2026-10-08 00:15:54 | E10 submitted on Stampede3: s2 3579138 and s1 3579139 (provenance/rev4_jobs.tsv) |
| 2026-10-08 00:16:07 | E9 submitted on Lonestar6 with `--nice=100`: q12a 3495229, q12b 3495230, q24a 3495231, q24b 3495232 |
| 2026-10-08 00:15:57 to 01:06:06 | E10 s2 3579138 (32 ranks) ran 3009 s to its time limit. Both check runs and the probe returned 0; the timed gather stalled after initialization and no timed run completed. TACC drained node c552-003 with "GPU missing" at 01:06:50 |
| 2026-10-08 01:08:32 to 01:46:41 | E10 s1 3579139 (16 and 8 ranks) ran 2289 s and completed all six timed conditions |
| 2026-10-08 01:49 | both E10 run directories fetched and scored (validation_v2/results/rev4/E10_RESULT.json). Scored with the files of E9_FREEZE.sha256 and E10_FREEZE.sha256, unchanged. Charges: provenance/rev4_su.tsv (13.84 SU) |

## Harness v4 used by E6
Every round-4 job recorded harness v4 3aecaae1c1021269ae25a9e0468a7c7a6bcae7c27a7f64c8f275531e10207179,
which is git 35cf9be:harness/control_workload_benchmark_v4.py and the Lonestar6 file dated
2026-09-21 09:38 UTC. The current artifact file differs (ef3ed2d). E7 is pinned to 3aecaae1.

## Retrospective analyses (devised after the outcomes they analyze)
- validation_v2/analysis/analyze_r1.py (first committed 2026-09-30 01:20:56, 2b02a48)
- diagnostic/v1_1/* (first committed 2026-09-30 05:55:56, 51abcc3)
- validation_v2/analysis/identity.py (first committed 2026-09-30 06:00:45, a4be38d)

## File hashes
- sha256 350b3555c94a5bc26924155cd9915a74d46c60be5db6a72fa4fa10e521033fcf `raw/rev4/redctl-E10-2n-3579139.tgz`
- sha256 9a59840f76909c13b4d6849a936dde3d716fdf6d9db1ae81ce7b005c952c6f72 `raw/rev4/redctl-E10-4n-3579138.tgz`
- sha256 71329c336cc3e9646dbb6e6dd4ce20e12305d666c663f2a3e4b35195b535a822 `diagnostic/rules.py`
- sha256 95e240d86e79301e6740276f851a30bf6c9289c093cdc5147de92117bed1c7dd `diagnostic/features.py`
- sha256 928550323d0a5f8e9f48edc66c0b71fbd17489b4ae827afc3cf5d55a5dc9fede `diagnostic/score_e6.py`
- sha256 0c85be55c341f372d8fd311f53da853f99d551196bc808e1f3cc62a1495b7553 `diagnostic/E6_FINAL.txt`
- sha256 23e87a0bd6f3752a07a7ebd71ffaac63395845dc7ec743949561a6c7ca161d9f `diagnostic/E6_PROTOCOL.md`
- sha256 3299d5ffd57380ac715ca9a111a4f65e574301481a263953beb45d6b8b1caefa `validation_v2/ANALYSIS_RULES_v2.md`
- sha256 06ae0a65164052870e4943e95d281bb1b1e926e11f79f01dd89979e36fb5559a `validation_v2/analysis/analyze_v2.py`
- sha256 10ceedfacbdd37b67213949cd66beacaca9c28a7230ef51c9069a817e67d718a `validation_v2/analysis/analyze_r1.py`
