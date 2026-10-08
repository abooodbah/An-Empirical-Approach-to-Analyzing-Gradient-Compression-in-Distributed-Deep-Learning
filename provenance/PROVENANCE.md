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
| 2026-10-08 02:23:55 | The nine commits from 0fc68d1 to 22a09e4 pushed to GitHub (78b077e..22a09e4; the time is the repository's `pushed_at`). The author approved the push |
| 2026-10-08 02:26:39 | Recorded and pushed before the resubmission of E10 s2 (commit 516973c; the time is the push in provenance/github_activity_rev4.json. That commit's copy of this row carried 02:32 and "at 02:27", times written before the clock was read; they are corrected here). Job 3579138 completed no timed run, and section 4 of diagnostic/E10_PROTOCOL.md allows one resubmission if the Stampede3 ledger allows. At the protocol's 20 SU the ledger refuses a job long enough for the three 32-rank conditions (committed 13.84 SU; a 40-minute job on 4 nodes is bounded at 8.00 SU; `harness/launch/su_ledger.py` prints REFUSED at cap 20 and ALLOWED at cap 22). The author approved the full set on 2026-10-07 after being told the total (about 20.8 SU expected, 21.84 at most). This is a deviation from section 6, the budget, and from nothing else: the set (s2), its order and iteration counts, the staged files (their hashes compared with diagnostic/E10_FREEZE.sha256 before this record was pushed), and the scorer are unchanged. The time limit is 40 minutes where the first submission had 50, and node c552-003 is excluded |
| 2026-10-08 02:26:47 | E10 s2 resubmitted on Stampede3: 3579496 (4 nodes, 40-minute limit, c552-003 excluded; provenance/rev4_jobs.tsv). The time is the cluster's clock at submission, eight seconds after the push above |
| 2026-10-08 02:34:14 to 03:05:51 | E10 s2 resubmission 3579496 (32 ranks, nodes c551-002 and c552-[005-007]) ran 1897 s and completed its two check runs, the probe, and all three timed conditions. Checked by hand over every rank and row: 20 measured iterations on each of 32 ranks, `final_param_hashes_equal` True, the three timed runs start from the same model hash, and the equivalence run is True. The times are sacct's |
| 2026-10-08 03:09:40 | Its run directory fetched (raw/rev4/redctl-E10-4n-3579496.tgz; the time is the archive's file time) and all three E10 run directories scored again into validation_v2/results/rev4/E10_RESULT.json. The files of diagnostic/E10_FREEZE.sha256 were verified unchanged before the records were read. The result keeps both 32-rank allocations; the first has state `missing`. Charges: provenance/rev4_su.tsv, 20.17 SU over the three Stampede3 jobs, 0.17 SU above the 20 SU of section 6 |
| 2026-10-08 14:15 | Records corrected after an independent review of fce4c1b..8367fd9. The SU record is now written in UTC from `provenance/rev4_sacct_stampede3.txt` (sacct with TZ=UTC); its first version printed the cluster's local time under `start` and `end`. It is computed from sacct's elapsed time (20.168 SU); the system's own accounting is the record of charge. As a check, taccinfo showed the project at 69 SU at 02:12 and at 48 SU at 14:15 (it prints whole SU), and no other job of the project was in the Stampede3 queue at 02:24. The drain of node c552-003 is kept as read in `provenance/stampede3_node_c552-003.txt`; it is recorded 44 s after job 3579138 ended, and no record establishes that it caused the stalled run |
| 2026-10-08 14:15 | Retrospective corrections from the same review, none in a frozen file: the memory arithmetic counts p + 1 int8 copies and reproduces the measured eight-rank peak (`diagnostic/v1_2/MEMORY_ACCOUNT.md`); a rate of exactly 0 or 1 carries its count and no resampling interval (`diagnostic/v1_2/METRICS.*`); `environment/PLATFORMS.md` says that only the Lonestar6 jobs record node facts; `reproduce_diagnostic.sh` rebuilds and compares every generated file of Revision 4; `provenance/test_rev4_records.py` guards the E9 and E10 freeze lists and the order of pushes and submissions |
| 2026-10-08 14:15 | Found by the review, older than this revision: the int8 hook issued its collectives synchronously on NCCL as well as on oneCCL in every archived run. The harness chooses the synchronous path where `"ccl"` is in the backend name, and `nccl` contains it (v3 line 1120, v4 1137, v5 1279, v5.1 1297, v5.2 1422). All 100 rank-0 int8-hook provenance records on NCCL and all 10 on oneCCL carry `hook_state.mode` = `sync`; none says `async`. Harness v5.2 is frozen, so the E9 jobs will record the same. No measurement in this artifact covers the int8 hook with asynchronous collectives |
| 2026-10-08 16:57:45 to 18:28:05 | E9 ran on Lonestar6, every job COMPLETED (sacct, UTC): q12a 3495229 from 16:57:45 to 17:24:01 (1576 s), q24a 3495231 from 17:25:31 to 17:52:29 (1618 s), q24b 3495232 from 17:53:18 to 18:11:31 (1093 s), q12b 3495230 from 18:12:23 to 18:28:05 (942 s). Five runs failed at start-up with a rendezvous or connection error and recorded nothing: in q12a the fidelity run of resnet50 gpu-quant and pass 1 of resnet50 gpu-quant-reducible; in q12b pass 1 of resnet50 fp16-native; in q24a the fidelity run of resnet50 gpu-quant-reducible; in q24b pass 2 of transformer ddp-int8-hook. Nothing was rerun. Three conditions therefore have one valid pass in one allocation, and two fidelity values are absent |
| 2026-10-08 18:34:58 to 18:38:40 | The four run directories fetched (raw/rev4/e9-*.tgz; the times are the archives' file times), each after sacct showed a final state for its job, and scored with the frozen score_e9.py into validation_v2/results/rev4/E9_RESULT.json. The files of diagnostic/E9_FREEZE.sha256 were verified unchanged before the records were read. Checked by hand over every rank file: every outcome run present has status PASS, final_param_hashes_equal True and 100 measured iterations; both equivalence runs record equivalence_ok True in every allocation; the DDP bucket structure is identical for the three DDP modes in every allocation; NCCL rejected an int16 all-reduce ("Input tensor data type is not supported for NCCL process group: Short"). The int8 hook's provenance says mode sync in all four, as expected. Charges: provenance/rev4_su.tsv, 26.467 SU on Lonestar6 (cap 55) and 46.635 SU over both systems |
| 2026-10-08 18:34:58 to 18:38:40 | Node facts of the E9 jobs (provenance/node_facts.*.txt in each archive, 24 node records): NVIDIA A100-PCIE-40GB, 40960 MiB, driver 570.195.03, VBIOS 92.00.25.00.08; two AMD EPYC 7763 64-core processors; kernel 4.18.0-553.150.1.el8_10; adapter mlx5_0 on NUMA node 0, GPU 0 on NUMA node 0 and GPUs 1 and 2 on NUMA node 1. The adapter is a ConnectX-6 (MT28908, firmware 20.32.2004) on 22 records; one node of each 24-rank job carries a BlueField-2 integrated ConnectX-6 Dx (firmware 24.33.1048). PyTorch 2.4.1+cu121, NCCL 2.20.5 |
| 2026-10-08 18:44:10 | POST HOC, written after the E9 records were read (the time is the output's file time): `validation_v2/analysis/e9_tree_posthoc.py` fits the forced-tree probe, which the frozen scorer collects and does not read, into `validation_v2/results/rev4/E9_TREE_POSTHOC.json` with the protocol's slope rule. Nothing frozen changed |
| 2026-10-08 19:09:24 | POST HOC, a second reading of the E9 probes (the time is the output's file time): `validation_v2/analysis/e9_probes_posthoc.py` into `validation_v2/results/rev4/E9_PROBES_POSTHOC.json`. Found while the paper's paragraph was being written from the scored result: the protocol's line passes through three points of the probe grid (1, 16 and 64 MiB), and at 24 ranks over GPUDirect the forced-ring all-reduce of 1 MiB took 10.26 and 10.64 ms, longer than the 16 MiB one (3.80 and 4.01 ms). That point lowers the fitted ring slope, which enters P1 and P2; the default probe's own 1 MiB point (3.32 and 3.25 ms, against 3.70 and 3.64 ms at 16 MiB) does the same to P3. The script repeats every probe ratio with the line through 16 and 64 MiB, prints the measured points, and reads the forced-tree probe. It replaces `e9_tree_posthoc.py` and `E9_TREE_POSTHOC.json` of commit 5573a1d, whose 24-rank ratio of the tree to the ring carried the same point; under the protocol's range its numbers are the frozen scorer's (a test compares them). Nothing frozen changed. The pre-registered outcome stands as scored: P1, P2 and P3 did not hold at 24 ranks over GPUDirect |
| 2026-10-08 19:09:24 | `validation_v2/analysis/e9_expectations.py` sets the protocol's expectations X1 to X5 and probe checks P1 to P3 against `E9_RESULT.json` and writes `E9_EXPECTATIONS.json` and `E9_EXPECTATIONS.md` (the time is the output's file time). X1, X2, X3 and X5 held. X4 did not: the 16-bit hook is slower than dense DDP on ResNet-50 at 24 ranks. P1 to P3 held at 12 ranks and over the host-staged transport and did not hold at 24 ranks over GPUDirect |
| 2026-10-08 19:27:35 | `environment/node_facts.py` reads `provenance/node_facts.N.txt` of the four E9 run directories and writes `environment/E9_NODE_FACTS.json`: every distinct value with the number of node records that carry it (the time is the output's file time). 24 records. Every record has the same GPU, driver, VBIOS, processors, kernel and placement; the adapter is a ConnectX-6 with firmware 20.32.2004 on 22 records and a BlueField-2 integrated ConnectX-6 Dx with firmware 24.33.1048 on 2. The platform table of the paper takes its Lonestar6 processor, driver, firmware and placement cells from this file |
| 2026-10-08 20:31:20 to 20:37:54 | A second independent review, of the E9 material, reproduced every number of the new passages from the raw records and found one false statement and six incomplete ones (report: `final_review_e9_2026-10-08.md` in the plan's workspace). Changed in the artifact, each with tests written first (the times are the outputs' file times): (1) `environment/node_facts.py` counts NODES, not records. The two jobs of a scale ran on the same nodes and the 24-rank jobs contain the 12-rank nodes, so 24 records are 8 nodes, and the adapter that differs (BlueField-2 integrated ConnectX-6 Dx, firmware 24.33.1048) is on one node, c309-001, in both 24-rank jobs and in neither 12-rank job. The earlier row above, "on 22 records ... on 2", is right about records and was printed in the paper as nodes. (2) `e9_probes_posthoc.py` reports two more lines that leave the 1 MiB point out and the prediction errors of each. Its docstring said the forced ring's point moves all three checks; it moves P1 and P2, and P3 is scored on the default probe, whose own 1 MiB point is high. (3) `e9_expectations.py` prints every line with a verdict per check, states how each expectation is read and that the readings were fixed after the records, reads X2 and X4 from the values instead of the calls (no outcome changes), and treats a ratio that is exactly on a bound as on it. (4) `e9_pass_spread_posthoc.py`, new and post hoc, keeps each pass of every condition. No frozen file and no scored number changed |

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
- sha256 fb5d9c7c467407687f6a9aa200a916898a6b42dced431f7d9a2ddf2b6980825d `raw/rev4/redctl-E10-4n-3579496.tgz`
- sha256 f35f1c1f1de84c9d99f90540d552bb47317a06f90be7baae6e598a17e8d75c16 `raw/rev4/e9-q12a-4n-3495229.tgz`
- sha256 48cb255050e76f4e897277b6011dec491a76aec0cc3e50696b951644ebe35f5f `raw/rev4/e9-q12b-4n-3495230.tgz`
- sha256 6e17c6f54d68da20e3a9febd96a6a47c2f292a4baee3b72f4eeeeb9d2cdca438 `raw/rev4/e9-q24a-8n-3495231.tgz`
- sha256 257d430e560e87cc8442aa1fd81f9cbe05368496e9b5140ef5c9bf2a510b3798 `raw/rev4/e9-q24b-8n-3495232.tgz`
- sha256 71329c336cc3e9646dbb6e6dd4ce20e12305d666c663f2a3e4b35195b535a822 `diagnostic/rules.py`
- sha256 95e240d86e79301e6740276f851a30bf6c9289c093cdc5147de92117bed1c7dd `diagnostic/features.py`
- sha256 928550323d0a5f8e9f48edc66c0b71fbd17489b4ae827afc3cf5d55a5dc9fede `diagnostic/score_e6.py`
- sha256 0c85be55c341f372d8fd311f53da853f99d551196bc808e1f3cc62a1495b7553 `diagnostic/E6_FINAL.txt`
- sha256 23e87a0bd6f3752a07a7ebd71ffaac63395845dc7ec743949561a6c7ca161d9f `diagnostic/E6_PROTOCOL.md`
- sha256 3299d5ffd57380ac715ca9a111a4f65e574301481a263953beb45d6b8b1caefa `validation_v2/ANALYSIS_RULES_v2.md`
- sha256 06ae0a65164052870e4943e95d281bb1b1e926e11f79f01dd89979e36fb5559a `validation_v2/analysis/analyze_v2.py`
- sha256 10ceedfacbdd37b67213949cd66beacaca9c28a7230ef51c9069a817e67d718a `validation_v2/analysis/analyze_r1.py`
