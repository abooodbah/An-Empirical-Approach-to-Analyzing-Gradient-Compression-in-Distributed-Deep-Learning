# E10: a bucketed gather on the 1B workload, with its memory account (pre-registered)

Fixed on 2026-10-07, before any E10 job was submitted.

## 1. Question
Does a gather that holds at most p received buckets at once run on the
1.05-billion-parameter workload at 16 and 32 ranks? What does it cost
against reduce-shaped exchanges of the same buckets, and what memory does
it hold?

## 2. Platform and protocol
Stampede3 pvc, 8 ranks per node, oneCCL over OFI/TCP with the collective
algorithms pinned as in the earlier 1B runs. Harness v5.2. Two allocations:
s1 (2 nodes) and s2 (4 nodes). Workload llama1b, batch 2, sequence 2048,
extension profile, phase timing.

Bucket: 16,777,216 elements in every bucketed mode. That is 64 MiB per
collective in FP32 and 16 MiB in int8, both inside the calibrated range.

s1, in this order: at 16 ranks gpu-quant-bucketed,
gpu-quant-reducible-bucketed, native-bucketed; at 8 ranks on the first node
gpu-quant (whole gradient) and gpu-quant-bucketed; at 16 ranks native. Each
has 40 measured and 10 warm-up iterations.
s2, at 32 ranks: gpu-quant-bucketed, gpu-quant-reducible-bucketed,
native-bucketed. Each has 20 measured and 5 warm-up iterations, because one
gather iteration is expected to last about a minute.

Before the timed runs of each job: an equivalence run of gpu-quant-bucketed
on the transformer at the job's rank count, a two-iteration run on the 1B
workload, and a probe of both collectives from 4 B to 256 MiB.

## 3. Outcomes
Step: the mean over iterations of the largest observed_step_s across ranks.
Exchange: the mean network_s pooled over ranks and iterations. Peak memory:
the largest peak_memory_mb across ranks. Buffer account:
exchange_buffer_bytes from the sidecar.
Ratios, within one allocation and rank count: gather to native-bucketed and
gather to reducible-bucketed, in exchange time and in step time, printed
beside p/2c and p/2.
Bucketing cost: gpu-quant-bucketed against gpu-quant at 8 ranks, and
native-bucketed against native at 16 ranks, both in s1.
Model check: the sum over buckets of (intercept + slope x bucket bytes)
from this allocation's probe, against the measured exchange, as a percent
error.
Memory check: the gather's peak minus native-bucketed's, against five bytes
per gradient element plus exchange_buffer_bytes.

## 4. Validity, and what is reported on failure
A run counts if final_param_hashes_equal is True and every measured
iteration is recorded. A run cut off by a time limit is reported from its
progress file if every rank has at least five measured iterations on disk,
and it is marked incomplete. A run that fails is reported with the first
error line of its log. If the gather does not complete at a rank count, the
paper says so, reports the failure, and keeps its sentence that no
conclusion is drawn there. Nothing is rerun. A job that completes no timed
run may be resubmitted once if the Stampede3 ledger allows.

## 5. Expectations, stated now and not used in scoring
Y1. The bucketed gather completes at 16 and 32 ranks with a peak below
45 GB.
Y2. The exchange ratio of the gather to native-bucketed is above p/2c: 2 at
16 ranks and 4 at 32.
Y3. Bucketing changes the 8-rank gather's exchange time by less than 15%.

## 6. Budget
Charges of this plan on Stampede3 never exceed 20 SU committed
(harness/launch/su_ledger.py).

## 7. Frozen components
validation_v2/analysis/score_e10.py and rev4_records.py, harness v5.2,
harness/launch/pvc_e10.slurm and e10_sets.sh. Hashes:
diagnostic/E10_FREEZE.sha256.
