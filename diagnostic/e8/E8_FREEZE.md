# E8 freeze

The reference and the scorer below are frozen, as diagnostic/E8_PROTOCOL.md
requires. They were committed and pushed before any E8 record was copied from
Lonestar6; provenance/PROVENANCE.md records the push time beside the E8 jobs'
times. After this push nothing listed here changes; a later change is a new
version, and E8 is then reported as retrospective.

## Gate G2 at the freeze (diagnostic/e8/DEV_REPORT.md)
- Hook family (H): median absolute error 9.1% over six development points (V1
  traces, V1 to V3 block-timed ground truth); G2 passes.
- Gather family (G): no development trace exists at the freeze. G is reported as
  "not validated on development data" unless the frozen code, applied to the e7a
  traces before any E8 record is read, gives its errors in DEV_REPORT_E7A.md.

## Development changes before the freeze (development data only)
- The launch latency is estimated from launch-limited ops only; queued kernels
  leave gaps of a few microseconds and inflated it on the batch-128 traces.
- Hook regions are found among the children of the AccumulateGrad function
  that runs each bucket's DDP hook.
- The substituted fp16 copy-back runs on a side stream after its all-reduce,
  and the compute stream waits for the copies at the first compute op after the
  iteration's last hook, as the traced V1 fp16 hook does.
- Every iteration's gradient all-gather is substituted; the substituted
  all-reduce keeps the gather's waits; the final cast waits for the all-reduce.
- The all-reduce stays costed from the isolated calibration as section 7 fixes,
  although the V1 fp16 traces show in-training all-reduces 1.7 to 4.1 times
  slower (DEV_REPORT.md); the resulting optimism is reported, not corrected.

## Frozen files
- sha256 357546661f63e26448732fdd662f762a60f830f259d66d27ccd95d5f1d23392b `diagnostic/e8/replay_reference.py`
- sha256 0d00089cfbf24e4bdf4e3bb5cc803cc00fdea9dd6c8895a60d1abc5933260420 `diagnostic/e8/score_e8.py`
- sha256 5282a0b6562f28e7947071b76d714d83bdf328355fbc4c2fed4293363aebc858 `diagnostic/e8/test_replay_reference.py`
- sha256 50e717e84326534a539dfbcc84e94d27105ef2e728470e078d2ccb32994f58fb `diagnostic/e8/test_score_e8.py`
- sha256 e348826d307b6527a1cb52941e12fc298ad7f03baf59c09749b6beb70e147755 `diagnostic/e8/dev_v1.py`
- sha256 b1f8b971ef775b2da0e3a130961b9f7d59185cf347b1c5653062dff9b36dfaa0 `diagnostic/e8/DEV_REPORT.md`
- sha256 4b50a8e95954f0dd981b4929ba04409f7a0c534ea4b26bad6d11c78459950721 `raw/v2/v2-V1-traces-3465526.tgz`
