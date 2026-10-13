# E11: the int8 hook with asynchronous collectives on NCCL (development run)

Written 2026-10-13, before the job was submitted. Exploratory: a feasibility
run on the Lonestar6 development queue, requested by the author to decide
whether a full run at 12 and 24 ranks is worth doing.

## Why

In every earlier run the int8 hook issued its collectives synchronously.
The synchronous form exists for oneCCL, whose nested future callbacks
deadlock. Harness v5.2 chose it with the test `"ccl" in backend`, which
also matches `"nccl"`, so it ran synchronously on NVIDIA systems too.
Harness v5.3 (`validation_v2/harness/build_v53.py`, test
`validation_v2/analysis/tests/test_harness_v53.py`) chooses it on oneCCL
only. `RED_HOOK_SYNC=1` still forces it.

## What runs

`harness/launch/ls6_e11.slurm` with `harness/launch/e11_sets.sh`: two
Lonestar6 A100 nodes (6 ranks), GPUDirect as in E9
(`NCCL_NET_GDR_LEVEL=SYS`), batch 32, block timing, 100 measured
iterations after 20 warm-up, two passes (the second reversed). Workloads:
Transformer and ResNet-50. Per workload: dense DDP (`ddp-overlap`), the
16-bit hook, the int8 hook asynchronous, the int8 hook synchronous.
Budget: 2 nodes for at most 45 minutes, 4.5 SU at 3 SU per node-hour.

## How it will be read

- Each run's provenance must record the hook mode (`hook_state`): `async`
  for the asynchronous condition and `sync` for the other. A condition
  whose record says otherwise is invalid.
- Ratios: each hook's block-timed step over dense DDP's, and the
  asynchronous int8 hook's over the synchronous one's, per pass.
- Expectation, stated now: the asynchronous int8 hook is faster than the
  synchronous one but still slower than dense DDP and than the 16-bit hook
  on both workloads, because its all-gather volume grows with the rank
  count and each source is dequantized in turn.
- Six ranks is one scale on the development queue's two nodes. It does not
  replace a 12- and 24-rank run; it tells whether that run is worth doing
  and whether the asynchronous form completes on NCCL at all.
