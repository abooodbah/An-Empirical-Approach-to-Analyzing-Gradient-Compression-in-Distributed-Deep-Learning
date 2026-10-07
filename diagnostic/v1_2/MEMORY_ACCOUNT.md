# Peak device memory of the 1B conditions on record (RETROSPECTIVE, 2026-10-07)

Largest `peak_memory_mb` over ranks and iterations, in GB (1 GB = 1000 MB).

| run | mode | ranks | timing | peak GB | above the dense row of the same run |
|---|---|---|---|---|---|
| redctl-V-1n-3517750 | ddp-int8-hook | 8 | phase | 26.3 | 5.3 |
| redctl-V-1n-3517750 | native | 8 | phase | 21.0 |  |
| redctl-V-1n-3517751 | ddp-fp16-hook | 8 | phase | 34.1 | 0.5 |
| redctl-V-1n-3517751 | ddp-int8-hook | 8 | phase | 38.3 | 4.8 |
| redctl-V-1n-3517751 | gpu-quant | 8 | phase | 43.0 | 9.4 |
| redctl-V-1n-3517751 | gpu-quant-reducible | 8 | phase | 34.6 | 1.0 |
| redctl-V-1n-3517751 | native | 8 | phase | 33.5 |  |
| redctl-V-1n-3517832 | ddp-fp16-hook | 8 | wall | 34.1 | 0.0 |
| redctl-V-1n-3517832 | ddp-int8-hook | 8 | wall | 38.3 | 4.3 |
| redctl-V-1n-3517832 | native | 8 | wall | 34.1 |  |
| redctl-V-2n-3517752 | native | 16 | phase | 33.5 |  |
| redctl-V-2n-3517833 | gpu-quant-reducible | 16 | phase | 34.6 |  |
| redctl-V-2n-3517873 | ddp-fp16-hook | 16 | phase | 34.1 |  |
| redctl-V-2n-3517873 | ddp-fp16-hook | 16 | wall | 34.1 |  |
| redctl-V-4n-3517753 | gpu-quant-reducible | 32 | phase | 34.6 | 1.0 |
| redctl-V-4n-3517753 | native | 32 | phase | 33.5 |  |
| v2-L1-2n-3465529 | ddp-fp16-hook | 6 | block | 12.9 | 0.1 |
| v2-L1-2n-3465529 | ddp-fp16-hook | 6 | wall | 12.8 | 0.1 |
| v2-L1-2n-3465529 | ddp-int8-hook | 6 | block | 22.1 | 9.4 |
| v2-L1-2n-3465529 | ddp-int8-hook | 6 | wall | 22.1 | 9.4 |
| v2-L1-2n-3465529 | ddp-overlap | 6 | block | 12.7 |  |
| v2-L1-2n-3465529 | ddp-overlap | 6 | wall | 12.7 |  |

Whole-gradient gather, Section VIII arithmetic (32.8 GB footprint, 1.05 GB per copy):
16 ranks 49.6 GB, 32 ranks 66.4 GB, on a 64 GB device.

## The two hook failures on record

- Job 3517833, the int8 hook at 16 ranks: the run ended at the final hash on one rank with "RuntimeError: Native API failed. Native API returns: -5 (PI_ERROR_OUT_OF_RESOURCES) -5 (PI_ERROR_OUT_OF_RESOURCES)" (`raw/control/redctl-V-reruns-3517832-3517833.tgz`, `redctl-V-2n-3517833/logs/llama-ddp-int8-hook.14.err`).
- Job 3517753, the int8 hook at 32 ranks: ten rank logs end with GPU page faults on write, for example "Segmentation fault from GPU at 0xff00000f5a016000, ctx_id: 5 (CCS) type: 0 (NotPresent), level: 1 (PDE), access: 1 (Write), banned: 1, aborting." (`raw/control/redctl-V-v3-3517750-51-52-53.tgz`, `redctl-V-4n-3517753/logs/llama-ddp-int8-hook.*.err`).

## Units

`peak_memory_mb` is bytes divided by 10^6 (the harness's `peak_memory_mb()`), and the table above divides it by 1000. Section VIII of the 7 October manuscript prints 42.0 GB for the eight-rank whole-gradient gather and 32.8 GB for the model's footprint. Both come from `diagnostic/peak_mem_1b.py`, which divides the same records by 1024. In the unit of this table they are 43.0 GB and 33.5 GB. With those values the arithmetic of Section VIII (the footprint plus p int8 copies of 1.046 GB) gives 41.9 GB at 8 ranks, against 43.0 GB measured, 50.2 GB at 16 ranks, and 67.0 GB at 32 ranks. The two lines under the table repeat the manuscript's own numbers.
