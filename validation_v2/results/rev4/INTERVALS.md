Allocation-level summaries (RETROSPECTIVE, 2026-10-07).
n = allocations. Interval: 95% t, shown for n >= 3.

## validation campaign, Transformer, 12 ranks

| quantity | n | center | range | 95% interval |
|---|---|---|---|---|
| g_matched under B | 3 | 0.543 | 0.528 to 0.558 | 0.506 to 0.581 |
| g_matched under E | 3 | 0.504 | 0.496 to 0.516 | 0.478 to 0.530 |
| g_matched under S | 3 | 0.541 | 0.537 to 0.547 | 0.529 to 0.553 |
| h_topk under B | 3 | 0.086 | 0.070 to 0.102 | 0.047 to 0.126 |
| h_topk under E | 3 | 0.165 | 0.074 to 0.313 | -0.156 to 0.487 |
| h_topk under S | 3 | 0.079 | 0.031 to 0.119 | -0.032 to 0.189 |
| ddp-fp16-hook over dense DDP, batch 32 | 3 | 0.866 | 0.857 to 0.880 | 0.837 to 0.897 |
| ddp-fp16-hook over dense DDP, batch 128 | 3 | 0.927 | 0.897 to 0.968 | 0.841 to 1.022 |
| ddp-int8-hook over dense DDP, batch 32 | 3 | 1.797 | 1.738 to 1.841 | 1.668 to 1.936 |
| ddp-int8-hook over dense DDP, batch 128 | 3 | 1.379 | 1.334 to 1.407 | 1.282 to 1.483 |

## H100 overlap control

| quantity | n | center | range | 95% interval |
|---|---|---|---|---|
| resnet50 | 3 | 1.021 |  | 1.001 to 1.042 |
| transformer | 3 | 0.871 |  | 0.825 to 0.921 |

## Intel control, exchange ratio

| quantity | n | center | range | 95% interval |
|---|---|---|---|---|
| transformer gpu-quant over native, 8 ranks | 2 | 0.968 | 0.967 to 0.969 |  |
| transformer gpu-quant-reducible over native, 8 ranks | 2 | 0.304 | 0.293 to 0.316 |  |
| transformer gpu-quant over native, 16 ranks | 1 | 2.606 | 2.606 to 2.606 |  |
| transformer gpu-quant-reducible over native, 16 ranks | 1 | 0.295 | 0.295 to 0.295 |  |
| transformer gpu-quant over native, 32 ranks | 2 | 7.632 | 6.949 to 8.383 |  |
| transformer gpu-quant-reducible over native, 32 ranks | 2 | 0.358 | 0.353 to 0.362 |  |
| resnet50 gpu-quant over native, 8 ranks | 2 | 1.026 | 1.025 to 1.027 |  |
| resnet50 gpu-quant-reducible over native, 8 ranks | 2 | 0.442 | 0.413 to 0.472 |  |
| resnet50 gpu-quant over native, 16 ranks | 1 | 3.184 | 3.184 to 3.184 |  |
| resnet50 gpu-quant-reducible over native, 16 ranks | 1 | 0.335 | 0.335 to 0.335 |  |
| resnet50 gpu-quant over native, 32 ranks | 2 | 6.508 | 5.736 to 7.384 |  |
| resnet50 gpu-quant-reducible over native, 32 ranks | 2 | 0.364 | 0.352 to 0.376 |  |
