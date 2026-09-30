# TPDS v2 verdicts (rules frozen in ANALYSIS_RULES_v2.md)

## Experiment A

- transformer g_matched E vs B: direction same practical direction; magnitude allocation-dependent (gap in points by job: -5.7, -1.2, -4.8); ranking preserved
- transformer g_matched S vs B: direction same practical direction; magnitude preserved (gap in points by job: -1.1, +1.2, -0.7); ranking preserved
- transformer h_topk E vs B: direction same practical direction; magnitude allocation-dependent (gap in points by job: +22.6, +3.8, -2.8); ranking preserved
- transformer h_topk S vs B: direction allocation-dependent (same practical direction, crosses the 5% margin, same practical direction); magnitude preserved (gap in points by job: +3.2, -3.9, -1.7); ranking preserved
- resnet50 g_matched E vs B: incomplete (0 of 3 jobs)
- resnet50 g_matched S vs B: incomplete (0 of 3 jobs)
- resnet50 h_topk E vs B: incomplete (0 of 3 jobs)
- resnet50 h_topk S vs B: incomplete (0 of 3 jobs)
- transformer perturbation rho_E differs between paths (spread by job: 0.532, 0.079, 0.116)

## Experiment B

- transformer ddp-fp16-hook: relative cost inconsistent across allocations (change by job: +0.088, +0.041, +0.054); small batch faster than dense DDP; large batch inconsistent against dense DDP
- transformer ddp-int8-hook: relative cost changes with computation (change by job: -0.434, -0.404, -0.416); small batch slower than dense DDP; large batch slower than dense DDP
- resnet50 ddp-fp16-hook: incomplete (0 of 3 jobs)
- resnet50 ddp-int8-hook: incomplete (0 of 3 jobs)
