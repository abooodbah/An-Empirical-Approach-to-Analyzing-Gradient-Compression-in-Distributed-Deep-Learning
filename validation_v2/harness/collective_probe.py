# validation_v2/harness/collective_probe.py
# -*- coding: utf-8 -*-
"""Times all-reduce and all-gather over a payload sweep with the harness's
own calibration routine, under whatever collective settings the environment
selects (for example NCCL_ALGO). Rank 0 writes one JSON file.

usage (under torchrun or srun, like the harness):
  collective_probe.py --label L --output F [--collectives allreduce,allgather]
                      [--sizes 4,1024,...] [--max-rank-bytes N] [--dtype-check]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control_workload_benchmark_v5_2 as H  # noqa: E402


def dtype_check(runtime):
    """What the backend's all-reduce accepts on a two-byte wire."""
    torch, dist = H.torch, H.dist
    out = {}
    try:
        dist.all_reduce(torch.ones(4, dtype=torch.int16, device=runtime.device))
        out["int16_allreduce"] = "accepted"
    except Exception as exc:  # the backend names its own error type
        out["int16_allreduce"] = "rejected: %s" % str(exc).splitlines()[0][:160]
    level = H.levels_fp16(runtime.world_size)
    try:
        value = torch.full((4,), float(level), dtype=torch.float16, device=runtime.device)
        dist.all_reduce(value)
        out["fp16_integer_sum_exact"] = bool((value.to(torch.int32) == level * runtime.world_size).all().item())
    except Exception as exc:
        out["fp16_integer_sum_exact"] = False
        out["fp16_allreduce_error"] = str(exc).splitlines()[0][:160]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collectives", default="allreduce,allgather")
    parser.add_argument("--sizes", default="4,1024,1048576,16777216,67108864")
    parser.add_argument("--max-rank-bytes", type=int, default=8 * 1024 ** 3)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--dtype-check", action="store_true")
    a = parser.parse_args(argv)
    names = [name.strip() for name in a.collectives.split(",") if name.strip()]
    if not names or any(name not in ("allreduce", "allgather") for name in names):
        parser.error("--collectives takes allreduce, allgather, or both")
    hargs = H.parse_args(["--campaign-id", "probe", "--run-id", a.label,
                          "--output-dir", os.path.dirname(os.path.abspath(a.output)),
                          "--profile", "qualification", "--model", "transformer", "--skip-calibration",
                          "--accelerator", a.accelerator, "--backend", a.backend])
    runtime, _ = H.initialize_distributed(hargs)
    result = {"label": a.label, "world_size": runtime.world_size, "backend": runtime.backend,
              "environment": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("NCCL_", "CCL_"))}}
    sizes = H.parse_calibration_sizes(a.sizes)
    for name in names:
        result[name] = dict(H._calibrate_one(runtime, name, a.max_rank_bytes, sizes))
    if a.dtype_check:
        result["dtype_check"] = dtype_check(runtime)
    if runtime.rank == 0:
        with open(a.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, sort_keys=True)
    if H.dist.is_initialized():
        H.dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
