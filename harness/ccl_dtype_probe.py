import json
import os
import sys
import time

import torch
import torch.distributed as dist
import intel_extension_for_pytorch  # noqa: F401
import oneccl_bindings_for_pytorch  # noqa: F401

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
local = int(os.environ["LOCAL_RANK"])
device = torch.device(f"xpu:{local}")
torch.xpu.set_device(device)
dist.init_process_group("ccl", init_method="env://", rank=rank, world_size=world)
results = {}
for dtype_name in ("float32", "bfloat16", "float16", "int8"):
    dtype = getattr(torch, dtype_name)
    try:
        x = torch.ones(1 << 20, dtype=dtype, device=device)
        t0 = time.perf_counter()
        dist.all_reduce(x)
        torch.xpu.synchronize(device)
        dt = time.perf_counter() - t0
        ok = float(x.flatten()[0].item()) == float(world) if dtype != torch.int8 else int(x.flatten()[0].item()) == world
        results[dtype_name] = {"ok": bool(ok), "seconds": dt}
    except Exception as exc:
        results[dtype_name] = {"ok": False, "error": repr(exc)[:160]}
    dist.barrier()
try:
    y = torch.ones(1 << 16, dtype=torch.bfloat16, device=device)
    work = dist.all_reduce(y, async_op=True)
    fut = work.get_future()
    fut.wait()
    results["bf16_future"] = {"ok": True}
except Exception as exc:
    results["bf16_future"] = {"ok": False, "error": repr(exc)[:160]}
if rank == 0:
    print("PROBE_RESULT " + json.dumps({"alg": os.environ.get("CCL_ALLREDUCE", "default"), **results}), flush=True)
dist.destroy_process_group()
