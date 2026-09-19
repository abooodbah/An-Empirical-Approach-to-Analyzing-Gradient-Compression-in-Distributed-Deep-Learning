#!/usr/bin/env python3
"""Fail-closed XPU/IPEX/oneCCL qualification probe; not a benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
from datetime import timedelta
from typing import Any, Optional


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def package_version(names: tuple[str, ...]) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "unknown"


def import_stack() -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch as ccl
    import torchvision

    versions = {
        "torch": str(torch.__version__),
        "ipex": str(getattr(ipex, "__version__", package_version(("intel-extension-for-pytorch",)))),
        "torchvision": str(torchvision.__version__),
        "oneccl": str(
            getattr(
                ccl,
                "__version__",
                package_version(("oneccl-bind-pt", "oneccl_bind_pt", "oneccl_bindings_for_pytorch")),
            )
        ),
    }
    cpu_only = {name: version for name, version in versions.items() if "+cpu" in version.lower()}
    if cpu_only:
        raise RuntimeError(f"CPU-only package build rejected: {cpu_only}")
    if "+gpu" not in versions["oneccl"].lower():
        raise RuntimeError(
            "XPU oneCCL package must report a +gpu build; "
            f"active oneCCL version is {versions['oneccl']!r}"
        )
    expected_versions = {
        "torch": os.environ.get("PVC_EXPECT_TORCH_VERSION", "1.13.0a0+gitb1dde16"),
        "ipex": os.environ.get("PVC_EXPECT_IPEX_VERSION", "1.13.10+xpu"),
        "torchvision": os.environ.get("PVC_EXPECT_TORCHVISION_VERSION", "0.14.1a0+0504df5"),
        "oneccl": os.environ.get("PVC_EXPECT_ONECCL_VERSION", "1.13.100+gpu"),
    }
    mismatches = {
        name: {"expected": expected_versions[name], "active": versions[name]}
        for name in expected_versions
        if versions[name] != expected_versions[name]
    }
    if mismatches:
        raise RuntimeError(f"unqualified Intel XPU package versions rejected: {mismatches}")
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("torch.xpu is unavailable in the active Python environment")

    def xpu_name(index: int) -> str:
        if hasattr(torch.xpu, "get_device_name"):
            return str(torch.xpu.get_device_name(index))
        return str(torch.xpu.get_device_properties(index))

    info = {
        "versions": versions,
        "module_files": {
            "torch": str(Path(torch.__file__).resolve()),
            "ipex": str(Path(ipex.__file__).resolve()),
            "torchvision": str(Path(torchvision.__file__).resolve()),
            "oneccl": str(Path(ccl.__file__).resolve()),
        },
        "python": sys.executable,
        "python_version": sys.version,
        "xpu_device_count": int(torch.xpu.device_count()),
        "xpu_names": [xpu_name(index) for index in range(torch.xpu.device_count())],
    }
    return torch, ipex, ccl, info


def selected_device(torch: Any) -> int:
    count = int(torch.xpu.device_count())
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    if count == 1:
        return 0
    if not 0 <= local_rank < count:
        raise RuntimeError(f"local rank {local_rank} cannot address {count} visible XPU devices")
    return local_rank


def synchronize_xpu(torch: Any, device: Optional[Any] = None) -> None:
    try:
        torch.xpu.synchronize(device) if device is not None else torch.xpu.synchronize()
    except TypeError:
        torch.xpu.synchronize()


def loaded_accelerator_libraries() -> list[str]:
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    needles = ("ccl", "fabric", "ofi", "ze_loader", "sycl", "level_zero")
    libraries: set[str] = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        candidate = line.rsplit(maxsplit=1)[-1]
        lowered = candidate.lower()
        if candidate.startswith("/") and any(needle in lowered for needle in needles):
            libraries.add(candidate)
    return sorted(libraries)


def inspect_mode(output_dir: Path) -> None:
    torch, _, _, info = import_stack()
    if info["xpu_device_count"] != 8:
        raise RuntimeError(f"expected exactly eight visible PVC tiles, found {info['xpu_device_count']}")
    info.update({"mode": "inspect", "host": socket.gethostname(), "status": "PASS"})
    atomic_json(output_dir / "stack_inspect.json", info)
    print(json.dumps(info, sort_keys=True), flush=True)
    synchronize_xpu(torch)


def tile_mode(output_dir: Path) -> None:
    torch, _, _, info = import_stack()
    device_index = selected_device(torch)
    torch.xpu.set_device(device_index)
    device = torch.device("xpu", device_index)
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    left = torch.full((64, 64), float(local_rank + 1), dtype=torch.float32, device=device)
    right = torch.eye(64, dtype=torch.float32, device=device)
    result = left @ right
    synchronize_xpu(torch, device)
    if not torch.equal(result, left):
        raise RuntimeError(f"XPU matmul exactness failure on local rank {local_rank}")
    payload = {
        **info,
        "mode": "tile",
        "host": socket.gethostname(),
        "local_rank": local_rank,
        "selected_device": device_index,
        "process_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "status": "PASS",
    }
    atomic_json(output_dir / f"tile_{local_rank:02d}.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def affinity_mode(output_dir: Path) -> None:
    receipts = sorted(output_dir.glob("tile_*.json"))
    if not receipts:
        raise RuntimeError(f"no tile receipts found under {output_dir}")
    ordered: list[tuple[int, list[int]]] = []
    for receipt in receipts:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        local_rank = int(payload["local_rank"])
        affinity = [int(value) for value in payload["process_cpu_affinity"]]
        if not affinity:
            raise RuntimeError(f"empty process affinity in {receipt}")
        ordered.append((local_rank, affinity))
    ordered.sort()
    if [rank for rank, _ in ordered] != list(range(len(ordered))):
        raise RuntimeError(f"tile receipt local ranks are incomplete: {ordered}")
    # Stampede3 alternates task CPU sets across two NUMA domains.  The bundled
    # oneCCL 1.13 build intermittently rejects pthread affinity on the odd-rank
    # NUMA domain.  Workload srun steps therefore use --cpu-bind=none and bind
    # the eight oneCCL workers to two unique CPUs from each even-rank (NUMA-0)
    # discovery set.  This remains derived from the allocation, not hard-coded.
    cpus = [cpu for rank, affinity in ordered if rank % 2 == 0 for cpu in affinity[:2]]
    if len(cpus) != len(ordered):
        raise RuntimeError(
            f"cannot derive one NUMA-0 worker CPU per local rank from tile receipts: {ordered}"
        )
    if len(set(cpus)) != len(cpus):
        raise RuntimeError(f"derived worker CPUs are not unique: {cpus}")
    print(",".join(str(cpu) for cpu in cpus), flush=True)


def ccl_mode(output_dir: Path) -> None:
    torch, _, _, info = import_stack()
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device_index = selected_device(torch)
    torch.xpu.set_device(device_index)
    device = torch.device("xpu", device_index)
    dist.init_process_group(
        backend="ccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        value = float(rank + 1)
        tensor = torch.tensor([value, 2.0 * value], dtype=torch.float32, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        synchronize_xpu(torch, device)
        expected_sum = float(world_size * (world_size + 1) // 2)
        expected = torch.tensor([expected_sum, 2.0 * expected_sum], dtype=torch.float32, device=device)
        if not torch.equal(tensor, expected):
            raise RuntimeError(
                f"exact CCL all-reduce mismatch on rank {rank}: got={tensor.cpu().tolist()} "
                f"expected={expected.cpu().tolist()}"
            )
        dist.barrier()
        synchronize_xpu(torch, device)
        payload = {
            **info,
            "mode": "ccl",
            "backend": dist.get_backend(),
            "host": socket.gethostname(),
            "rank": rank,
            "world_size": world_size,
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "selected_device": device_index,
            "process_cpu_affinity": sorted(os.sched_getaffinity(0)),
            "ccl_worker_count": os.environ.get("CCL_WORKER_COUNT"),
            "ccl_worker_affinity": os.environ.get("CCL_WORKER_AFFINITY"),
            "ccl_local_rank": os.environ.get("CCL_LOCAL_RANK"),
            "ccl_local_size": os.environ.get("CCL_LOCAL_SIZE"),
            "allreduce_result": tensor.cpu().tolist(),
            "loaded_accelerator_libraries": loaded_accelerator_libraries(),
            "status": "PASS",
        }
        atomic_json(output_dir / f"ccl_rank_{rank:03d}.json", payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("inspect", "tile", "affinity", "ccl"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "inspect":
        inspect_mode(args.output_dir)
    elif args.mode == "tile":
        tile_mode(args.output_dir)
    elif args.mode == "affinity":
        affinity_mode(args.output_dir)
    else:
        ccl_mode(args.output_dir)


if __name__ == "__main__":
    main()
