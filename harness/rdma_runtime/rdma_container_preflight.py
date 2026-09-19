"""Fail-closed proof that a frozen Stampede3 RDMA layout works in-container."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import resource
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "2.0.0"
BRIDGE_SCHEMA_VERSION = "2.0.0"
FROZEN_SCHEMA_VERSION = "1.0.0"
FROZEN_PURPOSE = "tacc_h100_mlx5_frozen_runtime_layout"
EXPECTED_HCA = "mlx5_0"
EXPECTED_PORT = "1"
EXPECTED_NCCL_HCA = "=mlx5_0:1"
EXPECTED_SOCKET_INTERFACE = "=ibp92s0"
EXPECTED_PROVIDER = "mlx5"
MINIMUM_RATE_GBPS = 400.0
AUTHORIZED_FROZEN_SHA256 = (
    "2f11d9c1a84e0441d5a7c2333c6d452308dbe77f18e457f90db8ee48db6f8209"
)
AUTHORIZED_LAYOUT_DIGEST_SHA256 = (
    "8e15b75a379512d500a8d0dfa81d00bb40b00f28184af53b573c1603a8a6f978"
)
RATE_PATTERN = re.compile(r"(?P<gbps>[0-9]+(?:\.[0-9]+)?)\s*Gb/sec", re.IGNORECASE)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
BRIDGE_KEYS = {
    "schema_version",
    "generated_utc",
    "status",
    "purpose",
    "hostname",
    "bundle_root",
    "frozen_layout",
    "hardware",
    "libibverbs",
    "provider",
    "provider_config",
    "bundle_files",
    "dt_needed",
    "fixed_dependency_closure",
    "container_supplied_dependencies",
    "runtime_bind_plan",
    "bundle_digest_sha256",
    "failures",
}
FROZEN_REFERENCE_KEYS = {
    "staged_name",
    "path",
    "sha256",
    "layout_digest_sha256",
    "diagnostic",
}
BIND_PLAN_KEYS = {
    "source",
    "bundle_path",
    "container_path",
    "kind",
    "sha256",
    "bytes",
    "read_only",
}
SOURCE_IDENTITY_KEYS = {
    "requested_path",
    "real_path",
    "uid",
    "gid",
    "mode",
    "bytes",
    "sha256",
}
BUNDLE_FILE_KEYS = {
    "name",
    "source_path",
    "real_source_path",
    "source_uid",
    "source_gid",
    "source_mode",
    "bytes",
    "sha256",
}


class IbvDevice(ctypes.Structure):
    """Opaque libibverbs device handle."""


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _load_json_snapshot(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _source_identity_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != SOURCE_IDENTITY_KEYS:
        return False
    try:
        mode = int(str(value.get("mode")), 8)
    except ValueError:
        return False
    return (
        all(
            PurePosixPath(str(value.get(key, ""))).is_absolute()
            for key in ("requested_path", "real_path")
        )
        and value.get("uid") == 0
        and isinstance(value.get("gid"), int)
        and mode & 0o022 == 0
        and _is_positive_int(value.get("bytes"))
        and _is_digest(value.get("sha256"))
    )


def _bridge_bundle_digest(bridge: dict[str, Any]) -> str | None:
    try:
        bundle_files = bridge["bundle_files"]
        provider_config_source = bridge["provider_config"]["source"]
        plan = bridge["runtime_bind_plan"]
        identity = {
            "frozen_layout": {
                "sha256": bridge["frozen_layout"]["sha256"],
                "layout_digest_sha256": bridge["frozen_layout"][
                    "layout_digest_sha256"
                ],
            },
            "bundle_files": {
                name: {
                    "source_path": record["source_path"],
                    "real_source_path": record["real_source_path"],
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
                for name, record in sorted(bundle_files.items())
            },
            "provider_config": {
                "source_path": provider_config_source["requested_path"],
                "real_source_path": provider_config_source["real_path"],
                "bytes": provider_config_source["bytes"],
                "sha256": provider_config_source["sha256"],
            },
            "dt_needed": bridge["dt_needed"],
            "fixed_dependency_closure": bridge["fixed_dependency_closure"],
            "runtime_bind_plan": [
                {
                    "kind": entry["kind"],
                    "source": entry["source"],
                    "bundle_name": Path(entry["bundle_path"]).name,
                    "container_path": entry["container_path"],
                    "sha256": entry["sha256"],
                    "bytes": entry["bytes"],
                    "read_only": entry["read_only"],
                }
                for entry in plan
            ],
        }
    except (KeyError, TypeError, AttributeError):
        return None
    return _canonical_sha256(identity)


def _mapped_file_paths() -> list[str]:
    paths: list[str] = []
    maps = _read(Path("/proc/self/maps")) or ""
    for line in maps.splitlines():
        fields = line.split()
        if not fields or not fields[-1].startswith("/"):
            continue
        path = fields[-1]
        if path not in paths:
            paths.append(path)
    return paths


def _matching_mapped_path(paths: Iterable[str], names: set[str]) -> str | None:
    matches = [path for path in paths if Path(path).name in names]
    return matches[0] if len(matches) == 1 else None


def _verbs_devices(
    library: ctypes.CDLL, *, expected_hca: str
) -> tuple[list[str], bool]:
    count = ctypes.c_int(0)
    library.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.ibv_get_device_list.restype = ctypes.POINTER(ctypes.POINTER(IbvDevice))
    library.ibv_get_device_name.argtypes = [ctypes.POINTER(IbvDevice)]
    library.ibv_get_device_name.restype = ctypes.c_char_p
    library.ibv_free_device_list.argtypes = [
        ctypes.POINTER(ctypes.POINTER(IbvDevice))
    ]
    library.ibv_free_device_list.restype = None
    library.ibv_open_device.argtypes = [ctypes.POINTER(IbvDevice)]
    library.ibv_open_device.restype = ctypes.c_void_p
    library.ibv_close_device.argtypes = [ctypes.c_void_p]
    library.ibv_close_device.restype = ctypes.c_int
    devices = library.ibv_get_device_list(ctypes.byref(count))
    if not devices:
        raise RuntimeError("ibv_get_device_list returned NULL")
    try:
        names = []
        expected_opened = False
        for index in range(count.value):
            raw = library.ibv_get_device_name(devices[index])
            if raw:
                name = raw.decode("utf-8", errors="replace")
                names.append(name)
                if name == expected_hca:
                    context = library.ibv_open_device(devices[index])
                    if context:
                        expected_opened = library.ibv_close_device(context) == 0
        return names, expected_opened
    finally:
        library.ibv_free_device_list(devices)


def _validate_frozen_reference(
    bridge: dict[str, Any], failures: list[str]
) -> tuple[dict[str, Any], str | None]:
    reference = bridge.get("frozen_layout")
    if not isinstance(reference, dict) or set(reference) != FROZEN_REFERENCE_KEYS:
        failures.append("bridge frozen-layout reference has unexpected fields")
        return {}, None
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute() or path.name != reference.get("staged_name"):
        failures.append("bridge frozen-layout path/name is invalid")
        return {}, None
    try:
        frozen, observed_sha = _load_json_snapshot(path, "frozen provider layout")
    except (OSError, ValueError) as exc:
        failures.append(str(exc))
        return {}, None
    identity = {
        key: frozen.get(key)
        for key in (
            "diagnostic",
            "hardware",
            "libibverbs",
            "provider",
            "provider_config",
            "claim_boundary",
        )
    }
    computed_layout_digest = _canonical_sha256(identity)
    frozen_diagnostic = frozen.get("diagnostic")
    diagnostic = (
        {
            key: frozen_diagnostic.get(key)
            for key in ("job_id", "receipt_sha256", "raw_sha256")
        }
        if isinstance(frozen_diagnostic, dict)
        else None
    )
    if (
        frozen.get("schema_version") != FROZEN_SCHEMA_VERSION
        or frozen.get("status") != "FROZEN"
        or frozen.get("purpose") != FROZEN_PURPOSE
        or not _is_digest(observed_sha)
        or observed_sha != AUTHORIZED_FROZEN_SHA256
        or observed_sha != reference.get("sha256")
        or frozen.get("layout_digest_sha256") != computed_layout_digest
        or computed_layout_digest != AUTHORIZED_LAYOUT_DIGEST_SHA256
        or reference.get("layout_digest_sha256") != computed_layout_digest
        or reference.get("diagnostic") != diagnostic
        or bridge.get("hardware") != frozen.get("hardware")
        or bridge.get("libibverbs") != frozen.get("libibverbs")
        or bridge.get("provider") != frozen.get("provider")
        or bridge.get("provider_config") != frozen.get("provider_config")
    ):
        failures.append("bridge does not authenticate an exact frozen provider layout")
    return frozen, observed_sha


def _validate_hardware(hardware: object, failures: list[str]) -> dict[str, Any]:
    if not isinstance(hardware, dict):
        failures.append("frozen hardware record is missing")
        return {}
    rate = str(hardware.get("rate", ""))
    match = RATE_PATTERN.search(rate)
    parsed_rate = float(match.group("gbps")) if match else None
    observed = hardware.get("observed_rate_gbps")
    if (
        hardware.get("partition") != "h100"
        or hardware.get("gpu_model") != "NVIDIA H100"
        or hardware.get("gpu_count") != 4
        or hardware.get("hca") != EXPECTED_HCA
        or hardware.get("port") != EXPECTED_PORT
        or hardware.get("link_layer") != "InfiniBand"
        or hardware.get("minimum_rate_gbps") != int(MINIMUM_RATE_GBPS)
        or isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or float(observed) < MINIMUM_RATE_GBPS
        or parsed_rate is None
        or parsed_rate != float(observed)
        or "NDR" not in rate.upper()
        or "ACTIVE" not in str(hardware.get("state", "")).upper()
        or "LINKUP" not in str(hardware.get("physical_state", "")).upper()
    ):
        failures.append("frozen hardware is not exactly mlx5_0 port 1 at 400-Gb/s NDR")
    return hardware


def _validate_bind_plan(
    bridge: dict[str, Any], sidecar_root: Path, failures: list[str]
) -> dict[str, dict[str, Any]]:
    plan = bridge.get("runtime_bind_plan")
    if not isinstance(plan, list) or len(plan) != 2:
        failures.append("runtime bind plan must contain exactly two entries")
        return {}
    by_kind: dict[str, dict[str, Any]] = {}
    bundle_paths: set[str] = set()
    container_paths: set[str] = set()
    for item in plan:
        if not isinstance(item, dict) or set(item) != BIND_PLAN_KEYS:
            failures.append("runtime bind-plan entry has unexpected fields")
            continue
        kind = str(item.get("kind", ""))
        bundle_path = Path(str(item.get("bundle_path", "")))
        container_path = Path(str(item.get("container_path", "")))
        if kind in by_kind:
            failures.append(f"duplicate runtime bind-plan kind: {kind}")
        by_kind[kind] = item
        if str(bundle_path) in bundle_paths or str(container_path) in container_paths:
            failures.append("runtime bind plan contains a duplicate path")
        bundle_paths.add(str(bundle_path))
        container_paths.add(str(container_path))
        try:
            bundle_path.relative_to(sidecar_root)
        except ValueError:
            failures.append(f"runtime bind source escapes sidecar root: {bundle_path}")
        if (
            not bundle_path.is_file()
            or not PurePosixPath(str(container_path)).is_absolute()
            or not PurePosixPath(str(item.get("source", ""))).is_absolute()
            or item.get("read_only") is not True
            or not _is_digest(item.get("sha256"))
            or not _is_positive_int(item.get("bytes"))
        ):
            failures.append(f"runtime bind-plan entry is invalid: {kind}")
            continue
        if (
            _sha256(bundle_path) != item.get("sha256")
            or bundle_path.stat().st_size != item.get("bytes")
        ):
            failures.append(f"runtime bind source identity mismatch: {kind}")
    if set(by_kind) != {"provider-loader", "provider-config"}:
        failures.append("runtime bind-plan kinds are not exact")
        return by_kind

    provider = bridge.get("provider") or {}
    provider_source = provider.get("source") or {}
    loader = by_kind["provider-loader"]
    config = bridge.get("provider_config") or {}
    config_source = config.get("source") or {}
    config_bind = by_kind["provider-config"]
    if (
        loader.get("source") != provider_source.get("real_path")
        or loader.get("container_path") != provider.get("loader_target_path")
        or Path(str(loader.get("container_path", ""))).name
        != provider.get("loader_filename")
        or loader.get("sha256") != provider_source.get("sha256")
        or loader.get("bytes") != provider_source.get("bytes")
    ):
        failures.append("provider-loader bind does not match frozen provider identity")
    if (
        config_bind.get("source") != config_source.get("real_path")
        or config_bind.get("container_path") != config.get("container_target_path")
        or config_bind.get("sha256") != config_source.get("sha256")
        or config_bind.get("bytes") != config_source.get("bytes")
    ):
        failures.append("provider-config bind does not match frozen config identity")
    return by_kind


def inspect_environment(
    *,
    bridge_manifest_path: Path,
    expected_hca: str = EXPECTED_HCA,
) -> dict[str, Any]:
    failures: list[str] = []
    sidecar_root = Path(os.environ.get("ACM_RDMA_SIDECAR_ROOT", ""))
    if not sidecar_root.is_absolute() or not sidecar_root.is_dir():
        failures.append("ACM_RDMA_SIDECAR_ROOT is not an absolute readable directory")

    try:
        bridge = _load_json(bridge_manifest_path, "bridge manifest")
        bridge_sha256 = _sha256(bridge_manifest_path)
    except ValueError as exc:
        failures.append(str(exc))
        bridge = {}
        bridge_sha256 = None
    hostname = socket.gethostname().split(".", 1)[0].lower()
    if (
        set(bridge) != BRIDGE_KEYS
        or bridge.get("schema_version") != BRIDGE_SCHEMA_VERSION
        or bridge.get("status") != "PASS"
        or bridge.get("purpose")
        != "tacc_node_local_rdma_sidecar_from_frozen_layout"
        or bridge.get("hostname") != hostname
        or bridge.get("bundle_root") != str(sidecar_root)
        or bridge.get("failures") != []
        or not _is_digest(bridge.get("bundle_digest_sha256"))
        or bridge.get("bundle_digest_sha256") != _bridge_bundle_digest(bridge)
    ):
        failures.append("host RDMA bridge manifest is incomplete or mismatched")

    frozen, frozen_sha256 = _validate_frozen_reference(bridge, failures)
    hardware = _validate_hardware(bridge.get("hardware"), failures)
    provider = bridge.get("provider")
    provider_config = bridge.get("provider_config")
    libibverbs = bridge.get("libibverbs")
    fixed_closure = bridge.get("fixed_dependency_closure")
    bundle_files = bridge.get("bundle_files")
    if (
        not isinstance(provider, dict)
        or provider.get("name") != EXPECTED_PROVIDER
        or provider.get("ibv_drivers") != EXPECTED_PROVIDER
        or not _source_identity_valid(provider.get("source"))
        or not isinstance(provider.get("loader_target_path"), str)
        or not PurePosixPath(provider["loader_target_path"]).is_absolute()
        or Path(provider["loader_target_path"]).name != provider.get("loader_filename")
    ):
        failures.append("frozen mlx5 provider identity is invalid")
        provider = {}
    if (
        not isinstance(provider_config, dict)
        or provider_config.get("driver") != EXPECTED_PROVIDER
        or provider_config.get("active_lines") != ["driver mlx5"]
        or not _source_identity_valid(provider_config.get("source"))
        or not isinstance(provider_config.get("container_target_path"), str)
        or not PurePosixPath(provider_config["container_target_path"]).is_absolute()
    ):
        failures.append("frozen provider-config identity is invalid")
        provider_config = {}
    if (
        not isinstance(libibverbs, dict)
        or libibverbs.get("bundle_name") != "libibverbs.so.1"
        or not _source_identity_valid(libibverbs.get("source"))
    ):
        failures.append("frozen libibverbs identity is invalid")
        libibverbs = {}
    if (
        not isinstance(fixed_closure, dict)
        or set(fixed_closure) != {"libnl-3.so.200", "libnl-route-3.so.200"}
        or not all(_source_identity_valid(item) for item in fixed_closure.values())
    ):
        failures.append("fixed libibverbs dependency closure is invalid")
        fixed_closure = {}
    if not isinstance(bundle_files, dict) or not bundle_files:
        failures.append("bridge bundle-file manifest is missing")
        bundle_files = {}

    bind_plan = _validate_bind_plan(bridge, sidecar_root, failures)
    required_bundle_files = {
        str(libibverbs.get("bundle_name", "")),
        "libibverbs.so",
        "libmlx5.so",
        "libmlx5.so.1",
        str(provider.get("loader_filename", "")),
        *set(fixed_closure),
    }
    required_bundle_files.discard("")
    if set(bundle_files) != required_bundle_files:
        failures.append("bridge bundle-file set is not the exact frozen runtime bundle")
    expected_bundle_identities: dict[str, dict[str, Any]] = {}
    lib_source = libibverbs.get("source") or {}
    provider_source = provider.get("source") or {}
    for name in (str(libibverbs.get("bundle_name", "")), "libibverbs.so"):
        if name:
            expected_bundle_identities[name] = lib_source
    for name in (
        "libmlx5.so",
        "libmlx5.so.1",
        str(provider.get("loader_filename", "")),
    ):
        if name:
            expected_bundle_identities[name] = provider_source
    expected_bundle_identities.update(fixed_closure)
    for name, identity in expected_bundle_identities.items():
        bundle_record = bundle_files.get(name)
        if not isinstance(bundle_record, dict) or (
            set(bundle_record) != BUNDLE_FILE_KEYS
            or bundle_record.get("name") != name
            or bundle_record.get("real_source_path") != identity.get("real_path")
            or bundle_record.get("source_uid") != identity.get("uid")
            or bundle_record.get("source_gid") != identity.get("gid")
            or bundle_record.get("source_mode") != identity.get("mode")
            or bundle_record.get("bytes") != identity.get("bytes")
            or bundle_record.get("sha256") != identity.get("sha256")
        ):
            failures.append(f"bridge bundle identity differs from authorization: {name}")
    verified_bundle_files: dict[str, dict[str, Any]] = {}
    for name, record in bundle_files.items():
        path = sidecar_root / "lib" / str(name)
        if not isinstance(record, dict) or not path.is_file():
            failures.append(f"bridge bundle file is missing: {name}")
            continue
        observed_sha = _sha256(path)
        observed_bytes = path.stat().st_size
        if record.get("sha256") != observed_sha or record.get("bytes") != observed_bytes:
            failures.append(f"bridge bundle file identity mismatch: {name}")
        verified_bundle_files[str(name)] = {
            "path": str(path),
            "bytes": observed_bytes,
            "sha256": observed_sha,
        }

    expected_library_names = [
        str(libibverbs.get("bundle_name", "libibverbs.so.1")),
        *sorted(fixed_closure),
    ]
    environment = {
        "ACM_RDMA_SIDECAR_ROOT": os.environ.get("ACM_RDMA_SIDECAR_ROOT"),
        "ACM_RDMA_PROVIDER_LAYOUT": os.environ.get("ACM_RDMA_PROVIDER_LAYOUT"),
        "ACM_RDMA_PROVIDER_LAYOUT_SHA256": os.environ.get(
            "ACM_RDMA_PROVIDER_LAYOUT_SHA256"
        ),
        "ACM_RDMA_LAYOUT_DIGEST_SHA256": os.environ.get(
            "ACM_RDMA_LAYOUT_DIGEST_SHA256"
        ),
        "IBV_DRIVERS": os.environ.get("IBV_DRIVERS"),
        "IBV_SHOW_WARNINGS": os.environ.get("IBV_SHOW_WARNINGS"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE"),
        "NCCL_IB_HCA": os.environ.get("NCCL_IB_HCA"),
        "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME"),
    }
    expected_layout_path = str((bridge.get("frozen_layout") or {}).get("path", ""))
    expected_layout_digest = str(
        (bridge.get("frozen_layout") or {}).get("layout_digest_sha256", "")
    )
    if (
        environment["ACM_RDMA_PROVIDER_LAYOUT"] != expected_layout_path
        or environment["ACM_RDMA_PROVIDER_LAYOUT_SHA256"] != frozen_sha256
        or environment["ACM_RDMA_LAYOUT_DIGEST_SHA256"] != expected_layout_digest
    ):
        failures.append("container environment does not authenticate the frozen layout")
    if environment["IBV_DRIVERS"] != EXPECTED_PROVIDER:
        failures.append("IBV_DRIVERS is not exactly mlx5")
    if environment["NCCL_IB_DISABLE"] != "0":
        failures.append("NCCL_IB_DISABLE is not exactly 0")
    if environment["NCCL_IB_HCA"] != EXPECTED_NCCL_HCA:
        failures.append(f"NCCL_IB_HCA is not exactly {EXPECTED_NCCL_HCA}")
    if environment["NCCL_SOCKET_IFNAME"] != EXPECTED_SOCKET_INTERFACE:
        failures.append(
            f"NCCL_SOCKET_IFNAME is not exactly {EXPECTED_SOCKET_INTERFACE}"
        )
    if environment["IBV_SHOW_WARNINGS"] != "1":
        failures.append("IBV_SHOW_WARNINGS is not exactly 1")

    libraries: dict[str, dict[str, Any]] = {}
    loaded: dict[str, ctypes.CDLL] = {}
    for name in expected_library_names:
        path = sidecar_root / "lib" / name
        record: dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
            "loaded": False,
        }
        if not path.is_file():
            failures.append(f"sidecar library is missing: {name}")
        try:
            loaded[name] = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            record["loaded"] = True
        except OSError as exc:
            record["load_error"] = str(exc)
            failures.append(f"cannot load {name}: {exc}")
        libraries[name] = record

    device_names: list[str] = []
    expected_device_opened = False
    verbs = loaded.get(str(libibverbs.get("bundle_name", "libibverbs.so.1")))
    if verbs is not None:
        try:
            device_names, expected_device_opened = _verbs_devices(
                verbs, expected_hca=expected_hca
            )
        except (AttributeError, OSError, RuntimeError, ValueError) as exc:
            failures.append(f"libibverbs device enumeration failed: {exc}")
    if expected_hca not in device_names:
        failures.append(f"{expected_hca} is absent from libibverbs device enumeration")
    if not expected_device_opened:
        failures.append(f"ibv_open_device failed for {expected_hca}")

    hca_root = Path("/sys/class/infiniband") / expected_hca
    port_root = hca_root / "ports" / EXPECTED_PORT
    state = _read(port_root / "state")
    physical_state = _read(port_root / "phys_state")
    link_layer = _read(port_root / "link_layer")
    rate = _read(port_root / "rate")
    if not state or "ACTIVE" not in state.upper():
        failures.append(f"{expected_hca} port {EXPECTED_PORT} is not ACTIVE")
    if not physical_state or "LINKUP" not in physical_state.upper():
        failures.append(f"{expected_hca} port {EXPECTED_PORT} is not LinkUp")
    if link_layer != "InfiniBand":
        failures.append(f"{expected_hca} link layer is not InfiniBand")
    rate_match = RATE_PATTERN.search(rate or "")
    rate_gbps = float(rate_match.group("gbps")) if rate_match else None
    if (
        rate_gbps is None
        or rate_gbps < MINIMUM_RATE_GBPS
        or "NDR" not in str(rate).upper()
    ):
        failures.append(f"{expected_hca} port {EXPECTED_PORT} is not 400-Gb/s NDR")

    device_nodes = sorted(str(path) for path in Path("/dev/infiniband").glob("uverbs*"))
    if not device_nodes:
        failures.append("no /dev/infiniband/uverbs* device is visible in the container")

    memlock_soft, memlock_hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    memlock_minimum = 64 * 1024 * 1024
    if memlock_soft != resource.RLIM_INFINITY and memlock_soft < memlock_minimum:
        failures.append("container memlock limit is below 64 MiB")

    ldd_records: dict[str, dict[str, Any]] = {}
    for name in sorted(bundle_files):
        path = sidecar_root / "lib" / name
        completed = subprocess.run(
            ["ldd", str(path)], check=False, capture_output=True, text=True, timeout=30
        )
        ldd_records[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.returncode != 0 or "not found" in completed.stdout.lower():
            failures.append(f"ldd dependency resolution failed for {name}")

    all_mapped_paths = _mapped_file_paths()
    mapped_paths: dict[str, str] = {}
    for name in expected_library_names:
        mapped = _matching_mapped_path(all_mapped_paths, {name})
        if mapped is None or not Path(mapped).is_relative_to(sidecar_root):
            failures.append(f"{name} was not loaded from the read-only RDMA sidecar")
        elif _sha256(Path(mapped)) != (bundle_files.get(name) or {}).get("sha256"):
            failures.append(f"mapped {name} differs from the authenticated bridge copy")
        if mapped is not None:
            mapped_paths[name] = mapped

    loader_bind = bind_plan.get("provider-loader") or {}
    provider_names = {
        str(provider.get("loader_filename", "")),
        Path(str((provider.get("source") or {}).get("real_path", ""))).name,
        Path(str(loader_bind.get("bundle_path", ""))).name,
    }
    provider_names.discard("")
    provider_mapped = _matching_mapped_path(all_mapped_paths, provider_names)
    provider_observed_sha256: str | None = None
    if provider_mapped is None:
        failures.append("authenticated mlx5 provider was not mapped exactly once")
    else:
        provider_path = Path(provider_mapped)
        try:
            provider_observed_sha256 = _sha256(provider_path)
        except OSError as exc:
            failures.append(f"cannot authenticate mapped mlx5 provider: {exc}")
        allowed_provider_paths = {
            Path(str(loader_bind.get("container_path", ""))),
            Path(str(loader_bind.get("bundle_path", ""))),
        }
        if provider_path not in allowed_provider_paths:
            failures.append("mlx5 provider was loaded from an unexpected path")
        if provider_observed_sha256 != loader_bind.get("sha256"):
            failures.append("mapped mlx5 provider hash differs from the frozen layout")
        mapped_paths[str(provider.get("loader_filename", "provider-loader"))] = provider_mapped

    config_bind = bind_plan.get("provider-config") or {}
    provider_config_path = Path(str(config_bind.get("container_path", "")))
    config_observed_sha256: str | None = None
    config_observed_bytes: int | None = None
    try:
        config_observed_sha256 = _sha256(provider_config_path)
        config_observed_bytes = provider_config_path.stat().st_size
        config_text = provider_config_path.read_text(encoding="utf-8")
    except OSError as exc:
        config_text = ""
        failures.append(f"cannot authenticate mapped mlx5 provider config: {exc}")
    config_lines = [
        line.strip()
        for line in config_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if config_lines != ["driver mlx5"]:
        failures.append("mapped provider config does not select only driver mlx5")
    if (
        config_observed_sha256 != config_bind.get("sha256")
        or config_observed_bytes != config_bind.get("bytes")
    ):
        failures.append("mapped mlx5 provider config differs from the frozen layout")

    return {
        "schema_version": SCHEMA_VERSION,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": hostname,
        "status": "PASS" if not failures else "FAIL",
        "expected": {
            "hca": expected_hca,
            "port": EXPECTED_PORT,
            "minimum_rate_gbps": int(MINIMUM_RATE_GBPS),
            "fabric_generation": "NDR",
            "nccl_hca": EXPECTED_NCCL_HCA,
            "socket_interface": EXPECTED_SOCKET_INTERFACE,
        },
        "environment": environment,
        "frozen_layout": {
            "path": str((bridge.get("frozen_layout") or {}).get("path", "")),
            "sha256": frozen_sha256,
            "layout_digest_sha256": (bridge.get("frozen_layout") or {}).get(
                "layout_digest_sha256"
            ),
            "diagnostic": (bridge.get("frozen_layout") or {}).get("diagnostic"),
        },
        "bridge_manifest": {
            "path": str(bridge_manifest_path.resolve()),
            "sha256": bridge_sha256,
            "bundle_digest_sha256": bridge.get("bundle_digest_sha256"),
        },
        "hardware_authorization": hardware,
        "runtime_bind_plan": list(bind_plan.values()),
        "verified_bundle_files": verified_bundle_files,
        "libraries": libraries,
        "mapped_library_paths": mapped_paths,
        "provider": {
            "name": provider.get("name"),
            "source_real_path": (provider.get("source") or {}).get("real_path"),
            "loader_filename": provider.get("loader_filename"),
            "loader_target_path": provider.get("loader_target_path"),
            "mapped_path": provider_mapped,
            "observed_sha256": provider_observed_sha256,
            "expected_sha256": loader_bind.get("sha256"),
            "expected_bytes": loader_bind.get("bytes"),
        },
        "provider_config": {
            "path": str(provider_config_path),
            "observed_sha256": config_observed_sha256,
            "observed_bytes": config_observed_bytes,
            "expected_sha256": config_bind.get("sha256"),
            "expected_bytes": config_bind.get("bytes"),
            "driver_lines": config_lines,
        },
        "ldd": ldd_records,
        "verbs_devices": device_names,
        "expected_device_opened": expected_device_opened,
        "hca": {
            "state": state,
            "physical_state": physical_state,
            "link_layer": link_layer,
            "rate": rate,
            "rate_gbps": rate_gbps,
        },
        "device_nodes": device_nodes,
        "memlock": {
            "soft_bytes": memlock_soft,
            "hard_bytes": memlock_hard,
            "minimum_bytes": memlock_minimum,
        },
        "failures": failures,
    }


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bridge-manifest", type=Path, required=True)
    parser.add_argument("--expected-hca", default=EXPECTED_HCA)
    args = parser.parse_args(argv)
    if args.expected_hca != EXPECTED_HCA:
        parser.error(f"--expected-hca must be exactly {EXPECTED_HCA}")
    result = inspect_environment(
        bridge_manifest_path=args.bridge_manifest,
        expected_hca=args.expected_hca,
    )
    output = args.output_dir / f"rdma-container-preflight-{result['hostname']}.json"
    _exclusive_json(output, result)
    print(json.dumps({"output": str(output), "status": result["status"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
