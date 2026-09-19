"""Build an authenticated, manifest-driven host-RDMA sidecar for TACC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "2.0.0"
FROZEN_LAYOUT_SCHEMA_VERSION = "1.0.0"
FROZEN_LAYOUT_PURPOSE = "tacc_h100_mlx5_frozen_runtime_layout"
AUTHORIZED_LAYOUT_FILE_SHA256 = (
    "2f11d9c1a84e0441d5a7c2333c6d452308dbe77f18e457f90db8ee48db6f8209"
)
AUTHORIZED_LAYOUT_DIGEST_SHA256 = (
    "8e15b75a379512d500a8d0dfa81d00bb40b00f28184af53b573c1603a8a6f978"
)
PROVIDER_NAME = "mlx5"
BRIDGE_PURPOSE = "tacc_node_local_rdma_sidecar_from_frozen_layout"
PROVIDER_CONFIG_CONTAINER_PATH = PurePosixPath("/etc/libibverbs.d/mlx5.driver")

FROZEN_LAYOUT_KEYS = {
    "schema_version",
    "frozen_utc",
    "status",
    "purpose",
    "diagnostic",
    "hardware",
    "libibverbs",
    "provider",
    "provider_config",
    "claim_boundary",
    "layout_digest_sha256",
}
DIAGNOSTIC_KEYS = {
    "job_id",
    "hostname",
    "receipt_sha256",
    "raw_sha256",
    "raw_recorded_path",
}
HARDWARE_KEYS = {
    "partition",
    "gpu_model",
    "gpu_count",
    "hca",
    "port",
    "link_layer",
    "minimum_rate_gbps",
    "observed_rate_gbps",
    "rate",
    "state",
    "physical_state",
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
LIBIBVERBS_KEYS = {"bundle_name", "source"}
PROVIDER_KEYS = {
    "name",
    "ibv_drivers",
    "source",
    "loader_target_path",
    "loader_filename",
}
PROVIDER_CONFIG_KEYS = {
    "driver",
    "active_lines",
    "source",
    "container_target_path",
}
BRIDGE_MANIFEST_KEYS = {
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
FROZEN_LAYOUT_REFERENCE_KEYS = {
    "staged_name",
    "path",
    "sha256",
    "layout_digest_sha256",
    "diagnostic",
}
FROZEN_DIAGNOSTIC_REFERENCE_KEYS = {
    "job_id",
    "receipt_sha256",
    "raw_sha256",
}
RUNTIME_BIND_KEYS = {
    "source",
    "bundle_path",
    "container_path",
    "kind",
    "sha256",
    "bytes",
    "read_only",
}

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_POSIX_PATH = re.compile(r"^/[A-Za-z0-9._/+%-]+$")
SAFE_FIELD = re.compile(r"^[^\t\r\n]+$")
NEEDED_PATTERN = re.compile(r"Shared library: \[(?P<name>[^\]]+)\]")
LIBIBVERBS_PATTERN = re.compile(r"^libibverbs\.so(?:\.[0-9]+)*$")
PROVIDER_SOURCE_PATTERN = re.compile(r"^libmlx5\.so(?:\.[0-9]+)*$")
PROVIDER_LOADER_PATTERN = re.compile(r"^libmlx5-rdmav(?P<abi>[0-9]+)\.so$")
PROVIDER_LOADER_TARGET_PATTERN = re.compile(
    r"^/usr/lib64/libibverbs/libmlx5-rdmav(?P<abi>[0-9]+)\.so$"
)
MODE_PATTERN = re.compile(r"^0o[0-7]{3,4}$")

LIBRARY_ROOTS = (
    "/usr/lib64",
    "/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/lib/x86_64-linux-gnu",
    "/usr/lib",
    "/lib",
)
CONFIG_ROOTS = ("/etc/libibverbs.d", "/usr/share/libibverbs.d")
EXPECTED_RUNTIME_DEPENDENCIES = {
    "libnl-3.so.200": "/usr/lib64/libnl-3.so.200",
    "libnl-route-3.so.200": "/usr/lib64/libnl-route-3.so.200",
}
PROVIDER_CANONICAL_BUNDLE_NAME = "libmlx5.so.1"
PROVIDER_ALIAS_BUNDLE_NAME = "libmlx5.so"
LIBIBVERBS_ALIAS_BUNDLE_NAME = "libibverbs.so"

CORE_SEED_LIBRARIES = {
    "libibverbs.so.1": Path("/usr/lib64/libibverbs.so.1"),
}
CORE_ALIASES = {LIBIBVERBS_ALIAS_BUNDLE_NAME: "libibverbs.so.1"}
COPYABLE_DEPENDENCIES = {
    "libnl-3.so.200",
    "libnl-route-3.so.200",
    "libmnl.so.0",
    "libnuma.so.1",
    "libudev.so.1",
}
CONTAINER_DEPENDENCIES = {
    "libc.so.6",
    "libdl.so.2",
    "libgcc_s.so.1",
    "libm.so.6",
    "libpthread.so.0",
    "librt.so.1",
    "ld-linux-x86-64.so.2",
}


class BridgeError(RuntimeError):
    """Raised when a frozen layout or live source fails authentication."""


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


def _exact_dict(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BridgeError(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        raise BridgeError(
            f"{label} keys differ; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise BridgeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BridgeError(f"{label} must be a positive integer")
    return value


def _posix_path(value: object, label: str, roots: Sequence[str]) -> str:
    if not isinstance(value, str) or SAFE_POSIX_PATH.fullmatch(value) is None:
        raise BridgeError(f"{label} must be a safe absolute POSIX path")
    parsed = PurePosixPath(value)
    if str(parsed) != value or "." in parsed.parts or ".." in parsed.parts:
        raise BridgeError(f"{label} is not lexically canonical")
    if not any(root == "/" or value == root or value.startswith(f"{root}/") for root in roots):
        raise BridgeError(f"{label} escapes its allowlisted roots")
    return value


def _validate_source_identity(
    value: object,
    label: str,
    *,
    roots: Sequence[str],
    filename_pattern: re.Pattern[str] | None = None,
) -> dict[str, Any]:
    source = _exact_dict(value, SOURCE_IDENTITY_KEYS, label)
    requested = _posix_path(source["requested_path"], f"{label}.requested_path", roots)
    real = _posix_path(source["real_path"], f"{label}.real_path", roots)
    if filename_pattern is not None and filename_pattern.fullmatch(
        PurePosixPath(real).name
    ) is None:
        raise BridgeError(f"{label}.real_path has an unauthorized filename")
    if source["uid"] != 0:
        raise BridgeError(f"{label}.uid must be zero")
    if isinstance(source["gid"], bool) or not isinstance(source["gid"], int):
        raise BridgeError(f"{label}.gid must be an integer")
    mode = source["mode"]
    if not isinstance(mode, str) or MODE_PATTERN.fullmatch(mode) is None:
        raise BridgeError(f"{label}.mode is invalid")
    if int(mode, 8) & 0o022:
        raise BridgeError(f"{label} is group/world writable")
    return {
        "requested_path": requested,
        "real_path": real,
        "uid": 0,
        "gid": int(source["gid"]),
        "mode": mode,
        "bytes": _positive_int(source["bytes"], f"{label}.bytes"),
        "sha256": _digest(source["sha256"], f"{label}.sha256"),
    }


def _validate_needed_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise BridgeError(f"{label} must be a nonempty ordered list")
    if any(
        not isinstance(item, str)
        or not item
        or "/" in item
        or "\x00" in item
        for item in value
    ):
        raise BridgeError(f"{label} contains an invalid soname")
    if len(value) != len(set(value)):
        raise BridgeError(f"{label} contains duplicate sonames")
    return list(value)


def _validate_frozen_layout(value: object) -> dict[str, Any]:
    layout = _exact_dict(value, FROZEN_LAYOUT_KEYS, "provider layout")
    if (
        layout["schema_version"] != FROZEN_LAYOUT_SCHEMA_VERSION
        or layout["status"] != "FROZEN"
        or layout["purpose"] != FROZEN_LAYOUT_PURPOSE
    ):
        raise BridgeError("provider layout identity is not an authorized FROZEN layout")
    if not isinstance(layout["frozen_utc"], str) or not layout["frozen_utc"].strip():
        raise BridgeError("provider layout frozen_utc is missing")

    diagnostic = _exact_dict(layout["diagnostic"], DIAGNOSTIC_KEYS, "diagnostic")
    if not isinstance(diagnostic["job_id"], str) or not diagnostic["job_id"].isdigit():
        raise BridgeError("diagnostic.job_id must be numeric")
    if not isinstance(diagnostic["hostname"], str) or not diagnostic["hostname"]:
        raise BridgeError("diagnostic.hostname is missing")
    _digest(diagnostic["receipt_sha256"], "diagnostic.receipt_sha256")
    _digest(diagnostic["raw_sha256"], "diagnostic.raw_sha256")
    _posix_path(diagnostic["raw_recorded_path"], "diagnostic.raw_recorded_path", ("/",))

    hardware = _exact_dict(layout["hardware"], HARDWARE_KEYS, "hardware")
    if (
        hardware["partition"] != "h100"
        or hardware["gpu_model"] != "NVIDIA H100"
        or hardware["gpu_count"] != 4
        or hardware["hca"] != "mlx5_0"
        or hardware["port"] != "1"
        or hardware["link_layer"] != "InfiniBand"
        or hardware["minimum_rate_gbps"] != 400
        or not isinstance(hardware["observed_rate_gbps"], (int, float))
        or isinstance(hardware["observed_rate_gbps"], bool)
        or float(hardware["observed_rate_gbps"]) < 400.0
        or "NDR" not in str(hardware["rate"]).upper()
        or not str(hardware["state"]).startswith("4: ACTIVE")
        or not str(hardware["physical_state"]).startswith("5: LinkUp")
    ):
        raise BridgeError("frozen hardware is not four H100 GPUs with active 400-Gb/s NDR")

    libibverbs = _exact_dict(layout["libibverbs"], LIBIBVERBS_KEYS, "libibverbs")
    if libibverbs["bundle_name"] != "libibverbs.so.1":
        raise BridgeError("libibverbs.bundle_name must be libibverbs.so.1")
    _validate_source_identity(
        libibverbs["source"],
        "libibverbs.source",
        roots=LIBRARY_ROOTS,
        filename_pattern=LIBIBVERBS_PATTERN,
    )

    provider = _exact_dict(layout["provider"], PROVIDER_KEYS, "provider")
    if provider["name"] != PROVIDER_NAME or provider["ibv_drivers"] != PROVIDER_NAME:
        raise BridgeError("provider name and IBV_DRIVERS must be mlx5")
    _validate_source_identity(
        provider["source"],
        "provider.source",
        roots=LIBRARY_ROOTS,
        filename_pattern=PROVIDER_SOURCE_PATTERN,
    )
    loader_filename = provider["loader_filename"]
    loader_target = _posix_path(
        provider["loader_target_path"],
        "provider.loader_target_path",
        ("/usr/lib64/libibverbs",),
    )
    filename_match = (
        PROVIDER_LOADER_PATTERN.fullmatch(loader_filename)
        if isinstance(loader_filename, str)
        else None
    )
    target_match = PROVIDER_LOADER_TARGET_PATTERN.fullmatch(loader_target)
    if (
        filename_match is None
        or target_match is None
        or PurePosixPath(loader_target).name != loader_filename
        or filename_match.group("abi") != target_match.group("abi")
    ):
        raise BridgeError("provider loader filename/target is not an authorized mlx5 ABI path")

    config = _exact_dict(layout["provider_config"], PROVIDER_CONFIG_KEYS, "provider_config")
    if (
        config["driver"] != PROVIDER_NAME
        or config["active_lines"] != ["driver mlx5"]
        or config["container_target_path"] != str(PROVIDER_CONFIG_CONTAINER_PATH)
    ):
        raise BridgeError("provider config is not the exact mlx5 configuration")
    config_source = _validate_source_identity(
        config["source"], "provider_config.source", roots=CONFIG_ROOTS
    )
    if PurePosixPath(config_source["requested_path"]).name != "mlx5.driver":
        raise BridgeError("provider config source must be mlx5.driver")
    if not isinstance(layout["claim_boundary"], str) or not layout["claim_boundary"].strip():
        raise BridgeError("provider layout claim boundary is missing")

    identity = {
        key: layout[key]
        for key in (
            "diagnostic",
            "hardware",
            "libibverbs",
            "provider",
            "provider_config",
            "claim_boundary",
        )
    }
    expected_layout_digest = _digest(
        layout["layout_digest_sha256"], "layout_digest_sha256"
    )
    if _canonical_sha256(identity) != expected_layout_digest:
        raise BridgeError("provider layout canonical digest differs")
    return layout


def _load_provider_layout(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BridgeError(f"cannot read frozen provider layout {path}: {exc}") from exc
    layout_file_sha256 = hashlib.sha256(raw).hexdigest()
    if layout_file_sha256 != AUTHORIZED_LAYOUT_FILE_SHA256:
        raise BridgeError("frozen provider-layout file is not the authorized byte snapshot")
    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeError(f"cannot parse frozen provider layout {path}: {exc}") from exc
    layout = _validate_frozen_layout(payload)
    if layout["layout_digest_sha256"] != AUTHORIZED_LAYOUT_DIGEST_SHA256:
        raise BridgeError("frozen provider-layout identity digest is not authorized")
    return layout, layout_file_sha256


def _secure_source(
    path: Path, *, allowed_roots: Sequence[Path] = (Path("/usr/lib64"),)
) -> tuple[Path, os.stat_result]:
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise BridgeError(f"cannot resolve host RDMA source {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise BridgeError(f"host RDMA source is not a regular file: {resolved}")
    if metadata.st_uid != 0 or metadata.st_mode & 0o022:
        raise BridgeError(f"host RDMA source is not root-owned and non-writable: {resolved}")
    resolved_roots: list[Path] = []
    for root in allowed_roots:
        try:
            resolved_roots.append(root.resolve(strict=True))
        except OSError:
            continue
    if not any(
        resolved == root or resolved.is_relative_to(root) for root in resolved_roots
    ):
        roots = ", ".join(str(item) for item in allowed_roots)
        raise BridgeError(f"host RDMA source escapes allowlisted roots ({roots}): {resolved}")
    return resolved, metadata


def _authenticate_frozen_source(
    identity: Mapping[str, Any],
    label: str,
    *,
    allowed_roots: Sequence[Path],
) -> tuple[Path, os.stat_result]:
    requested = Path(str(identity["requested_path"]))
    source, metadata = _secure_source(requested, allowed_roots=allowed_roots)
    if str(source) != identity["real_path"]:
        raise BridgeError(f"{label} real path differs from the frozen layout")
    if (
        metadata.st_uid != identity["uid"]
        or metadata.st_gid != identity["gid"]
        or oct(stat.S_IMODE(metadata.st_mode)) != identity["mode"]
        or metadata.st_size != identity["bytes"]
        or _sha256(source) != identity["sha256"]
    ):
        raise BridgeError(f"{label} live identity differs from the frozen layout")
    return source, metadata


def _authenticate_runtime_dependency(
    soname: str, requested_path: str
) -> tuple[Path, os.stat_result]:
    if EXPECTED_RUNTIME_DEPENDENCIES.get(soname) != requested_path:
        raise BridgeError(f"runtime dependency {soname} is not in the fixed closure")
    requested = Path(requested_path)
    source, metadata = _secure_source(requested, allowed_roots=(Path("/usr/lib64"),))
    return source, metadata


def _needed(path: Path) -> list[str]:
    completed = subprocess.run(
        ["readelf", "--wide", "--dynamic", str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise BridgeError(f"readelf failed for {path}: {completed.stderr.strip()}")
    return [match.group("name") for match in NEEDED_PATTERN.finditer(completed.stdout)]


def _copy_source(
    source_path: Path,
    metadata: os.stat_result,
    target: Path,
    *,
    source_path_label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    before = _sha256(source_path)
    if expected_sha256 is not None and before != expected_sha256:
        raise BridgeError(f"{source_path_label} hash changed before copy")
    if expected_bytes is not None and metadata.st_size != expected_bytes:
        raise BridgeError(f"{source_path_label} byte count changed before copy")
    shutil.copyfile(source_path, target)
    os.chmod(target, 0o444)
    after_source = _sha256(source_path)
    copied = _sha256(target)
    if before != after_source or before != copied or target.stat().st_size != metadata.st_size:
        raise BridgeError(f"{source_path_label} changed while copying")
    return {
        "name": target.name,
        "source_path": source_path_label,
        "real_source_path": str(source_path),
        "source_uid": int(metadata.st_uid),
        "source_gid": int(metadata.st_gid),
        "source_mode": oct(stat.S_IMODE(metadata.st_mode)),
        "bytes": int(target.stat().st_size),
        "sha256": copied,
    }


def _write_manifest(bundle: Path, manifest_path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    with (bundle / "bridge_manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(bundle / "bridge_manifest.json", 0o444)


def _prepare_provider_bundle(
    bundle: Path, manifest_path: Path, provider_layout_path: Path
) -> dict[str, Any]:
    layout, layout_file_sha256 = _load_provider_layout(provider_layout_path)
    lib_identity = layout["libibverbs"]["source"]
    provider_identity = layout["provider"]["source"]
    loader_filename = layout["provider"]["loader_filename"]
    config_identity = layout["provider_config"]["source"]

    lib_source, lib_metadata = _authenticate_frozen_source(
        lib_identity, "libibverbs", allowed_roots=(Path("/usr/lib64"),)
    )
    provider_source, provider_metadata = _authenticate_frozen_source(
        provider_identity, "provider", allowed_roots=(Path("/usr/lib64"),)
    )
    config_source, config_metadata = _authenticate_frozen_source(
        config_identity,
        "provider config",
        allowed_roots=(Path("/etc/libibverbs.d"), Path("/usr/share/libibverbs.d")),
    )
    try:
        config_text = config_source.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeError(f"cannot read authenticated provider config: {exc}") from exc
    config_lines = [
        line.strip()
        for line in config_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if config_lines != layout["provider_config"]["active_lines"]:
        raise BridgeError("live provider config differs from frozen active lines")

    observed_main_needed = {
        "libibverbs": _validate_needed_list(
            _needed(lib_source), "observed libibverbs DT_NEEDED"
        ),
        "provider": _validate_needed_list(
            _needed(provider_source), "observed provider DT_NEEDED"
        ),
    }
    if not set(EXPECTED_RUNTIME_DEPENDENCIES).issubset(observed_main_needed["libibverbs"]):
        raise BridgeError("libibverbs omits the exact fixed libnl dependency closure")
    if any(
        soname not in set(EXPECTED_RUNTIME_DEPENDENCIES) | CONTAINER_DEPENDENCIES
        for soname in observed_main_needed["libibverbs"]
    ):
        raise BridgeError("libibverbs DT_NEEDED escapes the fixed closure")
    if "libibverbs.so.1" not in observed_main_needed["provider"] or any(
        soname not in {"libibverbs.so.1"} | CONTAINER_DEPENDENCIES
        for soname in observed_main_needed["provider"]
    ):
        raise BridgeError("provider DT_NEEDED escapes the fixed closure")

    dependency_sources: dict[str, tuple[Path, os.stat_result, str]] = {}
    dependency_needed: dict[str, list[str]] = {}
    for soname, requested_path in EXPECTED_RUNTIME_DEPENDENCIES.items():
        source, metadata = _authenticate_runtime_dependency(soname, requested_path)
        needed = _validate_needed_list(_needed(source), f"observed {soname} DT_NEEDED")
        if any(
            item not in set(EXPECTED_RUNTIME_DEPENDENCIES) | CONTAINER_DEPENDENCIES
            for item in needed
        ):
            raise BridgeError(f"{soname} DT_NEEDED escapes the fixed closure")
        dependency_sources[soname] = (source, metadata, requested_path)
        dependency_needed[soname] = needed

    library_dir = bundle / "lib"
    provider_config_dir = bundle / "etc" / "libibverbs.d"
    library_dir.mkdir(mode=0o700)
    provider_config_dir.mkdir(mode=0o700, parents=True)

    records: dict[str, dict[str, Any]] = {}
    lib_bundle_name = layout["libibverbs"]["bundle_name"]
    records[lib_bundle_name] = _copy_source(
        lib_source,
        lib_metadata,
        library_dir / lib_bundle_name,
        source_path_label=lib_identity["requested_path"],
        expected_sha256=lib_identity["sha256"],
        expected_bytes=lib_identity["bytes"],
    )
    records[LIBIBVERBS_ALIAS_BUNDLE_NAME] = _copy_source(
        lib_source,
        lib_metadata,
        library_dir / LIBIBVERBS_ALIAS_BUNDLE_NAME,
        source_path_label=lib_identity["real_path"],
        expected_sha256=lib_identity["sha256"],
        expected_bytes=lib_identity["bytes"],
    )
    for name in (
        PROVIDER_CANONICAL_BUNDLE_NAME,
        PROVIDER_ALIAS_BUNDLE_NAME,
        loader_filename,
    ):
        records[name] = _copy_source(
            provider_source,
            provider_metadata,
            library_dir / name,
            source_path_label=provider_identity["real_path"],
            expected_sha256=provider_identity["sha256"],
            expected_bytes=provider_identity["bytes"],
        )
    dependency_closure: dict[str, dict[str, Any]] = {}
    for soname in sorted(dependency_sources):
        source, metadata, requested_path = dependency_sources[soname]
        record = _copy_source(
            source,
            metadata,
            library_dir / soname,
            source_path_label=requested_path,
        )
        records[soname] = record
        dependency_closure[soname] = {
            "requested_path": requested_path,
            "real_path": record["real_source_path"],
            "uid": record["source_uid"],
            "gid": record["source_gid"],
            "mode": record["source_mode"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }

    config_bundle_path = provider_config_dir / "mlx5.driver"
    config_record = _copy_source(
        config_source,
        config_metadata,
        config_bundle_path,
        source_path_label=config_identity["requested_path"],
        expected_sha256=config_identity["sha256"],
        expected_bytes=config_identity["bytes"],
    )

    runtime_bind_plan = [
        {
            "source": provider_identity["real_path"],
            "bundle_path": str(library_dir / loader_filename),
            "container_path": layout["provider"]["loader_target_path"],
            "kind": "provider-loader",
            "sha256": provider_identity["sha256"],
            "bytes": provider_identity["bytes"],
            "read_only": True,
        },
        {
            "source": config_identity["real_path"],
            "bundle_path": str(config_bundle_path),
            "container_path": layout["provider_config"]["container_target_path"],
            "kind": "provider-config",
            "sha256": config_identity["sha256"],
            "bytes": config_identity["bytes"],
            "read_only": True,
        },
    ]
    dt_needed = {
        lib_bundle_name: observed_main_needed["libibverbs"],
        PROVIDER_CANONICAL_BUNDLE_NAME: observed_main_needed["provider"],
        **dependency_needed,
    }
    frozen_reference = {
        "staged_name": provider_layout_path.name,
        "path": str(provider_layout_path.resolve()),
        "sha256": layout_file_sha256,
        "layout_digest_sha256": layout["layout_digest_sha256"],
        "diagnostic": {
            key: layout["diagnostic"][key]
            for key in ("job_id", "receipt_sha256", "raw_sha256")
        },
    }
    digest_identity = {
        "frozen_layout": {
            "sha256": layout_file_sha256,
            "layout_digest_sha256": layout["layout_digest_sha256"],
        },
        "bundle_files": {
            name: {
                "source_path": record["source_path"],
                "real_source_path": record["real_source_path"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
            for name, record in sorted(records.items())
        },
        "provider_config": {
            "source_path": config_record["source_path"],
            "real_source_path": config_record["real_source_path"],
            "bytes": config_record["bytes"],
            "sha256": config_record["sha256"],
        },
        "dt_needed": dt_needed,
        "fixed_dependency_closure": dependency_closure,
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
            for entry in runtime_bind_plan
        ],
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "purpose": BRIDGE_PURPOSE,
        "hostname": socket.gethostname().split(".", 1)[0].lower(),
        "bundle_root": str(bundle),
        "frozen_layout": frozen_reference,
        "hardware": layout["hardware"],
        "libibverbs": layout["libibverbs"],
        "provider": layout["provider"],
        "provider_config": layout["provider_config"],
        "bundle_files": records,
        "dt_needed": dt_needed,
        "fixed_dependency_closure": dependency_closure,
        "container_supplied_dependencies": sorted(CONTAINER_DEPENDENCIES),
        "runtime_bind_plan": runtime_bind_plan,
        "bundle_digest_sha256": _canonical_sha256(digest_identity),
        "failures": [],
    }
    _write_manifest(bundle, manifest_path, payload)
    os.chmod(provider_config_dir, 0o555)
    os.chmod(provider_config_dir.parent, 0o555)
    os.chmod(library_dir, 0o555)
    os.chmod(bundle, 0o555)
    return payload


def _dependency_path(soname: str) -> Path:
    candidate = Path("/usr/lib64") / soname
    if not candidate.exists():
        raise BridgeError(f"allowlisted host RDMA dependency is missing: {soname}")
    return candidate


def _prepare_core_bundle(bundle: Path, manifest_path: Path) -> dict[str, Any]:
    library_dir = bundle / "lib"
    library_dir.mkdir(mode=0o700)
    sources = dict(CORE_SEED_LIBRARIES)
    queue = list(sources)
    observed_dependencies: dict[str, list[str]] = {}
    while queue:
        soname = queue.pop(0)
        source, _ = _secure_source(sources[soname])
        dependencies = _needed(source)
        observed_dependencies[soname] = dependencies
        for dependency in dependencies:
            if dependency in CONTAINER_DEPENDENCIES or dependency in sources:
                continue
            if dependency not in COPYABLE_DEPENDENCIES:
                raise BridgeError(
                    f"unapproved DT_NEEDED dependency {dependency!r} from {soname}"
                )
            sources[dependency] = _dependency_path(dependency)
            queue.append(dependency)

    records: dict[str, dict[str, Any]] = {}
    for soname, requested in sorted(sources.items()):
        source, metadata = _secure_source(requested)
        records[soname] = _copy_source(
            source,
            metadata,
            library_dir / soname,
            source_path_label=str(requested),
        )
    for alias, canonical in CORE_ALIASES.items():
        canonical_record = records[canonical]
        source = Path(canonical_record["real_source_path"])
        metadata = source.stat()
        records[alias] = _copy_source(
            source,
            metadata,
            library_dir / alias,
            source_path_label=str(source),
        )
    bundle_identity = {
        name: {"bytes": record["bytes"], "sha256": record["sha256"]}
        for name, record in sorted(records.items())
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "purpose": "tacc_node_local_rdma_core_sidecar",
        "hostname": socket.gethostname().split(".", 1)[0].lower(),
        "bundle_root": str(bundle),
        "bundle_files": records,
        "dt_needed": observed_dependencies,
        "container_supplied_dependencies": sorted(CONTAINER_DEPENDENCIES),
        "bundle_digest_sha256": _canonical_sha256(bundle_identity),
        "failures": [],
    }
    _write_manifest(bundle, manifest_path, payload)
    os.chmod(library_dir, 0o555)
    os.chmod(bundle, 0o555)
    return payload


def prepare_bundle(
    bundle: Path,
    manifest_path: Path,
    *,
    include_provider: bool = True,
    provider_layout: Path | None = None,
) -> dict[str, Any]:
    bundle = bundle.resolve()
    manifest_path = manifest_path.resolve()
    if not bundle.is_dir() or bundle.is_symlink() or any(bundle.iterdir()):
        raise BridgeError("RDMA sidecar bundle must be an existing empty real directory")
    if manifest_path.exists():
        raise BridgeError(f"refusing to overwrite bridge manifest: {manifest_path}")
    if include_provider:
        if provider_layout is None:
            raise BridgeError("full RDMA sidecar requires --provider-layout")
        return _prepare_provider_bundle(
            bundle, manifest_path, provider_layout.resolve(strict=True)
        )
    if provider_layout is not None:
        raise BridgeError("prepare-core does not accept a provider layout")
    return _prepare_core_bundle(bundle, manifest_path)


def _load_bridge_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeError(f"cannot read bridge manifest {path}: {exc}") from exc
    manifest = _exact_dict(value, BRIDGE_MANIFEST_KEYS, "bridge manifest")
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["status"] != "PASS"
        or manifest["purpose"] != BRIDGE_PURPOSE
        or manifest["failures"] != []
    ):
        raise BridgeError("bridge manifest is not a clean full-provider PASS")
    _digest(manifest["bundle_digest_sha256"], "bundle_digest_sha256")
    _exact_dict(manifest["frozen_layout"], FROZEN_LAYOUT_REFERENCE_KEYS, "frozen_layout")
    _exact_dict(
        manifest["frozen_layout"]["diagnostic"],
        FROZEN_DIAGNOSTIC_REFERENCE_KEYS,
        "frozen_layout.diagnostic",
    )
    return manifest


def _validated_runtime_plan(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load_bridge_manifest(manifest_path)
    bundle_root = Path(str(manifest["bundle_root"]))
    if not bundle_root.is_absolute():
        raise BridgeError("bridge bundle root is not absolute")
    plan = manifest["runtime_bind_plan"]
    if not isinstance(plan, list) or len(plan) != 2:
        raise BridgeError("runtime bind plan must contain exactly two entries")
    records: dict[str, dict[str, Any]] = {}
    targets: set[str] = set()
    bundle_paths: set[str] = set()
    for index, raw in enumerate(plan):
        record = _exact_dict(raw, RUNTIME_BIND_KEYS, f"runtime_bind_plan[{index}]")
        kind = record["kind"]
        if kind not in {"provider-loader", "provider-config"} or kind in records:
            raise BridgeError("runtime bind plan kinds are unexpected or duplicated")
        if record["read_only"] is not True:
            raise BridgeError("runtime bind plan must be read-only")
        source = _posix_path(record["source"], f"runtime_bind_plan[{index}].source", ("/",))
        target = _posix_path(
            record["container_path"], f"runtime_bind_plan[{index}].container_path", ("/",)
        )
        digest = _digest(record["sha256"], f"runtime_bind_plan[{index}].sha256")
        byte_count = _positive_int(record["bytes"], f"runtime_bind_plan[{index}].bytes")
        bundle_path = Path(str(record["bundle_path"]))
        try:
            bundle_path.resolve(strict=True).relative_to(bundle_root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise BridgeError("runtime bind source escapes the sidecar bundle") from exc
        if (
            not bundle_path.is_file()
            or bundle_path.stat().st_size != byte_count
            or _sha256(bundle_path) != digest
        ):
            raise BridgeError(f"runtime bind file for {kind} is not authenticated")
        if target in targets or str(bundle_path) in bundle_paths:
            raise BridgeError("runtime bind targets or bundle sources are duplicated")
        targets.add(target)
        bundle_paths.add(str(bundle_path))
        records[kind] = {
            **record,
            "source": source,
            "container_path": target,
        }
    if set(records) != {"provider-loader", "provider-config"}:
        raise BridgeError("runtime bind plan is incomplete")
    provider = manifest["provider"]
    config = manifest["provider_config"]
    provider_plan = records["provider-loader"]
    config_plan = records["provider-config"]
    if (
        provider_plan["source"] != provider["source"]["real_path"]
        or provider_plan["container_path"] != provider["loader_target_path"]
        or Path(provider_plan["bundle_path"]).name != provider["loader_filename"]
        or provider_plan["sha256"] != provider["source"]["sha256"]
        or provider_plan["bytes"] != provider["source"]["bytes"]
        or config_plan["source"] != config["source"]["real_path"]
        or config_plan["container_path"] != config["container_target_path"]
        or config_plan["sha256"] != config["source"]["sha256"]
        or config_plan["bytes"] != config["source"]["bytes"]
    ):
        raise BridgeError("runtime bind plan differs from frozen provider/config identities")
    return manifest, [records["provider-loader"], records["provider-config"]]


def render_bind_plan(manifest_path: Path) -> str:
    _, plan = _validated_runtime_plan(manifest_path)
    lines: list[str] = []
    for record in plan:
        fields = (
            record["kind"],
            record["bundle_path"],
            record["container_path"],
            record["source"],
            record["sha256"],
            str(record["bytes"]),
            "true",
        )
        if any(SAFE_FIELD.fullmatch(str(field)) is None for field in fields):
            raise BridgeError("runtime bind projection contains an unsafe field")
        lines.append("\t".join(str(field) for field in fields))
    return "\n".join(lines) + "\n"


def render_runtime_metadata(manifest_path: Path) -> str:
    manifest, _ = _validated_runtime_plan(manifest_path)
    reference = manifest["frozen_layout"]
    fields = (
        reference["path"],
        reference["sha256"],
        reference["layout_digest_sha256"],
        manifest["provider"]["ibv_drivers"],
    )
    if any(SAFE_FIELD.fullmatch(str(field)) is None for field in fields):
        raise BridgeError("runtime metadata projection contains an unsafe field")
    return "\t".join(str(field) for field in fields) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--bundle", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--provider-layout", type=Path, required=True)
    prepare_core = subparsers.add_parser("prepare-core")
    prepare_core.add_argument("--bundle", type=Path, required=True)
    prepare_core.add_argument("--manifest", type=Path, required=True)
    render = subparsers.add_parser("render-bind-plan")
    render.add_argument("--manifest", type=Path, required=True)
    metadata = subparsers.add_parser("render-runtime-metadata")
    metadata.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            payload = prepare_bundle(
                args.bundle,
                args.manifest,
                provider_layout=args.provider_layout,
            )
            print(
                json.dumps(
                    {
                        "bundle_digest_sha256": payload["bundle_digest_sha256"],
                        "layout_digest_sha256": payload["frozen_layout"][
                            "layout_digest_sha256"
                        ],
                        "manifest": str(args.manifest.resolve()),
                        "status": "PASS",
                    },
                    sort_keys=True,
                )
            )
        elif args.command == "prepare-core":
            payload = prepare_bundle(
                args.bundle,
                args.manifest,
                include_provider=False,
            )
            print(
                json.dumps(
                    {
                        "bundle_digest_sha256": payload["bundle_digest_sha256"],
                        "manifest": str(args.manifest.resolve()),
                        "status": "PASS",
                    },
                    sort_keys=True,
                )
            )
        elif args.command == "render-bind-plan":
            sys.stdout.write(render_bind_plan(args.manifest))
        else:
            sys.stdout.write(render_runtime_metadata(args.manifest))
    except (BridgeError, OSError, subprocess.SubprocessError) as exc:
        print(f"RDMA_BRIDGE_ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
