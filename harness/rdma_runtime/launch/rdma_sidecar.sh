#!/bin/bash
# Shared fail-closed RDMA sidecar setup for gate, pilot, and evidence workers.

prepare_acm_rdma_sidecar() {
    if [[ "$#" -ne 1 ]]; then
        echo "usage: prepare_acm_rdma_sidecar OUTPUT_DIR" >&2
        return 2
    fi
    local output_dir="$1"
    local authorized_layout_source="${SOURCE_ROOT}/configs/tacc_h100_rdma_layout.json"
    local provider_layout="${ACM_RDMA_PROVIDER_LAYOUT_INPUT:-${SOURCE_ROOT}/configs/tacc_h100_rdma_layout.json}"
    local authorized_layout_file_sha256="2f11d9c1a84e0441d5a7c2333c6d452308dbe77f18e457f90db8ee48db6f8209"
    local authorized_layout_digest_sha256="8e15b75a379512d500a8d0dfa81d00bb40b00f28184af53b573c1603a8a6f978"
    local source_layout_sha256
    local staged_layout_sha256
    local staged_layout_digest_sha256
    local hostname_short
    local template
    local bind_plan
    local runtime_metadata
    local line
    local kind
    local bundle_path
    local container_path
    local source_path
    local expected_sha256
    local expected_bytes
    local read_only
    local extra
    local observed_sha256
    local observed_bytes
    local observed_mode
    local provider_seen=0
    local config_seen=0
    local -a bind_lines=()
    if [[ ! -d "${output_dir}" ]]; then
        echo "RDMA receipt output directory is missing: ${output_dir}" >&2
        return 2
    fi
    if [[ ! -s "${authorized_layout_source}" || ! -s "${provider_layout}" ]]; then
        echo "Authorized or staged RDMA provider layout is missing" >&2
        return 2
    fi
    if ! source_layout_sha256="$(sha256sum "${authorized_layout_source}" | awk '{print $1}')" \
        || ! staged_layout_sha256="$(sha256sum "${provider_layout}" | awk '{print $1}')" \
        || ! staged_layout_digest_sha256="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["layout_digest_sha256"])' "${provider_layout}")"; then
        echo "Could not authenticate the staged RDMA provider layout" >&2
        return 2
    fi
    if [[ "${source_layout_sha256}" != "${authorized_layout_file_sha256}" \
          || "${staged_layout_sha256}" != "${authorized_layout_file_sha256}" \
          || "${staged_layout_digest_sha256}" != "${authorized_layout_digest_sha256}" ]] \
        || ! cmp -s "${authorized_layout_source}" "${provider_layout}"; then
        echo "Staged RDMA provider layout differs from the authorized byte snapshot" >&2
        return 2
    fi
    hostname_short="$(hostname -s | tr '[:upper:]' '[:lower:]')"
    template="${SLURM_TMPDIR:-/tmp}/acm-rdma-${SLURM_JOB_ID}-${SLURM_STEP_ID:-nostep}-${SLURM_NODEID}.XXXXXX"
    ACM_RDMA_SIDECAR_ROOT="$(mktemp -d "${template}")"
    ACM_RDMA_BRIDGE_MANIFEST="${output_dir}/rdma-host-bridge-${hostname_short}.json"
    if ! python3 "${SOURCE_ROOT}/tacc_rdma_bridge.py" prepare \
        --bundle "${ACM_RDMA_SIDECAR_ROOT}" \
        --manifest "${ACM_RDMA_BRIDGE_MANIFEST}" \
        --provider-layout "${provider_layout}"; then
        return 2
    fi
    ACM_RDMA_LIBRARY_PATH="${ACM_RDMA_SIDECAR_ROOT}/lib:/.singularity.d/libs"
    if ! bind_plan="$(python3 "${SOURCE_ROOT}/tacc_rdma_bridge.py" render-bind-plan \
        --manifest "${ACM_RDMA_BRIDGE_MANIFEST}")"; then
        return 2
    fi
    mapfile -t bind_lines <<< "${bind_plan}"
    if [[ "${#bind_lines[@]}" -ne 2 ]]; then
        echo "Authenticated RDMA bind plan must contain exactly two entries" >&2
        return 2
    fi
    for line in "${bind_lines[@]}"; do
        IFS=$'\t' read -r kind bundle_path container_path source_path \
            expected_sha256 expected_bytes read_only extra <<< "${line}"
        if [[ -n "${extra}" || "${read_only}" != "true" \
              || ! "${expected_sha256}" =~ ^[0-9a-f]{64}$ \
              || ! "${expected_bytes}" =~ ^[1-9][0-9]*$ \
              || "${bundle_path}" != "${ACM_RDMA_SIDECAR_ROOT}/"* \
              || "${container_path}" != /* || "${source_path}" != /* \
              || ! -f "${bundle_path}" || -L "${bundle_path}" ]]; then
            echo "Authenticated RDMA bind entry is malformed: ${kind}" >&2
            return 2
        fi
        observed_sha256="$(sha256sum "${bundle_path}" | awk '{print $1}')"
        observed_bytes="$(stat -c '%s' "${bundle_path}")"
        observed_mode="$(stat -c '%a' "${bundle_path}")"
        if [[ "${observed_sha256}" != "${expected_sha256}" \
              || "${observed_bytes}" != "${expected_bytes}" \
              || "${observed_mode}" != "444" ]]; then
            echo "Authenticated RDMA bind file changed after bridge creation: ${bundle_path}" >&2
            return 2
        fi
        case "${kind}" in
            provider-loader)
                if [[ "${provider_seen}" -ne 0 ]]; then
                    echo "Duplicate RDMA provider-loader bind" >&2
                    return 2
                fi
                ACM_RDMA_PROVIDER_SOURCE="${bundle_path}"
                ACM_RDMA_PROVIDER_TARGET="${container_path}"
                provider_seen=1
                ;;
            provider-config)
                if [[ "${config_seen}" -ne 0 ]]; then
                    echo "Duplicate RDMA provider-config bind" >&2
                    return 2
                fi
                ACM_RDMA_PROVIDER_CONFIG_SOURCE="${bundle_path}"
                ACM_RDMA_PROVIDER_CONFIG_TARGET="${container_path}"
                config_seen=1
                ;;
            *)
                echo "Unexpected authenticated RDMA bind kind: ${kind}" >&2
                return 2
                ;;
        esac
    done
    if [[ "${provider_seen}" -ne 1 || "${config_seen}" -ne 1 ]]; then
        echo "Authenticated RDMA bind plan is incomplete" >&2
        return 2
    fi
    if ! runtime_metadata="$(python3 "${SOURCE_ROOT}/tacc_rdma_bridge.py" \
        render-runtime-metadata --manifest "${ACM_RDMA_BRIDGE_MANIFEST}")"; then
        return 2
    fi
    IFS=$'\t' read -r ACM_RDMA_PROVIDER_LAYOUT ACM_RDMA_PROVIDER_LAYOUT_SHA256 \
        ACM_RDMA_LAYOUT_DIGEST_SHA256 ACM_RDMA_DRIVER_PATH extra <<< "${runtime_metadata}"
    if [[ -n "${extra}" || "${ACM_RDMA_PROVIDER_LAYOUT}" != "${provider_layout}" \
          || ! "${ACM_RDMA_PROVIDER_LAYOUT_SHA256}" =~ ^[0-9a-f]{64}$ \
          || ! "${ACM_RDMA_LAYOUT_DIGEST_SHA256}" =~ ^[0-9a-f]{64}$ \
          || "${ACM_RDMA_PROVIDER_LAYOUT_SHA256}" != "${authorized_layout_file_sha256}" \
          || "${ACM_RDMA_LAYOUT_DIGEST_SHA256}" != "${authorized_layout_digest_sha256}" \
          || "${ACM_RDMA_DRIVER_PATH}" != "mlx5" \
          || "$(sha256sum "${provider_layout}" | awk '{print $1}')" \
             != "${ACM_RDMA_PROVIDER_LAYOUT_SHA256}" ]]; then
        echo "Authenticated RDMA runtime metadata is invalid" >&2
        return 2
    fi
    ACM_RDMA_CONTAINER_BINDS="${WORK}:${WORK},${SCRATCH}:${SCRATCH},${ACM_RDMA_SIDECAR_ROOT}:${ACM_RDMA_SIDECAR_ROOT}:ro,${ACM_RDMA_PROVIDER_SOURCE}:${ACM_RDMA_PROVIDER_TARGET}:ro,${ACM_RDMA_PROVIDER_CONFIG_SOURCE}:${ACM_RDMA_PROVIDER_CONFIG_TARGET}:ro"
    export ACM_RDMA_SIDECAR_ROOT
    export ACM_RDMA_BRIDGE_MANIFEST
    export ACM_RDMA_LIBRARY_PATH
    export ACM_RDMA_DRIVER_PATH
    export ACM_RDMA_PROVIDER_SOURCE
    export ACM_RDMA_PROVIDER_TARGET
    export ACM_RDMA_PROVIDER_CONFIG_SOURCE
    export ACM_RDMA_PROVIDER_CONFIG_TARGET
    export ACM_RDMA_PROVIDER_LAYOUT
    export ACM_RDMA_PROVIDER_LAYOUT_SHA256
    export ACM_RDMA_LAYOUT_DIGEST_SHA256
    export ACM_RDMA_CONTAINER_BINDS
}

# Provider-neutral container/runtime smoke path for CPU-only release validation.
# This authenticates libibverbs and its dependency closure but deliberately does
# not require an mlx5 provider, driver config, HCA, or GDR capability.  Those
# hardware claims belong exclusively to the H100 interconnect gate.
prepare_acm_rdma_core_sidecar() {
    if [[ "$#" -ne 1 ]]; then
        echo "usage: prepare_acm_rdma_core_sidecar OUTPUT_DIR" >&2
        return 2
    fi
    local output_dir="$1"
    local hostname_short
    local template
    if [[ ! -d "${output_dir}" ]]; then
        echo "RDMA core receipt output directory is missing: ${output_dir}" >&2
        return 2
    fi
    hostname_short="$(hostname -s | tr '[:upper:]' '[:lower:]')"
    template="${SLURM_TMPDIR:-/tmp}/acm-rdma-core-${SLURM_JOB_ID}-${SLURM_STEP_ID:-nostep}-${SLURM_NODEID}.XXXXXX"
    ACM_RDMA_CORE_SIDECAR_ROOT="$(mktemp -d "${template}")"
    ACM_RDMA_CORE_BRIDGE_MANIFEST="${output_dir}/rdma-core-host-bridge-${hostname_short}.json"
    python3 "${SOURCE_ROOT}/tacc_rdma_bridge.py" prepare-core \
        --bundle "${ACM_RDMA_CORE_SIDECAR_ROOT}" \
        --manifest "${ACM_RDMA_CORE_BRIDGE_MANIFEST}"
    ACM_RDMA_CORE_LIBRARY_PATH="${ACM_RDMA_CORE_SIDECAR_ROOT}/lib:/.singularity.d/libs"
    if [[ ! -r "${ACM_RDMA_CORE_SIDECAR_ROOT}/lib/libibverbs.so.1" ]]; then
        echo "RDMA core sidecar libibverbs is missing" >&2
        return 2
    fi
    ACM_RDMA_CORE_LIBIBVERBS_SHA256="$(sha256sum "${ACM_RDMA_CORE_SIDECAR_ROOT}/lib/libibverbs.so.1" | awk '{print $1}')"
    if [[ ! "${ACM_RDMA_CORE_LIBIBVERBS_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "RDMA core sidecar libibverbs hash is invalid" >&2
        return 2
    fi
    ACM_RDMA_CORE_CONTAINER_BINDS="${WORK}:${WORK},${SCRATCH}:${SCRATCH},${ACM_RDMA_CORE_SIDECAR_ROOT}:${ACM_RDMA_CORE_SIDECAR_ROOT}:ro"
    export ACM_RDMA_CORE_SIDECAR_ROOT
    export ACM_RDMA_CORE_BRIDGE_MANIFEST
    export ACM_RDMA_CORE_LIBRARY_PATH
    export ACM_RDMA_CORE_LIBIBVERBS_SHA256
    export ACM_RDMA_CORE_CONTAINER_BINDS
}
