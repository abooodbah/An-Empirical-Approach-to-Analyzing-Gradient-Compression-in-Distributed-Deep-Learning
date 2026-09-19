# -*- coding: utf-8 -*-
"""Generate control_workload_benchmark.py from the frozen July harness.

Adds, without touching the archived file:
  * gpu-quant-reducible: shared-scale summable int8 exchanged by all-reduce
    (exact int8 sum by construction: floor(127/world_size) levels per side);
  * gpu-topk-localef: Top-k with standard local error feedback;
  * --timing-mode wall: uninstrumented end-to-end step timing (no explicit
    barrier, no per-phase device synchronization, one sync at step end).
Phase-mode behavior for the legacy modes is unchanged.
"""
import io
import os
import py_compile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "tacc_paper_workloads", "paper_workload_benchmark.py")
DST = os.path.join(HERE, "control_workload_benchmark.py")

s = io.open(SRC, encoding="utf-8").read()
n = [0]


def rep(old, new, count=1):
    global s
    found = s.count(old)
    assert found == count, "anchor x%d (want %d): %r" % (found, count, old[:90])
    s = s.replace(old, new)
    n[0] += 1


# --- constants -------------------------------------------------------------
rep('SCHEMA_VERSION = "tacc-paper-workloads/v2"',
    'SCHEMA_VERSION = "tacc-paper-workloads/v2.1-control"')

rep('COMM_MODES = LEGACY_MODES + ("ddp-overlap",)',
    'CONTROL_MODES = ("gpu-quant-reducible", "gpu-topk-localef")\n'
    'COMM_MODES = LEGACY_MODES + ("ddp-overlap",) + CONTROL_MODES')

rep('LEGACY_TOPK_RESIDUAL = "local_corrected_gradient_minus_global_average_reconstruction"',
    'LEGACY_TOPK_RESIDUAL = "local_corrected_gradient_minus_global_average_reconstruction"\n'
    'LOCALEF_TOPK_RESIDUAL = "standard_local_error_feedback_untransmitted_mass"\n'
    'WALL_SEMANTICS = "uninstrumented_wall_clock_step_no_phase_boundaries_no_explicit_barrier"')

rep('    "comm_mode",\n    "sync_every",',
    '    "comm_mode",\n    "timing_mode",\n    "sync_every",')

# --- CLI -------------------------------------------------------------------
rep('    parser.add_argument("--sync-every", type=int, default=1)\n',
    '    parser.add_argument("--sync-every", type=int, default=1)\n'
    '    parser.add_argument(\n'
    '        "--timing-mode",\n'
    '        choices=("phase", "wall"),\n'
    '        default="phase",\n'
    '        help=(\n'
    '            "phase: device-synchronized attribution (paper convention); "\n'
    '            "wall: uninstrumented end-to-end step timing only."\n'
    '        ),\n'
    '    )\n')

# --- deviations ------------------------------------------------------------
rep('    if args.comm_mode == "ddp-overlap":\n'
    '        deviations.append("supplemental_overlap_baseline_not_in_legacy_matrix")\n'
    '    return deviations',
    '    if args.comm_mode == "ddp-overlap":\n'
    '        deviations.append("supplemental_overlap_baseline_not_in_legacy_matrix")\n'
    '    if args.comm_mode in CONTROL_MODES:\n'
    '        deviations.append("control_mode_not_in_legacy_matrix")\n'
    '    if args.timing_mode != "phase":\n'
    '        deviations.append("uninstrumented_wall_clock_timing")\n'
    '    return deviations')

# --- measurement labels ----------------------------------------------------
rep('def measurement_labels(comm_mode: str) -> Tuple[str, str, str]:\n'
    '    if comm_mode == "ddp-overlap":',
    'def measurement_labels(comm_mode: str, timing_mode: str = "phase") -> Tuple[str, str, str]:\n'
    '    if timing_mode == "wall":\n'
    '        return (\n'
    '            "uninstrumented_control",\n'
    '            WALL_SEMANTICS,\n'
    '            "observed_step_only;no_phase_attribution",\n'
    '        )\n'
    '    if comm_mode == "ddp-overlap":')

rep('        continuity_status, semantics, decomposition = measurement_labels(args.comm_mode)\n'
    '        track = "overlap-baseline" if args.comm_mode == "ddp-overlap" else "legacy-continuity"',
    '        continuity_status, semantics, decomposition = measurement_labels(\n'
    '            args.comm_mode, args.timing_mode\n'
    '        )\n'
    '        if args.comm_mode in CONTROL_MODES or args.timing_mode == "wall":\n'
    '            track = "control-extension"\n'
    '        elif args.comm_mode == "ddp-overlap":\n'
    '            track = "overlap-baseline"\n'
    '        else:\n'
    '            track = "legacy-continuity"')

# --- validators ------------------------------------------------------------
rep('    if record["track"] not in ("legacy-continuity", "overlap-baseline"):',
    '    if record["track"] not in ("legacy-continuity", "overlap-baseline", "control-extension"):')

rep('    if record["mode"] != record["comm_mode"]:\n'
    '        raise ValueError("mode and comm_mode must agree")',
    '    if record["mode"] != record["comm_mode"]:\n'
    '        raise ValueError("mode and comm_mode must agree")\n'
    '    if record["timing_mode"] not in ("phase", "wall"):\n'
    '        raise ValueError("invalid timing mode in result")')

rep('    if record["comm_mode"] == "gpu-topk" and record["topk_residual_semantics"] != LEGACY_TOPK_RESIDUAL:\n'
    '        raise ValueError("Top-k rows must declare the retained legacy residual semantics")',
    '    if record["comm_mode"] == "gpu-topk" and record["topk_residual_semantics"] != LEGACY_TOPK_RESIDUAL:\n'
    '        raise ValueError("Top-k rows must declare the retained legacy residual semantics")\n'
    '    if record["comm_mode"] == "gpu-topk-localef" and record["topk_residual_semantics"] != LOCALEF_TOPK_RESIDUAL:\n'
    '        raise ValueError("local-EF Top-k rows must declare the standard residual semantics")')

rep('    required_finite = [\n'
    '        "forward_s", "barrier_wait_s", "serialize_s", "deserialize_s", "optimizer_step_s",\n'
    '        "observed_step_s", "phase_sum_s", "unattributed_s",\n'
    '    ]\n'
    '    if args.comm_mode == "ddp-overlap":\n'
    '        required_finite.append("backward_plus_comm_s")\n'
    '    elif args.comm_mode == "powersgd":\n'
    '        required_finite.extend(("backward_plus_comm_s", "backward_compute_s", "network_s"))\n'
    '    else:\n'
    '        required_finite.extend(("barrier_wait_s", "backward_compute_s", "serialize_s", "network_s", "deserialize_s"))',
    '    if args.timing_mode == "wall":\n'
    '        required_finite = ["observed_step_s"]\n'
    '    else:\n'
    '        required_finite = [\n'
    '            "forward_s", "barrier_wait_s", "serialize_s", "deserialize_s", "optimizer_step_s",\n'
    '            "observed_step_s", "phase_sum_s", "unattributed_s",\n'
    '        ]\n'
    '        if args.comm_mode == "ddp-overlap":\n'
    '            required_finite.append("backward_plus_comm_s")\n'
    '        elif args.comm_mode == "powersgd":\n'
    '            required_finite.extend(("backward_plus_comm_s", "backward_compute_s", "network_s"))\n'
    '        else:\n'
    '            required_finite.extend(("barrier_wait_s", "backward_compute_s", "serialize_s", "network_s", "deserialize_s"))')

# --- condition id ----------------------------------------------------------
rep("        condition_id = (\n"
    "            f\"{args.model}.{args.comm_mode}.ws{runtime.world_size}.tau{args.sync_every}.\"\n"
    "            f\"bs{args.batch_size}\"\n"
    "        )",
    "        condition_id = (\n"
    "            f\"{args.model}.{args.comm_mode}.ws{runtime.world_size}.tau{args.sync_every}.\"\n"
    "            f\"bs{args.batch_size}\"\n"
    "        )\n"
    "        if args.timing_mode != \"phase\":\n"
    "            condition_id += f\".{args.timing_mode}\"")

# --- wall-aware measurement helper ------------------------------------------
rep('        records: List[Dict[str, Any]] = []\n\n        for iteration in range(args.iters):',
    '        records: List[Dict[str, Any]] = []\n'
    '        wall_mode = args.timing_mode == "wall"\n'
    '\n'
    '        def measure_op(operation: Callable[[], Any]) -> Tuple[Any, float]:\n'
    '            if wall_mode:\n'
    '                return operation(), 0.0\n'
    '            return measure(runtime, operation)\n'
    '\n'
    '        for iteration in range(args.iters):')

# --- per-phase measure() calls become measure_op() --------------------------
rep('            (_, loss), forward_s = measure(runtime, forward)\n'
    '            loss_value = float(loss.detach().cpu().item())',
    '            (_, loss), forward_s = measure_op(forward)\n'
    '            loss_value = 0.0 if wall_mode else float(loss.detach().cpu().item())')

rep('            if args.comm_mode != "ddp-overlap":\n'
    '                _, barrier_wait_s = measure(runtime, lambda: _distributed_barrier(runtime))',
    '            if args.comm_mode != "ddp-overlap" and not wall_mode:\n'
    '                _, barrier_wait_s = measure(runtime, lambda: _distributed_barrier(runtime))')

rep('                _, backward_plus_comm_s = measure(runtime, loss.backward)',
    '                _, backward_plus_comm_s = measure_op(loss.backward)')
rep('                _, backward_compute_s = measure(runtime, loss.backward)',
    '                _, backward_compute_s = measure_op(loss.backward)')
rep('                    _, network_s = measure(runtime, lambda: dist.all_reduce(flat))',
    '                    _, network_s = measure_op(lambda: dist.all_reduce(flat))')
rep('                    _, deserialize_s = measure(runtime, average_and_copy)',
    '                    _, deserialize_s = measure_op(average_and_copy)')
rep('                    (scale, encoded), serialize_s = measure(runtime, lambda: gpu_quantize(flat))',
    '                    (scale, encoded), serialize_s = measure_op(lambda: gpu_quantize(flat))')
rep('                    (scales, payloads), network_s = measure(\n'
    '                        runtime, lambda: _collect_gpu_quant(scale, encoded, runtime)\n'
    '                    )',
    '                    (scales, payloads), network_s = measure_op(\n'
    '                        lambda: _collect_gpu_quant(scale, encoded, runtime)\n'
    '                    )')
rep('                    _, deserialize_s = measure(runtime, reconstruct_quantized)',
    '                    _, deserialize_s = measure_op(reconstruct_quantized)')
rep('                    payload, serialize_s = measure(runtime, lambda: cpu_quantize(flat))',
    '                    payload, serialize_s = measure_op(lambda: cpu_quantize(flat))')
rep('                    (outputs, lengths, maximum), network_s = measure(\n'
    '                        runtime, lambda: _collect_cpu_quant(payload, runtime)\n'
    '                    )',
    '                    (outputs, lengths, maximum), network_s = measure_op(\n'
    '                        lambda: _collect_cpu_quant(payload, runtime)\n'
    '                    )')
rep('                    _, deserialize_s = measure(runtime, reconstruct_cpu_quantized)',
    '                    _, deserialize_s = measure_op(reconstruct_cpu_quantized)')
rep('                    (indices, values), serialize_s = measure(\n'
    '                        runtime, lambda: gpu_topk(corrected, count)\n'
    '                    )',
    '                    (indices, values), serialize_s = measure_op(\n'
    '                        lambda: gpu_topk(corrected, count)\n'
    '                    )')
rep('                    (index_outputs, value_outputs, sizes, maximum), network_s = measure(\n'
    '                        runtime, lambda: _collect_topk(indices, values, runtime)\n'
    '                    )',
    '                    (index_outputs, value_outputs, sizes, maximum), network_s = measure_op(\n'
    '                        lambda: _collect_topk(indices, values, runtime)\n'
    '                    )')
rep('                    reconstructed, deserialize_s = measure(\n'
    '                        runtime,\n'
    '                        lambda: _reconstruct_topk(\n'
    '                            index_outputs, value_outputs, sizes, flat, runtime.world_size\n'
    '                        ),\n'
    '                    )',
    '                    reconstructed, deserialize_s = measure_op(\n'
    '                        lambda: _reconstruct_topk(\n'
    '                            index_outputs, value_outputs, sizes, flat, runtime.world_size\n'
    '                        )\n'
    '                    )')
rep('                    _, copy_s = measure(runtime, lambda: unflatten_and_copy(model, reconstructed))',
    '                    _, copy_s = measure_op(lambda: unflatten_and_copy(model, reconstructed))')

# --- the two control-mode branches ------------------------------------------
rep('                    legacy_bytes = indices.numel() * 12\n'
    '                    collective_input_bytes = 8 + maximum * 12\n'
    '                    compression_ratio = bytes_uncompressed / max(1, legacy_bytes)\n'
    '\n'
    '            _, optimizer_step_s',
    '                    legacy_bytes = indices.numel() * 12\n'
    '                    collective_input_bytes = 8 + maximum * 12\n'
    '                    compression_ratio = bytes_uncompressed / max(1, legacy_bytes)\n'
    '\n'
    '                elif do_communication and args.comm_mode == "gpu-quant-reducible":\n'
    '                    # Shared-scale summable int8 control: one global scale chosen\n'
    '                    # so the exact int8 sum across ranks cannot overflow. The\n'
    '                    # collective is all-reduce at 1-byte wire width; the price is\n'
    '                    # floor(127/world_size) quantization levels per side.\n'
    '                    levels = float(max(1, 127 // runtime.world_size))\n'
    '\n'
    '                    def shared_scale() -> torch.Tensor:\n'
    '                        absolute_maximum = flat.detach().abs().max().to(torch.float32)\n'
    '                        dist.all_reduce(absolute_maximum, op=dist.ReduceOp.MAX)\n'
    '                        return absolute_maximum\n'
    '\n'
    '                    absolute_maximum, scale_sync_s = measure_op(shared_scale)\n'
    '                    scale_tensor = (absolute_maximum / levels).clamp_min(1e-30)\n'
    '\n'
    '                    def encode_shared() -> torch.Tensor:\n'
    '                        quantized = torch.clamp(\n'
    '                            torch.round(flat / scale_tensor), min=-levels, max=levels\n'
    '                        )\n'
    '                        return quantized.to(torch.int8)\n'
    '\n'
    '                    encoded, serialize_s = measure_op(encode_shared)\n'
    '                    _, network_s = measure_op(lambda: dist.all_reduce(encoded))\n'
    '                    network_s = network_s + scale_sync_s\n'
    '\n'
    '                    def decode_shared() -> None:\n'
    '                        reconstructed = encoded.to(torch.float32)\n'
    '                        reconstructed.mul_(scale_tensor / runtime.world_size)\n'
    '                        unflatten_and_copy(model, reconstructed)\n'
    '\n'
    '                    _, deserialize_s = measure_op(decode_shared)\n'
    '                    legacy_bytes = encoded.numel() + 4\n'
    '                    collective_input_bytes = legacy_bytes\n'
    '                    compression_ratio = bytes_uncompressed / max(1, legacy_bytes)\n'
    '\n'
    '                elif do_communication and args.comm_mode == "gpu-topk-localef":\n'
    '                    corrected = flat if residual is None else flat + residual\n'
    '                    count = max(1, int(args.topk_frac * corrected.numel()))\n'
    '                    (indices, values), serialize_s = measure_op(\n'
    '                        lambda: gpu_topk(corrected, count)\n'
    '                    )\n'
    '                    (index_outputs, value_outputs, sizes, maximum), network_s = measure_op(\n'
    '                        lambda: _collect_topk(indices, values, runtime)\n'
    '                    )\n'
    '                    reconstructed, deserialize_s = measure_op(\n'
    '                        lambda: _reconstruct_topk(\n'
    '                            index_outputs, value_outputs, sizes, flat, runtime.world_size\n'
    '                        )\n'
    '                    )\n'
    '                    _, copy_s = measure_op(lambda: unflatten_and_copy(model, reconstructed))\n'
    '                    deserialize_s += copy_s\n'
    '                    # Standard local error feedback: the residual keeps exactly\n'
    '                    # the mass this rank did not transmit.\n'
    '                    next_residual = corrected.detach().clone()\n'
    '                    next_residual[indices] = 0.0\n'
    '                    residual = next_residual\n'
    '                    legacy_bytes = indices.numel() * 12\n'
    '                    collective_input_bytes = 8 + maximum * 12\n'
    '                    compression_ratio = bytes_uncompressed / max(1, legacy_bytes)\n'
    '\n'
    '            _, optimizer_step_s')

# --- step end: optimizer, wall sync, phase nulling ---------------------------
rep('            _, optimizer_step_s = measure(runtime, optimizer.step)\n'
    '            observed_step_s = time.perf_counter() - step_started\n'
    '            if backward_plus_comm_s is not None:',
    '            _, optimizer_step_s = measure_op(optimizer.step)\n'
    '            if wall_mode:\n'
    '                runtime.synchronize()\n'
    '            observed_step_s = time.perf_counter() - step_started\n'
    '            if wall_mode:\n'
    '                loss_value = float(loss.detach().cpu().item())\n'
    '                forward_s = None\n'
    '                barrier_wait_s = None\n'
    '                backward_compute_s = None\n'
    '                backward_plus_comm_s = None\n'
    '                serialize_s = None\n'
    '                network_s = None\n'
    '                network_s_is_estimate = False\n'
    '                deserialize_s = None\n'
    '                optimizer_step_s = None\n'
    '                phase_sum_s = None\n'
    '                unattributed_s = None\n'
    '            elif backward_plus_comm_s is not None:')

rep('            unattributed_s = max(0.0, observed_step_s - phase_sum_s)\n',
    '            if not wall_mode:\n'
    '                unattributed_s = max(0.0, observed_step_s - phase_sum_s)\n')

# --- record and provenance additions -----------------------------------------
rep('                "mode": args.comm_mode,\n'
    '                "comm_mode": args.comm_mode,\n'
    '                "sync_every": args.sync_every,',
    '                "mode": args.comm_mode,\n'
    '                "comm_mode": args.comm_mode,\n'
    '                "timing_mode": args.timing_mode,\n'
    '                "sync_every": args.sync_every,')

rep('                "topk_residual_semantics": LEGACY_TOPK_RESIDUAL if args.comm_mode == "gpu-topk" else None,',
    '                "topk_residual_semantics": (\n'
    '                    LEGACY_TOPK_RESIDUAL\n'
    '                    if args.comm_mode == "gpu-topk"\n'
    '                    else LOCALEF_TOPK_RESIDUAL\n'
    '                    if args.comm_mode == "gpu-topk-localef"\n'
    '                    else None\n'
    '                ),')

rep('                "comm_mode": args.comm_mode,\n'
    '                "sync_every": args.sync_every,\n'
    '                "batch_size_per_rank": args.batch_size,',
    '                "comm_mode": args.comm_mode,\n'
    '                "sync_every": args.sync_every,\n'
    '                "timing_mode": args.timing_mode,\n'
    '                "batch_size_per_rank": args.batch_size,')

rep('                "topk_residual_semantics": (\n'
    '                    LEGACY_TOPK_RESIDUAL if args.comm_mode == "gpu-topk" else None\n'
    '                ),',
    '                "topk_residual_semantics": (\n'
    '                    LEGACY_TOPK_RESIDUAL\n'
    '                    if args.comm_mode == "gpu-topk"\n'
    '                    else LOCALEF_TOPK_RESIDUAL\n'
    '                    if args.comm_mode == "gpu-topk-localef"\n'
    '                    else None\n'
    '                ),')

rep('"""Portable, fail-closed reproduction of the paper\'s legacy workloads.',
    '"""Control-campaign variant of the frozen July harness (2026-09-18).\n'
    '\n'
    'Generated by make_control_harness.py; the archived\n'
    'paper_workload_benchmark.py is unchanged. Additions: gpu-quant-reducible\n'
    '(shared-scale summable int8 over all-reduce), gpu-topk-localef (standard\n'
    'local error feedback), and --timing-mode wall (uninstrumented step\n'
    'timing). Phase-mode legacy behavior is byte-identical.\n'
    '\n'
    'Original header follows.')

io.open(DST, "w", encoding="utf-8", newline="\n").write(s)
py_compile.compile(DST, doraise=True)
print("control harness written:", DST)
print("edit groups applied:", n[0])
