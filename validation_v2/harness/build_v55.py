# validation_v2/harness/build_v55.py
# -*- coding: utf-8 -*-
"""Build harness v5.5 (2026-10-18) from v5.4 for the E12 (A-plus) campaign.
Two additions; with default arguments a run is the v5.4 run.

1. --dtype {fp32,bf16} (default fp32). bf16 runs the forward pass and the
   loss under torch.autocast with bfloat16, the usual mixed-precision setup:
   parameters, gradients, DDP buckets and the optimizer stay in float32. Under
   bf16 the 16-bit hook (ddp-fp16-hook) sends bfloat16 instead of float16, as
   PyTorch's bf16_compress_hook does; RED_HOOK16_DTYPE still overrides. The
   condition id gains ".bf16", the deviations list gains
   "compute_dtype=bf16_autocast", and the provenance condition records
   compute_dtype.
2. --verify-fidelity for the two hook modes (ddp-fp16-hook, ddp-int8-hook),
   qualification profile only, as for the other modes. Before the hook runs
   on a bucket, the wrapper all-reduces a float32 copy of the bucket and
   divides by the world size (the dense average of the same local
   gradients); after the hook it accumulates the squared error, the two
   squared norms and the dot product. After backward the iteration's sums
   give one relative L2 error and one cosine over all buckets, appended to
   HOOK_STATE["fidelity"] and written to the v52 sidecar like the other
   modes' fidelity. Dense DDP (ddp-overlap) sends the dense average itself;
   its error is zero by definition and it still refuses --verify-fidelity.
Timing paths of every existing mode are unchanged."""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "control_workload_benchmark_v5_4.py")
DST = os.path.join(HERE, "control_workload_benchmark_v5_5.py")

HEADER_OLD = '"""v5.1 (2026-09-26)'
HEADER_NEW = ('"""v5.5 (2026-10-18): --dtype bf16 (autocast compute, bfloat16 16-bit hook) and --verify-fidelity '
              'for the two DDP hooks; see build_v55.py.\n'
              'v5.4 (2026-10-14): int8-hook tensors kept out of HOOK_STATE. v5.3 (2026-10-13): int8 hook\n'
              'synchronous on oneCCL only. v5.2 (2026-10-07): bucketed and stochastic modes, fidelity.\n'
              'v5.1 (2026-09-26)')

IMPORT_OLD = 'import argparse\nimport csv\n'
IMPORT_NEW = 'import argparse\nimport contextlib\nimport csv\n'

PARSER_OLD = '''                        help="Compare the exchanged gradient with the dense average (qualification only).")
'''
PARSER_NEW = '''                        help="Compare the exchanged gradient with the dense average (qualification only).")
    parser.add_argument("--dtype", choices=COMPUTE_DTYPES, default="fp32",
                        help="v5.5: bf16 runs forward and loss under bfloat16 autocast (parameters, "
                             "gradients and DDP buckets stay float32); the 16-bit hook then sends bfloat16.")
'''

CONST_OLD = 'HOOK_STATE: Dict[str, Any] = {}\n'
CONST_NEW = ('HOOK_STATE: Dict[str, Any] = {}\n'
             '# v5.5: compute precisions; fp32 is every earlier run.\n'
             'COMPUTE_DTYPES = ("fp32", "bf16")\n')

FIDELITY_CHECK_OLD = '''    if args.verify_fidelity and args.comm_mode in DDP_MODES:
        raise ConfigurationError("--verify-fidelity needs the gradient before the exchange; DDP modes hide it")
'''
FIDELITY_CHECK_NEW = '''    if args.verify_fidelity and args.comm_mode in DDP_MODES and args.comm_mode not in HOOK_MODES:
        # v5.5: the two hook modes measure it inside the hook (_with_fidelity).
        raise ConfigurationError("--verify-fidelity needs the gradient before the exchange; DDP modes hide it")
'''

DEVIATION_OLD = '''        deviations.append("cuda_sync_debug_mode_not_timing_evidence")
    return deviations
'''
DEVIATION_NEW = '''        deviations.append("cuda_sync_debug_mode_not_timing_evidence")
    if getattr(args, "dtype", "fp32") != "fp32":
        deviations.append(f"compute_dtype={args.dtype}_autocast")
    return deviations
'''

HELPERS_OLD = 'def _half_compress_hook(process_group, bucket):'
HELPERS_NEW = '''def compute_autocast(runtime: Any, args: argparse.Namespace) -> Any:
    """v5.5: the forward-pass context. fp32: none. bf16: torch.autocast with
    bfloat16 on the run's device type; parameters and gradients stay float32."""
    if getattr(args, "dtype", "fp32") == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=runtime.kind, dtype=torch.bfloat16)


def hook16_dtype(backend: str, compute_dtype: str = "fp32") -> str:
    """v5.5: the 16-bit hook's wire type. RED_HOOK16_DTYPE wins; otherwise
    bfloat16 under bf16 compute (PyTorch's bf16_compress_hook) or on oneCCL,
    which has no float16 ring all-reduce; float16 otherwise (v5.4)."""
    wanted = os.environ.get("RED_HOOK16_DTYPE")
    if wanted:
        return wanted
    return "bfloat16" if compute_dtype == "bf16" or backend == "ccl" else "float16"


def _with_fidelity(hook: Callable[..., Any], args: argparse.Namespace) -> Callable[..., Any]:
    """v5.5: with --verify-fidelity, wrap a DDP hook so that each bucket's
    exchanged average is compared with the dense float32 average of the same
    local gradients (an extra all-reduce before the hook; qualification runs
    only, never timing evidence). Without the flag the hook is returned as is."""
    if not getattr(args, "verify_fidelity", False):
        return hook

    def fidelity_hook(process_group, bucket):  # unannotated: DDP checks hook annotations
        group_to_use = process_group if process_group is not None else dist.group.WORLD
        reference = bucket.buffer().detach().clone().to(torch.float32)
        dist.all_reduce(reference, group=group_to_use)
        reference.div_(group_to_use.size())

        def record(fut):
            value = fut.value()
            out = value[0] if isinstance(value, list) else value
            exchanged = out.detach().to(torch.float32)
            HOOK_TENSORS.setdefault("fidelity_acc", []).append(torch.stack([
                (exchanged - reference).pow(2).sum(), reference.pow(2).sum(),
                exchanged.pow(2).sum(), (exchanged * reference).sum()]))
            return out

        return hook(process_group, bucket).then(record)

    return fidelity_hook


def finish_hook_fidelity(iteration: int) -> Optional[Mapping[str, Any]]:
    """v5.5: one fidelity entry per iteration from the buckets' sums."""
    acc = HOOK_TENSORS.pop("fidelity_acc", None)
    if not acc:
        return None
    err2, ref2, out2, dot = (float(v) for v in torch.stack(acc).sum(0).tolist())
    entry = {
        "iteration": iteration,
        "relative_l2_error": math.sqrt(max(0.0, err2)) / max(1e-30, math.sqrt(max(0.0, ref2))),
        "cosine": dot / max(1e-30, math.sqrt(max(0.0, ref2) * max(0.0, out2))),
        "buckets": len(acc),
        "reference": "v5.5_per_bucket_dense_float32_average",
    }
    HOOK_STATE.setdefault("fidelity", []).append(entry)
    return entry


def _half_compress_hook(process_group, bucket):'''

HOOK16_OLD = '''        wanted = os.environ.get("RED_HOOK16_DTYPE") or ("bfloat16" if runtime.backend == "ccl" else "float16")
'''
HOOK16_NEW = '''        wanted = hook16_dtype(runtime.backend, getattr(args, "dtype", "fp32"))
'''

REG16_OLD = 'register_comm_hook(dist.group.WORLD, _half_compress_hook)'
REG16_NEW = 'register_comm_hook(dist.group.WORLD, _with_fidelity(_half_compress_hook, args))'
REG8_OLD = 'register_comm_hook(dist.group.WORLD, _pertensor_int8_hook)'
REG8_NEW = 'register_comm_hook(dist.group.WORLD, _with_fidelity(_pertensor_int8_hook, args))'

WARMUP_OLD = '''            with ddp_model.no_sync():
                output = run_model(inputs)
                loss = loss_fn(output, labels)
                _, elapsed = measure(runtime, loss.backward)
        else:
            output = run_model(inputs)
            loss = loss_fn(output, labels)
            _, elapsed = measure(runtime, loss.backward)
'''
WARMUP_NEW = '''            with ddp_model.no_sync():
                with compute_autocast(runtime, args):
                    output = run_model(inputs)
                    loss = loss_fn(output, labels)
                _, elapsed = measure(runtime, loss.backward)
        else:
            with compute_autocast(runtime, args):
                output = run_model(inputs)
                loss = loss_fn(output, labels)
            _, elapsed = measure(runtime, loss.backward)
'''

FORWARD_OLD = '''            def forward() -> Tuple[torch.Tensor, torch.Tensor]:
                output = run_model(inputs)
                return output, loss_fn(output, labels)
'''
FORWARD_NEW = '''            def forward() -> Tuple[torch.Tensor, torch.Tensor]:
                with compute_autocast(runtime, args):
                    output = run_model(inputs)
                    return output, loss_fn(output, labels)
'''

BACKWARD_OLD = '''                _, backward_plus_comm_s = measure_op(loss.backward)
'''
BACKWARD_NEW = '''                _, backward_plus_comm_s = measure_op(loss.backward)
                if args.verify_fidelity:
                    finish_hook_fidelity(iteration)
'''

CLEAR_OLD = '''        flat_reference: Optional[torch.Tensor] = None
'''
CLEAR_NEW = '''        flat_reference: Optional[torch.Tensor] = None
        HOOK_TENSORS.pop("fidelity_acc", None)  # v5.5: drop the warm-up buckets' sums
'''

CONDITION_OLD = '''            condition_id += f".seq{args.seq_len}"
'''
CONDITION_NEW = '''            condition_id += f".seq{args.seq_len}"
        if args.dtype != "fp32":
            condition_id += f".{args.dtype}"
'''

PROV_OLD = '''                "scaling_regime": "weak_scaling_fixed_local_batch",
'''
PROV_NEW = '''                "scaling_regime": "weak_scaling_fixed_local_batch",
                "compute_dtype": args.dtype,
'''

SUBS = [(HEADER_OLD, HEADER_NEW), (IMPORT_OLD, IMPORT_NEW), (CONST_OLD, CONST_NEW), (PARSER_OLD, PARSER_NEW),
        (FIDELITY_CHECK_OLD, FIDELITY_CHECK_NEW), (DEVIATION_OLD, DEVIATION_NEW), (HELPERS_OLD, HELPERS_NEW),
        (HOOK16_OLD, HOOK16_NEW), (REG16_OLD, REG16_NEW), (REG8_OLD, REG8_NEW), (WARMUP_OLD, WARMUP_NEW),
        (FORWARD_OLD, FORWARD_NEW), (BACKWARD_OLD, BACKWARD_NEW), (CLEAR_OLD, CLEAR_NEW),
        (CONDITION_OLD, CONDITION_NEW), (PROV_OLD, PROV_NEW)]


def build():
    text = io.open(SRC, encoding="utf-8").read()
    for index, (old, new) in enumerate(SUBS):
        if text.count(old) != 1:
            raise SystemExit("anchor %d occurs %d times in v5.4" % (index, text.count(old)))
        text = text.replace(old, new)
    io.open(DST, "w", encoding="utf-8", newline="\n").write(text)
    digest = hashlib.sha256(open(DST, "rb").read()).hexdigest()
    io.open(os.path.join(os.path.dirname(HERE), "CODE_v55.sha256"), "w", encoding="utf-8", newline="\n").write(
        "%s  harness/control_workload_benchmark_v5_5.py\n" % digest)
    return digest


if __name__ == "__main__":
    print(build())
