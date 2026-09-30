# -*- coding: utf-8 -*-
"""Build the R1 rerun (2026-09-26): harness v5.1 adds a parameters-only
final hash beside the full-state hash (which includes BatchNorm buffers and
therefore differs across ranks for ResNet-50 under per-rank inputs);
manifest_R1/run_list_R1 are the 32 ResNet-50 conditions of V1; ls6_v2r.slurm
points at the v5.1 harness. Nothing in v5 is changed."""
import io
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- harness v5.1 --------------------------------------------
src = io.open(os.path.join(HERE, "control_workload_benchmark_v5.py"),
              encoding="utf-8").read()

subs = [
# new params-only hash next to model_sha256
("""def model_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        cpu = tensor.detach().cpu().contiguous()
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(np.asarray(cpu).tobytes(order="C"))
    return digest.hexdigest()""",
 """def model_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        cpu = tensor.detach().cpu().contiguous()
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(np.asarray(cpu).tobytes(order="C"))
    return digest.hexdigest()


def params_sha256(model: nn.Module) -> str:
    \"\"\"Hash the learned parameters only. The full state_dict hash above
    also covers buffers (BatchNorm running statistics), which legitimately
    differ across ranks under per-rank inputs; this hash isolates the
    quantity the validity rules name: the parameters.\"\"\"
    digest = hashlib.sha256()
    for name, tensor in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        digest.update(name.encode("utf-8"))
        cpu = tensor.detach().cpu().contiguous()
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(np.asarray(cpu).tobytes(order="C"))
    return digest.hexdigest()"""),
# record template gains the two fields
("""                "final_model_sha256": None,
                "final_hashes_equal": None,""",
 """                "final_model_sha256": None,
                "final_hashes_equal": None,
                "final_param_model_sha256": None,
                "final_param_hashes_equal": None,"""),
# compute and fill at the final checkpoint
("""        final_hash = model_sha256(model)
        final_hashes = gather_hashes(final_hash, runtime)
        final_hashes_equal = len(set(final_hashes)) == 1
        completed_validation = validate_completed_samples(records, args)
        for record in records:
            record["final_model_sha256"] = final_hash
            record["final_hashes_equal"] = final_hashes_equal""",
 """        final_hash = model_sha256(model)
        final_hashes = gather_hashes(final_hash, runtime)
        final_hashes_equal = len(set(final_hashes)) == 1
        final_param_hash = params_sha256(model)
        final_param_hashes = gather_hashes(final_param_hash, runtime)
        final_param_hashes_equal = len(set(final_param_hashes)) == 1
        completed_validation = validate_completed_samples(records, args)
        for record in records:
            record["final_model_sha256"] = final_hash
            record["final_hashes_equal"] = final_hashes_equal
            record["final_param_model_sha256"] = final_param_hash
            record["final_param_hashes_equal"] = final_param_hashes_equal"""),
# provenance block
("""            "parameter_hashes": {
                "initial": initial_hashes,
                "measured_start": measured_start_hashes,
                "final": final_hashes,
                "final_equal": final_hashes_equal,
            },""",
 """            "parameter_hashes": {
                "initial": initial_hashes,
                "measured_start": measured_start_hashes,
                "final": final_hashes,
                "final_equal": final_hashes_equal,
                "final_params": final_param_hashes,
                "final_params_equal": final_param_hashes_equal,
            },"""),
# field registry
("""    "final_model_sha256",
    "final_hashes_equal",""",
 """    "final_model_sha256",
    "final_hashes_equal",
    "final_param_model_sha256",
    "final_param_hashes_equal","""),
]
for a, b in subs:
    assert src.count(a) == 1, a[:70]
    src = src.replace(a, b)
src = src.replace(
    '"""', '"""v5.1 (2026-09-26): adds a parameters-only final hash '
    '(final_param_hashes_equal) beside the full-state hash; no other '
    'behavior changes.\n\n', 1)
io.open(os.path.join(HERE, "control_workload_benchmark_v5_1.py"), "w",
        encoding="utf-8", newline="\n").write(src)
print("harness v5.1 written")

# ---------------- manifest + run list -------------------------------------
m = json.load(open(os.path.join(HERE, "v2", "manifest_V1.json"),
                   encoding="utf-8"))
keep = [e for e in m["runs"] if "resnet50" in e["id"]]
assert len(keep) == 32, len(keep)
m2 = dict(m)
m2["job"] = "R1"
m2["runs"] = keep
m2["note"] = ("ResNet-50 rerun with harness v5.1: parameters-only final "
              "hash added; conditions and seed identical to V1.")
json.dump(m2, io.open(os.path.join(HERE, "v2", "manifest_R1.json"), "w",
                      encoding="utf-8", newline="\n"), indent=1)

rl = io.open(os.path.join(HERE, "v2", "run_list_V1.sh"),
             encoding="utf-8").read().splitlines()
out = ["# generated by build_r1.py: the 32 ResNet-50 conditions of V1, "
       "harness v5.1"]
kept = 0
for line in rl:
    if line.startswith("run_condition") and "resnet50" in line:
        out.append(line)
        kept += 1
assert kept == 32, kept
io.open(os.path.join(HERE, "v2", "run_list_R1.sh"), "w", encoding="utf-8",
        newline="\n").write("\n".join(out) + "\n")
print("manifest_R1 + run_list_R1 written (%d conditions)" % kept)

# ---------------- slurm wrapper -------------------------------------------
sl = io.open(os.path.join(HERE, "ls6_v2.slurm"), encoding="utf-8").read()
old = 'BENCHMARK="$RED_ROOT/control_workload_benchmark_v5.py"'
assert sl.count(old) == 1
sl = sl.replace(old, 'BENCHMARK="$RED_ROOT/control_workload_benchmark_v5_1.py"')
io.open(os.path.join(HERE, "ls6_v2r.slurm"), "w", encoding="utf-8",
        newline="\n").write(sl)
print("ls6_v2r.slurm written")
