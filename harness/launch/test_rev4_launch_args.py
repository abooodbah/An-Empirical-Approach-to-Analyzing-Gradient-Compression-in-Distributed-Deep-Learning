# harness/launch/test_rev4_launch_args.py
# -*- coding: utf-8 -*-
"""Every command the E9 and E10 launch scripts emit is accepted by the
harness's own argument parser and configuration check, and carries the run id
the frozen scorers look for. A flag the harness rejects would otherwise be
found only after the job has waited in the queue."""
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for folder in ("validation_v2/harness", "validation_v2/analysis"):
    path = os.path.join(ROOT, *folder.split("/"))
    if path not in sys.path:
        sys.path.insert(0, path)

import collective_probe as P  # noqa: E402
import control_workload_benchmark_v5_2 as H  # noqa: E402
import score_e9 as E9  # noqa: E402
import score_e10 as E10  # noqa: E402

BASH = os.environ.get("GIT_BASH") or ("C:/PROGRA~1/Git/bin/bash.exe"
                                      if os.path.exists("C:/PROGRA~1/Git/bin/bash.exe") else "bash")
E9_MODES = ("native", "fp16-native", "ddp-overlap", "ddp-fp16-hook", "ddp-int8-hook", "gpu-quant",
            "gpu-quant-reducible", "gpu-quant-sr-int8", "gpu-quant-sr-fp16")


def between(name, first, last):
    """The lines of a launch file from the one that starts with `first` up
    to, not including, the one that starts with `last`."""
    lines = open(os.path.join(HERE, name), encoding="utf-8").read().split("\n")
    a = next(i for i, l in enumerate(lines) if l.startswith(first))
    b = next(i for i, l in enumerate(lines) if l.startswith(last) and i > a)
    return "\n".join(lines[a:b])


def commands(script):
    done = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    out = []
    for line in done.stdout.split("\n"):
        if line.startswith("CMD\t"):
            out.append(line.split("\t")[1:])
    return out


def e9_commands(red_set, world):
    body = between("ls6_e9.slurm", "common=(", "# shellcheck source=/dev/null")
    script = "\n".join([
        "set -eu",
        "RED_SET=%s; RED_ROOT=/red; BENCH=BENCH; COLLECTIVE_PROBE=PROBE; RUN_DIR=/run" % red_set,
        "SEED=1; CAMPAIGN_ID=c; WORLD=%d; NODES=%d" % (world, world // 3),
        "NCCL_NET_GDR_LEVEL=SYS; NCCL_DEBUG_SUBSYS=INIT,NET,ENV",
        "srun() { :; }",
        "run_script() { local c=\"$1\" s=\"$2\"; shift 2; printf 'CMD\\t%s\\t%s\\t%s' \"$c\" \"$s\" "
        "\"${NCCL_ALGO:-default}/${NCCL_NET_GDR_LEVEL}\"; printf '\\t%s' \"$@\"; printf '\\n'; }",
        body,
        'source "%s"' % os.path.join(HERE, "e9_sets.sh").replace("\\", "/"),
    ])
    return commands(script)


def e10_commands(red_set, nodes):
    body = between("pvc_e10.slurm", "BUCKET=", "# shellcheck source=/dev/null")
    script = "\n".join([
        "set -eu",
        "RED_SET=%s; RED_ROOT=/red; BENCHMARK=BENCH; COLLECTIVE_PROBE=PROBE; RUN_DIR=/run" % red_set,
        "PAPER_SEED=1; CAMPAIGN_ID=c; NODES=%d; WORLD=%d; SCRIPT_DIR=/launch" % (nodes, nodes * 8),
        "run_n() { local n=\"$1\" c=\"$2\" l=\"$3\" s=\"$4\"; shift 4; printf 'CMD\\t%s\\t%s\\t%s' \"$c\" \"$s\" "
        "\"$n/$l\"; printf '\\t%s' \"$@\"; printf '\\n'; }",
        body,
        'source "%s"' % os.path.join(HERE, "e10_sets.sh").replace("\\", "/"),
    ])
    return commands(script)


def accepted_by_harness(args):
    try:
        return H.parse_args(list(args))
    except SystemExit as exc:  # argparse reports a rejected flag this way
        raise AssertionError("the harness rejects: %s" % " ".join(args)) from exc


class Reached(Exception):
    pass


def accepted_by_probe(args, monkeypatch):
    def stop(_):
        raise Reached()
    monkeypatch.setattr(P.H, "initialize_distributed", stop)
    try:
        P.main(list(args))
    except Reached:
        return
    except SystemExit as exc:
        raise AssertionError("the probe rejects: %s" % " ".join(args)) from exc
    raise AssertionError("the probe did not reach the backend")


def value(args, flag):
    return args[args.index(flag) + 1]


def test_the_parser_rejects_a_misspelled_flag():
    good = ["--campaign-id", "c", "--run-id", "r", "--output-dir", "o", "--model", "transformer",
            "--profile", "qualification", "--comm-mode", "gpu-quant-bucketed", "--exchange-bucket-elems", "8"]
    accepted_by_harness(good)
    with pytest.raises(AssertionError):
        accepted_by_harness(good[:-2] + ["--exchange-bucket-elements", "8"])
    with pytest.raises(AssertionError):
        accepted_by_harness(good[:-2])  # a bucketed mode without its bucket size


@pytest.mark.parametrize("red_set,world", [("q12a", 12), ("q12b", 12), ("q24a", 24), ("q24b", 24)])
def test_every_e9_command_is_accepted_and_named_for_the_scorer(red_set, world, monkeypatch):
    cmds = e9_commands(red_set, world)
    block = set()
    for condition, script, _env, *args in cmds:
        if script == "PROBE":
            accepted_by_probe(args, monkeypatch)
            continue
        parsed = accepted_by_harness(args)
        run_id = value(args, "--run-id")
        assert parsed.accelerator == "cuda" and parsed.backend == "nccl"
        if condition.startswith("block-"):
            m = E9.BLOCK.match(run_id)
            assert m and int(m.group(1)) == world and parsed.profile == "validation"
            assert parsed.timing_mode == "block" and m.group(3) == parsed.comm_mode
            block.add((m.group(2), m.group(3), m.group(4)))
        elif condition.startswith("phase-"):
            m = E9.PHASE.match(run_id)
            assert m and m.group(3) == parsed.comm_mode and parsed.profile == "extension"
            assert parsed.iters == H.PAPER_MEASURED_ITERS  # 40 measured iterations give status PASS
        elif condition.startswith("fid-"):
            m = E9.FID.match(run_id)
            assert m and m.group(3) == parsed.comm_mode and parsed.verify_fidelity
        elif condition.startswith("equiv-"):
            m = E9.EQUIV.match(run_id)
            assert m and m.group(3) == parsed.comm_mode and parsed.verify_equivalence
        else:
            assert condition in ("preflight-transformer", "paper-resnet50-native"), condition
    assert block == {(model, mode, n) for model in ("transformer", "resnet50") for mode in E9_MODES
                     for n in ("1", "2")}
    reference = [value(a, "--run-id") for c, s, _e, *a in cmds if c == "paper-resnet50-native"]
    used = {value(a, "--calibration-reference") for c, s, _e, *a in cmds if "--calibration-reference" in a}
    assert used == set(reference) and len(reference) == 1


def test_e9_probes_set_the_algorithm_and_the_transport_they_are_named_for():
    env = {c: e for c, s, e, *a in e9_commands("q12a", 12) if c.startswith("probe-")}
    assert env == {"probe-default": "default/SYS", "probe-ring": "Ring/SYS", "probe-tree": "Tree/SYS",
                   "probe-hs-default": "default/LOC", "probe-hs-ring": "Ring/LOC", "probe-hs-tree": "Tree/LOC"}
    after = [e for c, s, e, *a in e9_commands("q12a", 12) if c == "paper-resnet50-native"]
    assert after == ["default/SYS"]  # the probes leave the environment as they found it


@pytest.mark.parametrize("red_set,nodes", [("s1", 2), ("s2", 4)])
def test_every_e10_command_is_accepted_and_named_for_the_scorer(red_set, nodes, monkeypatch):
    seen = set()
    for condition, script, where, *args in e10_commands(red_set, nodes):
        if script == "PROBE":
            accepted_by_probe(args, monkeypatch)
            continue
        parsed = accepted_by_harness(args)
        run_id = value(args, "--run-id")
        step_nodes = int(where.split("/")[0])
        assert parsed.accelerator == "xpu" and parsed.backend == "ccl" and parsed.model in ("llama1b", "transformer")
        if condition.startswith("smoke-"):
            assert "smoke" in run_id and parsed.profile == "qualification"
            continue
        m = E10.RUN.match(run_id)
        assert m and int(m.group(1)) == step_nodes * 8 and m.group(2) == parsed.comm_mode
        assert condition == "llama-ws%d-%s" % (step_nodes * 8, parsed.comm_mode)  # first_error() reads this log name
        assert parsed.profile == "extension" and parsed.timing_mode == "phase" and parsed.progress_dir is not None
        if parsed.comm_mode in E10.WIRE:
            assert parsed.exchange_bucket_elems == E10.BUCKET
        else:
            assert parsed.exchange_bucket_elems == 0
        seen.add((step_nodes * 8, parsed.comm_mode))
    want = {"s1": {(16, "gpu-quant-bucketed"), (16, "gpu-quant-reducible-bucketed"), (16, "native-bucketed"),
                   (8, "gpu-quant"), (8, "gpu-quant-bucketed"), (16, "native")},
            "s2": {(32, m) for m in E10.WIRE}}[red_set]
    assert seen == want


def test_each_job_fits_the_cap_its_launcher_enforces():
    import su_ledger as L
    text = open(os.path.join(HERE, "pvc_e10.slurm"), encoding="utf-8").read()
    default_cap = float(re.search(r'RED_SU_CAP="\$\{RED_SU_CAP:-([0-9.]+)\}"', text).group(1))
    assert L.bound(2, 55 * 60, 3) <= default_cap + 1e-5       # s1 at the file's own node count and time limit
    assert L.bound(4, 50 * 60, 3) == pytest.approx(10.0)      # s2 needs RED_SU_CAP=10 at submission
