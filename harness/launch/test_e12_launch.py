# harness/launch/test_e12_launch.py
# -*- coding: utf-8 -*-
"""The E12 job scripts (ls6_e12q/a/b.slurm with e12_sets.sh) emit only
commands that harness v5.5 accepts, named as the frozen E12 scorer reads
them, in the order diagnostic/E12_PROTOCOL.md section 3 fixes, within the
section 8 budget; and the freeze check refuses a staged file that differs."""
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for folder in ("validation_v2/harness", "diagnostic/e12", "harness/launch"):
    path = os.path.join(ROOT, *folder.split("/"))
    if path not in sys.path:
        sys.path.insert(0, path)

import control_workload_benchmark_v5_5 as H  # noqa: E402
import score_e12 as S  # noqa: E402
import su_ledger as L  # noqa: E402

BASH = os.environ.get("GIT_BASH") or ("C:/PROGRA~1/Git/bin/bash.exe"
                                      if os.path.exists("C:/PROGRA~1/Git/bin/bash.exe") else "bash")
SCRIPTS = {"q": "ls6_e12q.slurm", "e12a": "ls6_e12a.slurm", "e12b": "ls6_e12b.slurm"}


def text(name):
    return io.open(os.path.join(HERE, name), encoding="utf-8").read()


def between(name, first, last):
    lines = text(name).split("\n")
    a = next(i for i, l in enumerate(lines) if l.startswith(first))
    b = next(i for i, l in enumerate(lines) if l.startswith(last) and i > a)
    return "\n".join(lines[a:b])


def commands(red_set):
    body = between(SCRIPTS[red_set], "common=(", "# shellcheck source=/dev/null")
    script = "\n".join([
        "set -eu",
        "RED_SET=%s; RED_ROOT=/red; BENCH=BENCH; COLLECTIVE_PROBE=PROBE; RUN_DIR=/run" % red_set,
        "SEED=1; CAMPAIGN_ID=c; WORLD=6; NODES=2",
        "NCCL_NET_GDR_LEVEL=SYS; NCCL_DEBUG_SUBSYS=INIT,NET,ENV",
        "srun() { :; }",
        "run_script() { local c=\"$1\" s=\"$2\"; shift 2; printf 'CMD\\t%s\\t%s\\t%s' \"$c\" \"$s\" "
        "\"${RED_HOOK_SYNC:-}/${RED_HOOK16_DTYPE:-}\"; printf '\\t%s' \"$@\"; printf '\\n'; }",
        body,
        'source "%s"' % os.path.join(HERE, "e12_sets.sh").replace("\\", "/"),
    ])
    done = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return [line.split("\t")[1:] for line in done.stdout.split("\n") if line.startswith("CMD\t")]


def value(args, flag):
    return args[args.index(flag) + 1]


def parsed(args):
    try:
        return H.parse_args(list(args))
    except SystemExit as exc:
        raise AssertionError("harness v5.5 rejects: %s" % " ".join(args)) from exc


KIND = (("block-", S.BLOCK, "validation"), ("phase-", S.PHASE, "extension"), ("trace-", S.TRACE, "qualification"),
        ("fid-", S.FID, "qualification"), ("equiv-", S.EQUIV, "qualification"))


def classify(cmds):
    """[(kind, model, tag, pass or None, parsed args, env)] for every harness command."""
    out = []
    for condition, script, env, *args in cmds:
        assert script == "BENCH", condition
        a = parsed(args)
        assert a.accelerator == "cuda" and a.backend == "nccl"
        if condition in ("preflight-transformer", "paper-resnet50-native"):
            out.append((condition, None, None, None, a, env))
            continue
        kind, pat, profile = next(k for k in KIND if condition.startswith(k[0]))
        m = pat.match(value(args, "--run-id"))
        assert m, value(args, "--run-id")
        tag = m.group("tag")
        assert tag in S.TAGS and m.group("model") == a.model and int(m.group("b")) == a.batch_size == S.BATCH
        assert int(m.group("ws")) == S.P and a.profile == profile
        mode, dtype = S.split_tag(tag)
        assert a.comm_mode == mode and a.dtype == dtype
        hook_sync, hook16 = env.split("/")
        assert hook_sync == ("1" if mode == "ddp-int8-hook" else "") and hook16 == ""
        out.append((kind.rstrip("-"), a.model, tag, m.groupdict().get("p"), a, env))
    return out


@pytest.mark.parametrize("red_set", ["e12a", "e12b"])
def test_scored_allocation_runs_every_input_before_every_outcome(red_set):
    runs = classify(commands(red_set))
    kinds = [r[0] for r in runs]
    assert kinds[:2] == ["preflight-transformer", "paper-resnet50-native"]
    assert kinds.count("equiv") == 4 and kinds.count("phase") == 20 and kinds.count("trace") == 20
    assert kinds.count("block") == 40 and "fid" not in kinds and len(runs) == 86
    first_block = kinds.index("block")
    assert all(k != "block" for k in kinds[:first_block]) and set(kinds[first_block:]) == {"block"}
    phases = {(m, t) for k, m, t, *_ in runs if k == "phase"}
    assert phases == {(m, t) for m in S.MODELS for t in S.ORIGINALS + ["gpu-quant-reducible"]}
    assert {(m, t) for k, m, t, *_ in runs if k == "trace"} == {(m, t) for m in S.MODELS for t in S.TAGS}
    assert {(m, t) for k, m, t, *_ in runs if k == "equiv"} == {
        (m, t) for m in S.MODELS for t in ("gpu-quant-gathersum", "gpu-quant-reducible")}
    first = [(m, t) for k, m, t, p, *_ in runs if k == "block" and p == "1"]
    second = [(m, t) for k, m, t, p, *_ in runs if k == "block" and p == "2"]
    assert sorted(first) == sorted((m, t) for m in S.MODELS for t in S.TAGS) and second == first[::-1]


def test_scored_inputs_have_the_e8_shapes():
    for kind, model, tag, p, a, env in classify(commands("e12a")):
        if kind == "phase":
            assert a.timing_mode == "phase" and a.iters == H.PAPER_MEASURED_ITERS and a.warmup_iters == 10
            assert a.calibration_reference == S.CALIBRATION_RUN
        elif kind == "trace":
            assert a.timing_mode == "wall" and a.iters == 8 and a.warmup_iters == 20 and a.input_pool == 4
            assert a.trace_rank_list == (0, 3) and a.trace_dir is not None
        elif kind == "block":
            assert a.timing_mode == "block" and a.iters == 100 and a.warmup_iters == 20 and a.input_pool == 4
            assert a.calibration_reference == S.CALIBRATION_RUN
        elif kind == "equiv":
            assert a.verify_equivalence
        elif kind == "paper-resnet50-native":
            assert value_of(a, "run_id") == S.CALIBRATION_RUN and not a.skip_calibration


def value_of(a, name):
    return getattr(a, name)


def test_the_two_scored_allocations_run_the_same_list():
    strip = lambda cmds: [c[:1] + c[2:] for c in cmds]  # noqa: E731
    assert strip(commands("e12a")) == strip(commands("e12b"))


def test_qualification_job_measures_the_error_of_every_compressed_tag():
    runs = classify(commands("q"))
    fid = {(m, t) for k, m, t, *_ in runs if k == "fid"}
    assert fid == {(m, t) for m in S.MODELS for t in S.TAGS if t not in S.DENSE_TAGS}
    assert all(a.verify_fidelity and a.iters == S.FIDELITY_ITERS for k, m, t, p, a, e in runs if k == "fid")
    # parse checks: the frozen replay on the BF16 int8 hook (case 12), the
    # reverse substitution on the BF16 16-bit hook (mirror M12)
    assert [(m, t) for k, m, t, *_ in runs if k == "trace"] == [
        (m, t) for m in S.MODELS for t in ("ddp-int8-hook-bf16", "ddp-fp16-hook-bf16")]
    assert [(m, t) for k, m, t, *_ in runs if k == "block"] == [("resnet50", "cpu-quant"), ("transformer", "ddp-overlap-bf16")]
    assert len(runs) == 24


def sbatch(name):
    t = text(name)
    get = lambda key: re.search(r"^#SBATCH --%s=(\S+)" % key, t, re.M).group(1)  # noqa: E731
    h, m, s = (int(x) for x in get("time").split(":"))
    return get("partition"), int(get("nodes")), 3600 * h + 60 * m + s


def test_budget_caps():
    caps = {}
    for red_set, name in SCRIPTS.items():
        partition, nodes, limit = sbatch(name)
        assert partition == "gpu-a100-dev" and nodes == 2
        caps[red_set] = L.bound(nodes, limit, 3)
    assert caps == pytest.approx({"q": 1.5, "e12a": 4.0, "e12b": 4.0})
    assert sum(caps.values()) == pytest.approx(9.5)


def test_scripts_are_generated_and_use_harness_v55():
    import make_ls6_e12 as G
    for red_set, name in SCRIPTS.items():
        before = text(name)
        G.make(red_set)
        assert text(name) == before, name + " differs from make_ls6_e12.py output"
        assert 'BENCH="$RED_ROOT/control_workload_benchmark_v5_5.py"' in before
        assert 'RED_SET="%s"' % red_set in before and "NCCL_NET_GDR_LEVEL=\"${RED_GDR_LEVEL:-SYS}\"" in before


def freeze_segment():
    return between("ls6_e12a.slurm", "FREEZE=", "RPN=")


def run_freeze(red_root, script_path):
    script = "set -Eeuo pipefail\nRED_ROOT=%s\n%s\necho FREEZE-OK" % (red_root.replace("\\", "/"), freeze_segment())
    return subprocess.run([BASH, "-c", script, script_path.replace("\\", "/")], capture_output=True, text=True)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_freeze_check_accepts_the_staged_files_and_refuses_a_change(tmp_path):
    root = tmp_path / "red"
    (root / "prereg").mkdir(parents=True)
    staged = {"control_workload_benchmark_v5_5.py": os.path.join(ROOT, "validation_v2", "harness",
                                                                  "control_workload_benchmark_v5_5.py"),
              "e12_sets.sh": os.path.join(HERE, "e12_sets.sh")}
    for base, src in staged.items():
        shutil.copyfile(src, str(root / base))
    shutil.copyfile(os.path.join(ROOT, "diagnostic", "e12", "score_e12.py"), str(root / "prereg" / "score_e12.py"))
    job = os.path.join(HERE, "ls6_e12a.slurm")
    lines = ["%s  validation_v2/harness/control_workload_benchmark_v5_5.py" % sha(staged["control_workload_benchmark_v5_5.py"]),
             "%s  harness/launch/e12_sets.sh" % sha(staged["e12_sets.sh"]),
             "%s  diagnostic/e12/score_e12.py" % sha(str(root / "prereg" / "score_e12.py")),
             "%s  harness/launch/ls6_e12a.slurm" % sha(job)]
    (root / "prereg" / "E12_FREEZE.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok = run_freeze(str(root), job)
    assert ok.returncode == 0 and "FREEZE-OK" in ok.stdout, ok.stderr
    other = run_freeze(str(root), os.path.join(HERE, "ls6_e11b.slurm"))           # not a frozen job script
    assert other.returncode == 1 and "not a frozen E12 script" in other.stderr
    with io.open(str(root / "e12_sets.sh"), "a", encoding="utf-8") as fh:
        fh.write("# edited after the freeze\n")
    bad = run_freeze(str(root), job)
    assert bad.returncode == 1 and "e12_sets.sh differs" in bad.stderr
    (root / "prereg" / "E12_FREEZE.sha256").unlink()
    assert run_freeze(str(root), job).returncode == 1
