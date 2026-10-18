# -*- coding: utf-8 -*-
"""Write diagnostic/E12_FREEZE.sha256 (diagnostic/E12_PROTOCOL.md section 9).
Run after the last edit and before staging; commit the result, then stage
every listed file on Lonestar6 (RED_ROOT for the files a job runs, prereg/
for the rest) and submit. The job scripts refuse to run if a staged file
differs. Lines are "<sha256>  <path from the repository root>", LF endings."""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ART, "diagnostic", "E12_FREEZE.sha256")
FILES = [
    "diagnostic/E12_PROTOCOL.md",
    "diagnostic/e12/score_e12.py",
    "diagnostic/e12/test_score_e12.py",
    "diagnostic/e12/qual_check.py",
    "diagnostic/e12/test_qual_check.py",
    "diagnostic/rules.py",
    "diagnostic/features.py",
    "diagnostic/score_e6.py",
    "diagnostic/v1_1/rescore.py",
    "diagnostic/e8/replay_reference.py",
    "validation_v2/harness/build_v55.py",
    "validation_v2/harness/control_workload_benchmark_v5_5.py",
    "validation_v2/CODE_v55.sha256",
    "validation_v2/harness/collective_probe.py",
    "harness/launch/e12_sets.sh",
    "harness/launch/make_ls6_e12.py",
    "harness/launch/test_e12_launch.py",
    "harness/launch/ls6_e12q.slurm",
    "harness/launch/ls6_e12a.slurm",
    "harness/launch/ls6_e12b.slurm",
]


def digest(rel):
    with open(os.path.join(ART, *rel.split("/")), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def lines():
    return ["%s  %s" % (digest(rel), rel) for rel in FILES]


def write():
    io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(lines()) + "\n")
    return OUT


if __name__ == "__main__":
    print("wrote", write())
