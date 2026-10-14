# validation_v2/harness/build_v54.py
# -*- coding: utf-8 -*-
"""Build harness v5.4 (2026-10-14) from v5.3. One change: the int8 hook's
asynchronous path keeps the gathered scales and zero points in HOOK_TENSORS,
a module dictionary that the provenance record does not copy. In v5.3 they
were stored in HOOK_STATE, and writing the provenance failed on the tensors
after every asynchronous run (E11 job 3508877). Timing is unchanged."""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "control_workload_benchmark_v5_3.py")
DST = os.path.join(HERE, "control_workload_benchmark_v5_4.py")

SUBS = [
    ('HOOK_STATE: Dict[str, Any] = {}\n',
     'HOOK_STATE: Dict[str, Any] = {}\n'
     '# v5.4: tensors the int8 hook passes between its callbacks. Kept apart\n'
     '# from HOOK_STATE, which the provenance record copies as JSON.\n'
     'HOOK_TENSORS: Dict[str, Any] = {}\n'),
    ('        HOOK_STATE["s_and_z"] = gathered_s_and_z\n',
     '        HOOK_TENSORS["s_and_z"] = gathered_s_and_z\n'),
    ('        gathered_s_and_z = HOOK_STATE["s_and_z"]\n',
     '        gathered_s_and_z = HOOK_TENSORS["s_and_z"]\n'),
]


def build():
    text = io.open(SRC, encoding="utf-8").read()
    for index, (old, new) in enumerate(SUBS):
        if text.count(old) != 1:
            raise SystemExit("anchor %d occurs %d times in v5.3" % (index, text.count(old)))
        text = text.replace(old, new)
    io.open(DST, "w", encoding="utf-8", newline="\n").write(text)
    digest = hashlib.sha256(open(DST, "rb").read()).hexdigest()
    io.open(os.path.join(os.path.dirname(HERE), "CODE_v54.sha256"), "w", encoding="utf-8", newline="\n").write(
        "%s  harness/control_workload_benchmark_v5_4.py\n" % digest)
    return digest


if __name__ == "__main__":
    print(build())
