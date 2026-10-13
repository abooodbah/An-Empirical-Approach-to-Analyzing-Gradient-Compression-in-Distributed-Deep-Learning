# validation_v2/harness/build_v53.py
# -*- coding: utf-8 -*-
"""Build harness v5.3 (2026-10-13) from v5.2. One change: the int8 hook's
synchronous form is chosen on oneCCL only. In v5.2 the test
`"ccl" in backend` also matched "nccl", so on NVIDIA systems the hook ran
synchronously although its asynchronous form works there. RED_HOOK_SYNC=1
still forces the synchronous form, so both can run in one job."""
import hashlib
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "control_workload_benchmark_v5_2.py")
DST = os.path.join(HERE, "control_workload_benchmark_v5_3.py")

SELECT_OLD = '''        HOOK_STATE["mode"] = "sync" if "ccl" in backend or os.environ.get("RED_HOOK_SYNC") == "1" else "async"
'''
SELECT_NEW = '''        HOOK_STATE["mode"] = int8_hook_mode(backend)
'''

DEF_OLD = '''def _pertensor_int8_hook(process_group, bucket):'''
DEF_NEW = '''def int8_hook_mode(backend: str) -> str:
    """v5.3: the synchronous form exists for oneCCL, whose nested future
    callbacks deadlock. Exact backend names, so that "nccl" is not taken
    for oneCCL. RED_HOOK_SYNC=1 forces the synchronous form anywhere."""
    if os.environ.get("RED_HOOK_SYNC") == "1":
        return "sync"
    return "sync" if str(backend).lower() in ("ccl", "xccl") else "async"


def _pertensor_int8_hook(process_group, bucket):'''

SUBS = [(SELECT_OLD, SELECT_NEW), (DEF_OLD, DEF_NEW)]


def build():
    text = io.open(SRC, encoding="utf-8").read()
    for index, (old, new) in enumerate(SUBS):
        if text.count(old) != 1:
            raise SystemExit("anchor %d occurs %d times in v5.2" % (index, text.count(old)))
        text = text.replace(old, new)
    io.open(DST, "w", encoding="utf-8", newline="\n").write(text)
    digest = hashlib.sha256(open(DST, "rb").read()).hexdigest()
    io.open(os.path.join(os.path.dirname(HERE), "CODE_v53.sha256"), "w", encoding="utf-8", newline="\n").write(
        "%s  harness/control_workload_benchmark_v5_3.py\n" % digest)
    return digest


if __name__ == "__main__":
    print(build())
