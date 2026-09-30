# -*- coding: utf-8 -*-
"""Follow-up analysis including the R1 ResNet-50 rerun (LS6 job 3471881,
harness v5.1). analyze_v2.py stays frozen; this wrapper makes exactly two
documented adjustments before calling its main():

1. Validity gate on R1 records only. ANALYSIS_RULES_v2.md excludes runs
   with "parameter hashes differing". The v5 harness approximated the
   parameter hash with a full-state-dict hash, which also covers
   BatchNorm running buffers; those buffers legitimately differ across
   ranks under per-rank inputs, which excluded every ResNet-50 run in
   V1-V3. Harness v5.1 (R1) records the parameters-only hash the rule
   names. For records carrying that field (only R1 produces it), the
   gate reads final_param_hashes_equal; the full-state value is kept in
   final_full_state_hashes_equal. Raw result files are not modified.
2. VJOBS gains "R1" so the R1 rows enter the experiment tables. The
   pre-registered three-allocation verdicts still require V1-V3; ResNet
   verdict lines therefore read "incomplete (1 of 3 jobs)" and the R1
   rows are reported as single-allocation results.

usage: python analyze_r1.py <runs dir> <out dir> [--figures]
"""
import importlib.util
import json as _json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "analyze_v2", os.path.join(HERE, "analyze_v2.py"))
av = importlib.util.module_from_spec(spec)
spec.loader.exec_module(av)


class _JsonShim(object):
    load = staticmethod(_json.load)
    dump = staticmethod(_json.dump)
    dumps = staticmethod(_json.dumps)

    @staticmethod
    def loads(s):
        rec = _json.loads(s)
        if isinstance(rec, dict) and "final_param_hashes_equal" in rec:
            rec["final_full_state_hashes_equal"] = rec.get(
                "final_hashes_equal")
            rec["final_hashes_equal"] = rec["final_param_hashes_equal"]
        return rec


av.json = _JsonShim
av.VJOBS = ("V1", "V2", "V3", "R1")

if __name__ == "__main__":
    sys.exit(av.main(sys.argv[1:]))
