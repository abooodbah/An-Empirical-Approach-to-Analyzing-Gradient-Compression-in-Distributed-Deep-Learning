# environment/platform_facts.py
# -*- coding: utf-8 -*-
"""Collects the hardware and software facts that the archived per-rank
provenance records carry, per system, for the platform table. It reports
what is recorded and lists what is not."""
import glob
import io
import json
import os
import sys

SECTIONS = ("libraries", "platform", "accelerator", "host_layout", "collective_preflight",
            "allowlisted_environment")
ENV_PREFIXES = ("NCCL_", "CCL_", "FI_", "I_MPI_", "OMP_", "TORCH_", "ZE_", "CUDA_", "SLURM_JOB_PARTITION")
NOT_RECORDED = ("GPU driver version", "GPU and adapter firmware", "PCIe and NUMA placement of GPUs and adapters",
                "CPU model and core count where the platform block lacks them")


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for key, inner in value.items():
            flatten("%s.%s" % (prefix, key) if prefix else str(key), inner, out)
    elif not isinstance(value, list):
        out[prefix] = str(value)


def collect(paths):
    facts = {}
    for path in paths:
        record = json.load(io.open(path, encoding="utf-8"))
        flat = {}
        for section in SECTIONS:
            if section in record:
                flatten(section, record[section], flat)
        for key, value in flat.items():
            if key.startswith("allowlisted_environment.") and not key.split(".", 1)[1].startswith(ENV_PREFIXES):
                continue
            facts.setdefault(key, set()).add(value)
    return {key: sorted(values) for key, values in sorted(facts.items())}


def render(named_dirs):
    lines = ["# Platform facts from the archived run provenance", "",
             "Collected by `environment/platform_facts.py`. A key with several values varied",
             "between runs. Hostnames and per-rank identifiers are left out.", ""]
    for name, directory in named_dirs:
        paths = glob.glob(os.path.join(directory, "**", "*rank0000.provenance.json"), recursive=True)
        facts = collect(paths)
        lines += ["## %s (%d records)" % (name, len(paths)), "",
                  "Read from the run directory `%s`." % os.path.basename(os.path.normpath(directory)), "",
                  "| key | values |", "|---|---|"]
        for key, values in facts.items():
            if "hostname" in key.lower() or len(values) > 6:
                continue
            lines.append("| `%s` | %s |" % (key, "; ".join(values)))
        lines.append("")
    lines += ["## Not recorded before October 2026", ""] + ["- %s" % item for item in NOT_RECORDED]
    lines += ["", "The Lonestar6 jobs of Revision 4 (E9) record these in `provenance/node_facts.*.txt`;",
              "`environment/node_facts.py` counts each value over their node records in",
              "`environment/E9_NODE_FACTS.json`.",
              "The Stampede3 jobs (E10) do not, so they stay unrecorded for the Intel system.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    pairs = [arg.split("=", 1) for arg in sys.argv[1:]]
    text = render(pairs)
    here = os.path.dirname(os.path.abspath(__file__))
    io.open(os.path.join(here, "PLATFORMS.md"), "w", encoding="utf-8", newline="\n").write(text)
    print(text[:3000])
