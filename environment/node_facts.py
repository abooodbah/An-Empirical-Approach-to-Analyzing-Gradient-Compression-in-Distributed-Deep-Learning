# environment/node_facts.py
# -*- coding: utf-8 -*-
"""Node facts of the E9 jobs.

The Lonestar6 launch script of Revision 4 writes provenance/node_facts.N.txt
on every node of a job: nvidia-smi's GPU list and topology matrix, lscpu,
numactl, the adapter's firmware and NUMA node, lspci and the kernel. This
script reads those records from extracted run directories and writes, for
each fact, every distinct value with the number of node records that carry
it. The platform table of the paper takes its Lonestar6 cells from the
output. A fact a record does not hold is absent from that record's count,
never guessed.

usage: python node_facts.py <run_dir> [<run_dir> ...] --json OUT.json
"""
import glob
import io
import json
import os
import re
import sys

ANSI = re.compile("\x1b\\[[0-9;]*m")


def sections(text):
    out, name = {}, "host"
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("== "):
            name = line[3:].strip()
            continue
        out.setdefault(name, []).append(line)
    return out


def distinct(values):
    """One value, or the distinct values of a node in a fixed order."""
    return "; ".join(sorted(set(values)))


def parse(text):
    s, out = sections(text), {}
    rows = [[c.strip() for c in line.split(",")] for line in s.get("gpu", [])[1:] if line.strip()]
    rows = [r for r in rows if len(r) >= 5]
    if rows:
        out["gpu"] = distinct("%s, %s" % (r[1], r[2]) for r in rows)
        out["gpu_driver"] = distinct(r[3] for r in rows)
        out["gpu_vbios"] = distinct(r[4] for r in rows)
    cpu = dict((k.strip(), v.strip()) for k, v in (line.split(":", 1) for line in s.get("cpu", []) if ":" in line))
    if "Model name" in cpu:
        out["cpu"] = "%s x %s" % (cpu.get("Socket(s)", "?"), cpu["Model name"])
    table = [ANSI.sub("", line).split("\t") for line in s.get("topology", []) if "\t" in line]
    header = [c.strip() for c in table[0]] if table else []
    gpus = [row for row in table[1:] if row[0].strip().startswith("GPU")]
    if gpus and "NUMA Affinity" in header:
        column = header.index("NUMA Affinity")
        out["gpu_numa_nodes"] = ", ".join(row[column].strip() for row in gpus if len(row) > column)
    nics = [c for c in header if c.startswith("NIC")]
    if gpus and nics:
        column = header.index(nics[0])
        out["gpu_to_adapter"] = ", ".join(row[column].strip() for row in gpus if len(row) > column)
    for line in s.get("adapter", []):
        if "fw_ver:" in line:
            out["adapter_firmware"] = line.split("fw_ver:")[1].strip()
        if "numa_node=" in line:
            out["adapter_numa_node"] = line.split("numa_node=")[1].strip()
    cards = [line.split(": ", 1)[1].strip() for line in s.get("pci", [])
             if "Mellanox" in line and ("Infiniband controller" in line or "Ethernet controller" in line)]
    if cards:
        out["adapter"] = distinct(re.sub(r"\s*\(rev [0-9a-f]+\)$", "", c) for c in cards)
    kernel = [line.strip() for line in s.get("kernel", []) if line.strip()]
    if kernel:
        out["kernel"] = kernel[0]
    return out


def summary(run_dirs):
    facts, runs, records = {}, {}, 0
    for run_dir in run_dirs:
        name = os.path.basename(os.path.normpath(run_dir))
        paths = sorted(glob.glob(os.path.join(run_dir, "provenance", "node_facts.*.txt")))
        runs[name] = len(paths)
        for path in paths:
            records += 1
            parsed = parse(io.open(path, encoding="utf-8", errors="replace").read())
            for key, value in parsed.items():
                facts.setdefault(key, {})
                facts[key][value] = facts[key].get(value, 0) + 1
    return {"records": records, "runs": runs, "facts": facts}


def main(argv):
    out = argv[argv.index("--json") + 1] if "--json" in argv else None
    dirs = [a for a in argv if not a.startswith("--") and a != out]
    result = summary(dirs)
    if out:
        json.dump(result, io.open(out, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print("%d node records in %d run directories" % (result["records"], len(result["runs"])))
    for key in sorted(result["facts"]):
        for value, count in sorted(result["facts"][key].items(), key=lambda kv: -kv[1]):
            print("  %-18s %2d  %s" % (key, count, value))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
