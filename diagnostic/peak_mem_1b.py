"""Peak device memory of the 1B conditions, from the rank-0 records.

Usage: python diagnostic/peak_mem_1b.py [<folder>]
<folder> holds the extracted archives raw/control/redctl-V-v3-3517750-51-52-53.tgz,
raw/control/redctl-V-reruns-3517832-3517833.tgz and raw/control/redctl-V-r16b-3517873.tgz
(default: $REV4_MEM_ROOT, then work/mem in the repository)."""
import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("REV4_MEM_ROOT", os.path.join(ROOT, "work", "mem"))


def op(p):
    # long-path prefix, Windows only
    return "\\\\?\\" + os.path.abspath(p) if os.name == "nt" else p


for pat, label in (("redctl-V-1n-3517751", "ws8"), ("redctl-V-2n-3517833", "ws16"),
                   ("redctl-V-2n-3517873", "ws16"), ("redctl-V-4n-3517753", "ws32")):
    for f in sorted(glob.glob(os.path.join(base, pat, "results", "*llama1b*rank0000.csv"))):
        rows = list(csv.DictReader(open(op(f), newline="")))
        cond = os.path.basename(f).split("__")[1].replace("llama1b.", "").replace(".tau1.bs2", "")
        pk = [float(r["peak_memory_mb"]) for r in rows if r.get("peak_memory_mb")]
        print(f"{label:5s} {cond:36s} peak={max(pk)/1024:6.1f} GB" if pk else f"{label} {cond} n/a")
f = glob.glob(os.path.join(base, "redctl-V-1n-3517751", "results", "*llama1b.native*rank0000.provenance.json"))[0]
d = json.load(open(op(f)))
print("device:", d["accelerator"]["device_name"])
