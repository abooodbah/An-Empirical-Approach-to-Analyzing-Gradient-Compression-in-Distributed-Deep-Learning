import csv
import glob
import json
import os

base = (r"C:\Users\Abdulfatah\AppData\Local\Temp\claude"
        r"\C--Users-Abdulfatah--codex-skills-multi-agent-coordinator-v2"
        r"\abb3b79e-3d8f-4d74-b7ee-4c31c833e0df\scratchpad\redctl_retrieved")


def op(p):
    return "\\\\?\\" + p


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
