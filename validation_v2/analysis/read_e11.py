# validation_v2/analysis/read_e11.py
# -*- coding: utf-8 -*-
"""Reads an E11 run directory (diagnostic/E11_NOTE.md): block-timed step of
each condition and pass, the int8 hook's recorded mode, and the ratios the
note names. usage: read_e11.py <run dir> [out.json]"""
import glob
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev4_records as R  # noqa: E402

TAGS = ("ddp-overlap", "ddp-fp16-hook", "ddp-int8-hook-async", "ddp-int8-hook-sync")


def find_mode(obj):
    if isinstance(obj, dict):
        if "hook_state" in obj and isinstance(obj["hook_state"], dict):
            return obj["hook_state"].get("mode")
        for value in obj.values():
            found = find_mode(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_mode(value)
            if found:
                return found
    return None


def hook_modes(run_dir, run_id):
    modes = set()
    for path in glob.glob(os.path.join(run_dir, "results", run_id + "__*.provenance.json")):
        modes.add(find_mode(json.load(io.open(path, encoding="utf-8"))))
    return sorted(m for m in modes if m) or ["none recorded"]


def read(run_dir):
    runs = R.read_dir(run_dir)
    out = {"run_dir": os.path.basename(run_dir), "conditions": {}}
    for run_id, run in sorted(runs.items()):
        if "-block-p" not in run_id:
            continue
        head, pass_ = run_id.rsplit("-block-p", 1)
        model = "transformer" if "-transformer-" in head else "resnet50"
        tag = next(t for t in sorted(TAGS, key=len, reverse=True) if head.endswith(t))
        rows = run["rows"]
        entry = {"valid": R.valid(rows), "hook_modes": hook_modes(run_dir, run_id) if "int8" in tag else None}
        try:
            entry["step_ms"] = R.block_step_ms(rows)
        except (ValueError, KeyError) as exc:
            entry["step_ms"] = None
            entry["error"] = str(exc)
        out["conditions"].setdefault(model, {}).setdefault(tag, {})[pass_] = entry
    ratios = {}
    for model, by_tag in out["conditions"].items():
        for pass_ in ("1", "2"):
            step = {t: by_tag.get(t, {}).get(pass_, {}).get("step_ms") for t in TAGS}
            ddp = step["ddp-overlap"]
            r = {}
            for t in TAGS[1:]:
                if ddp and step[t]:
                    r[t + "/ddp"] = round(step[t] / ddp, 3)
            if step["ddp-int8-hook-async"] and step["ddp-int8-hook-sync"]:
                r["int8-async/int8-sync"] = round(step["ddp-int8-hook-async"] / step["ddp-int8-hook-sync"], 3)
            if step["ddp-int8-hook-async"] and step["ddp-fp16-hook"]:
                r["int8-async/fp16-hook"] = round(step["ddp-int8-hook-async"] / step["ddp-fp16-hook"], 3)
            ratios.setdefault(model, {})["pass" + pass_] = r
    out["ratios"] = ratios
    return out


if __name__ == "__main__":
    result = read(sys.argv[1])
    text = json.dumps(result, indent=1, sort_keys=True)
    if len(sys.argv) > 2:
        io.open(sys.argv[2], "w", encoding="utf-8", newline="\n").write(text + "\n")
    print(text)
