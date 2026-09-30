# -*- coding: utf-8 -*-
"""Checks PROVENANCE.md against the repository: every 'sha256 <hex> <path>'
line must match the file, and the E6 freeze push must precede the first
scored held-out submission."""
import hashlib
import io
import json
import os
import re


def verify(root):
    problems = []
    text = io.open(os.path.join(root, "provenance", "PROVENANCE.md"), encoding="utf-8").read()
    for hexd, path in re.findall(r"sha256 ([0-9a-f]{64}) `([^`]+)`", text):
        full = os.path.join(root, path)
        if not os.path.exists(full):
            problems.append("missing " + path)
            continue
        with open(full, "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != hexd:
                problems.append("hash mismatch " + path)
    events = json.load(io.open(os.path.join(root, "provenance", "github_push_events.json"),
                               encoding="utf-8"))
    push = [e["created_at"] for e in events if e.get("type") == "PushEvent"
            and str(e.get("payload", {}).get("head", "")).startswith("35cf9be")]
    jobs = io.open(os.path.join(root, "provenance", "lonestar6_jobs_utc.tsv"),
                   encoding="utf-8").read().splitlines()
    submits = {l.split("|")[0]: l.split("|")[2] for l in jobs if "|" in l}
    if not push:
        problems.append("no push event for 35cf9be")
    elif not push[0].rstrip("Z") < submits.get("3459786", ""):
        problems.append("E6 freeze push does not precede the first scored job")
    return problems


if __name__ == "__main__":
    import sys
    p = verify(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("OK" if not p else "\n".join(p))
