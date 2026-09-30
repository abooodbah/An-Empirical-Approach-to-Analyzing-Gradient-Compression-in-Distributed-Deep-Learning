# -*- coding: utf-8 -*-
"""Runs the feature extractor exactly as hashed in RULES_FROZEN.sha256
(git object 35cf9be:diagnostic/features.py) on the round-4 records, in a
scratch directory, and counts its rows by timing mode. RETROSPECTIVE."""
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def run(workdir):
    src = subprocess.run(["git", "-C", REPO, "show", "35cf9be:diagnostic/features.py"],
                         check=True, capture_output=True).stdout
    tmp = tempfile.mkdtemp(prefix="frozen_extractor_")
    path = os.path.join(tmp, "features.py")
    with open(path, "wb") as fh:
        fh.write(src)
    subprocess.run([sys.executable, path, workdir], check=True, capture_output=True)
    rows = json.load(io.open(os.path.join(tmp, "features_dev.json"), encoding="utf-8"))
    return {"sha256": hashlib.sha256(src).hexdigest(), "rows": len(rows),
            "phase_rows": len(rows),   # the frozen version reads phase rows only
            "wall_rows": sum(1 for r in rows if r.get("timing") == "wall")}


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1] if len(sys.argv) > 1 else "C:/r4w"), indent=1))
