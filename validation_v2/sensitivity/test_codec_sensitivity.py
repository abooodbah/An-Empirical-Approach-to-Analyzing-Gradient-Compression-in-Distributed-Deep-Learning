# -*- coding: utf-8 -*-
import csv
import io
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import codec_sensitivity as C  # noqa: E402

HARNESS = os.path.join(os.path.dirname(HERE), "harness", "control_workload_benchmark_v5_1.py")


def test_shared_encoder_is_verbatim():
    src = io.open(HARNESS, encoding="utf-8").read()
    for line in C.VERBATIM_LINES:
        assert line in src, line


def test_input_properties():
    z = C.make_input("zero_tied", 100000, 1, "cpu")
    assert 0.88 < float((z == 0).float().mean()) < 0.92
    o = C.make_input("outlier", 100000, 1, "cpu")
    assert float(o.abs().max()) / float(o.abs().median()) > 1e3
    h = C.make_input("heavy", 100000, 1, "cpu")
    g = C.make_input("gaussian", 100000, 1, "cpu")
    kurt = lambda x: float(((x - x.mean()) ** 4).mean() / x.var() ** 2)
    assert kurt(h) > 5 * kurt(g)


def test_small_run_writes_every_cell(tmp_path):
    out = str(tmp_path / "s.csv")
    C.run(HARNESS, out, "cpu", sizes={"tiny": 10000}, seeds=(1,), blocks=2)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == len(C.CODECS) * len(C.DISTS) * 1 * 2
    assert all(float(r["per_call_s"]) > 0 for r in rows)


def test_margin_verdict(tmp_path):
    path = str(tmp_path / "a.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["codec", "workload", "n", "dist", "seed", "block", "calls", "per_call_s", "device", "threads"])
        for b in range(10):
            w.writerow(["x", "tiny", 10, "gaussian", 1, b, 5, 1.00, "cpu", 1])
            w.writerow(["x", "tiny", 10, "heavy", 1, b, 5, 1.10, "cpu", 1])
            w.writerow(["x", "tiny", 10, "zero_tied", 1, b, 5, 1.00, "cpu", 1])
            w.writerow(["x", "tiny", 10, "outlier", 1, b, 5, 1.00, "cpu", 1])
    verdicts = C.verdicts(path)
    assert verdicts[("x", "tiny")] is False
