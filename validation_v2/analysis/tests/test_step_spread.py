# -*- coding: utf-8 -*-
import io
import os

import pytest

import step_spread as S

HEADER = "model,comm_mode,sync_every,rank,observed_step_s\n"
JULY = os.environ.get("REV4_JULY_ROOT", "C:/r5w/july")


def job(tmp_path, name, rows):
    d = tmp_path / name / "results"
    d.mkdir(parents=True)
    io.open(str(d / "a__rank0000.csv"), "w", encoding="utf-8", newline="\n").write(HEADER + rows)
    return str(tmp_path / name)


def test_spread_is_the_range_of_the_job_means_over_their_mean(tmp_path):
    a = job(tmp_path, "j1", "resnet50,native,1,0,0.010\nresnet50,native,1,0,0.030\n")       # mean 20 ms
    b = job(tmp_path, "j2", "resnet50,native,1,0,0.030\n")                                  # 30 ms
    c = job(tmp_path, "j3", "resnet50,native,1,0,0.040\n")                                  # 40 ms
    out = S.spread([a, b, c], [("resnet50", "native", 1)])
    row = out["conditions"]["resnet50|native|1"]
    assert row["job_means_ms"] == pytest.approx([20.0, 30.0, 40.0])
    assert row["mean_ms"] == pytest.approx(30.0)
    assert row["spread_pct"] == pytest.approx(100 * 20.0 / 30.0)
    assert out["largest"]["condition"] == "resnet50|native|1"


def test_a_condition_no_job_measured_is_an_error(tmp_path):
    a = job(tmp_path, "j1", "resnet50,native,1,0,0.010\n")
    with pytest.raises(KeyError):
        S.spread([a], [("transformer", "native", 1)])


def test_the_phase_figure_of_the_h100_system():
    # The ten bars of the phase-attribution figure, over the three July jobs.
    dirs = [os.path.join(JULY, "h100-%d" % j) for j in (3328658, 3328659, 3328660)]
    if not all(os.path.isdir(os.path.join(d, "results")) for d in dirs):
        pytest.skip("extract raw/july/h100-33286{58,59,60}.tar.gz first (reproduce_diagnostic.sh, or REV4_JULY_ROOT)")
    out = S.spread(dirs, S.FIGURE_CONDITIONS)
    assert len(out["conditions"]) == 10
    assert out["largest"]["condition"] == "resnet50|gpu-topk|1"
    assert out["largest"]["spread_pct"] == pytest.approx(12.5, abs=0.05)
    assert out["conditions"]["resnet50|gpu-quant|1"]["spread_pct"] == pytest.approx(11.3, abs=0.05)
    assert out["conditions"]["resnet50|cpu-quant|1"]["mean_ms"] == pytest.approx(567.57, abs=0.01)
