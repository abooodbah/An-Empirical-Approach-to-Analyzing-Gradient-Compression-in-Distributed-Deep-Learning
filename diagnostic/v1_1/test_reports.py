# -*- coding: utf-8 -*-
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report  # noqa: E402
import literal_frozen_extractor as L  # noqa: E402

FROZEN_FEATURES_SHA = "a3eef19d"


def test_report_is_deterministic_and_complete():
    a, b = report.build(), report.build()
    assert a == b
    for needle in ("RETROSPECTIVE", "reading frozen", "reading described",
                   "reading as_frozen_extractor", "A2", "cleared 5%"):
        assert needle in a


@pytest.mark.skipif(not os.path.isdir(os.environ.get("R4W", "C:/r4w")),
                    reason="round-4 records not extracted")
def test_literal_frozen_extractor_cannot_score_e6():
    out = L.run(os.environ.get("R4W", "C:/r4w"))
    assert out["sha256"].startswith(FROZEN_FEATURES_SHA)
    assert out["rows"] > 0 and out["wall_rows"] == 0
