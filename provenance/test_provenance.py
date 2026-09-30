# -*- coding: utf-8 -*-
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_provenance as C  # noqa: E402


def test_provenance_consistent():
    assert C.verify(os.path.dirname(HERE)) == []
