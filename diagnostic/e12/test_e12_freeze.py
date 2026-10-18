# -*- coding: utf-8 -*-
"""diagnostic/E12_FREEZE.sha256 lists every file the protocol freezes with
its current hash, the frozen E6 to E8 components are the ones E7 and E8 ran,
and no two staged files share a basename (the job looks them up by name)."""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_freeze as M  # noqa: E402

E7_E8_FROZEN = {
    "diagnostic/rules.py": "71329c336cc3e9646dbb6e6dd4ce20e12305d666c663f2a3e4b35195b535a822",
    "diagnostic/features.py": "95e240d86e79301e6740276f851a30bf6c9289c093cdc5147de92117bed1c7dd",
    "diagnostic/score_e6.py": "928550323d0a5f8e9f48edc66c0b71fbd17489b4ae827afc3cf5d55a5dc9fede",
    "diagnostic/v1_1/rescore.py": "246d3f929b2b26978fe5af910177a054d58c4ec7876ea30c61a6ddadf3423dea",
    "diagnostic/e8/replay_reference.py": "357546661f63e26448732fdd662f762a60f830f259d66d27ccd95d5f1d23392b",
}


def test_freeze_file_matches_the_files():
    text = io.open(M.OUT, encoding="utf-8", newline="").read()
    assert "\r" not in text
    assert text.strip().split("\n") == M.lines()


def test_reused_components_are_the_e7_e8_frozen_files():
    for rel, want in E7_E8_FROZEN.items():
        assert M.digest(rel) == want, rel


def test_basenames_are_unique():
    names = [rel.rsplit("/", 1)[-1] for rel in M.FILES]
    assert len(names) == len(set(names))


def test_the_reverse_substitutions_are_frozen():
    for rel in ("diagnostic/e12/replay_reverse.py", "diagnostic/e12/test_replay_reverse.py",
                "diagnostic/e12/dev_reverse_e7a.py", "diagnostic/e12/DEV_REPORT_REVERSE.md"):
        assert rel in M.FILES, rel
    assert len(M.FILES) == 24
