# -*- coding: utf-8 -*-
"""The v2 tests import the harness, the manifest builder, and the analysis
from their own folders, as they sat side by side in the campaign folder."""
import os
import sys

V2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for sub in ("harness", "manifests", "analysis"):
    path = os.path.join(V2, sub)
    if path not in sys.path:
        sys.path.insert(0, path)
