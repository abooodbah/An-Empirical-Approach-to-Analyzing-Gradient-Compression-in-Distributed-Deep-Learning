# Corrections to the E6 evaluation (RETROSPECTIVE, 2026-09-29)

Found by an audit after the E6 outcomes were known. The frozen files are
unchanged; `rescore.py` applies each correction to a copy of the frozen
rows, and `E6_CORRECTED.txt` reports every reading.

| Id | What | Why | Effect on counts |
|---|---|---|---|
| C1 | Hooks with no exchange or codec phase get codec_share None | the frozen extractor stores 0.0, turning "feature unavailable" (insufficient) into "rule did not fire" (none) | labels only; no count changes |
| C2 | A1's r_V from the configuration (p/2c) where no payload bytes were recorded | the paper defines A1 by r_V, which the hooks' configuration determines | A1 5 -> 6 correct (case 7) |
| C3 | Nominal r_V = p/8 for int8 gathers | cases 3 and 12 clear 1.5 only through scale and metadata bytes | A1 at nominal (frozen hooks) 3 correct, 3 cleared 5%; the paper printed 4 and 4 |
| C4 | Shape effect against the named intervention, exchange share x (1 - 2/p) | the named intervention is a reduce of the same encoding, not a dense all-reduce | e 0.337/0.197/0.364 -> 0.464/0.492/0.501 (cases 2, 3, 4); no outcome changes |
| C5 | "inapplicable" and "not tested" labels | the frozen scorer never compares the named and the tested intervention | A3 case 9 null -> inapplicable; A2 case 11 unscored -> inapplicable |
