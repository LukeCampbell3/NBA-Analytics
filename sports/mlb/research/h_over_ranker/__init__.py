"""H-OVER-specific ranking research (development-only; TEST block retired).

This package develops a ranker over the frozen H-OVER-eligible candidate
population found in sports/mlb/data/predictions/daily_runs/. It does NOT
optimize parlays/combos -- see the module docstrings in `manifest.py` for the
frozen H_OVER_RANKER_V1 config and the confirmation policy that governs it.

Data discipline (see `data_windows.py`):
  DERIVE (first third of archived days)  -> bias correction only
  SELECT (second third)                  -> chose H as the eligible target
  TEST   (final third, 9 days)           -> RETIRED. Frozen historical result
                                             only; never read by any code in
                                             this package again.
All ranker development in this package uses DERIVE + SELECT only.
"""
