Your working directory contains `stats.py` and `report.py`; `report.py`
imports and calls a function named `calc` from `stats.py`. The name `calc`
is too generic for what it actually does.

Rename `calc` to `weighted_average` everywhere it is defined or referenced
in this package (in both `stats.py` and `report.py`), without changing its
behavior. After the rename, no source file should still reference the name
`calc`.

Do not modify `test_stats.py`. Do not run `pip install` or access the
network. Run the tests yourself before you finish to confirm they still
pass.
