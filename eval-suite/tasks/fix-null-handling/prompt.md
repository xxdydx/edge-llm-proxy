Your working directory contains a small Python module `records.py` and its
test suite `test_records.py`. Running `python3 -m pytest -q` currently shows a
failing test caused by an unhandled exception, not just a wrong return value.

Fix the bug in `records.py` so that every test in `test_records.py` passes,
without changing its behavior for records that do have a usable email. Do not
modify `test_records.py` or add any new files. Do not run `pip install` or
access the network. Run the tests yourself before you finish to confirm they
pass.
