Your working directory contains `shipping_us.py` and `shipping_eu.py`. Both
define an identical `estimate_shipping_days(distance_km, express=False)`
function - the same implementation copy-pasted into two files.
`test_shipping.py` imports `estimate_shipping_days` from both modules and
exercises it; all tests currently pass.

Create a new module `shipping_common.py` containing a single implementation
of `estimate_shipping_days`, and change `shipping_us.py` and
`shipping_eu.py` so each imports and re-exposes it (a thin re-export or
wrapper is fine) instead of defining its own copy of the logic.
`estimate_shipping_days` must remain importable from both `shipping_us` and
`shipping_eu` with unchanged behavior.

Do not modify `test_shipping.py`. Do not run `pip install` or access the
network. Run the tests yourself before you finish to confirm they still
pass.
