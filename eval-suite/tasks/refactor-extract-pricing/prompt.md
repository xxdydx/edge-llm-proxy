Your working directory contains `orders.py` and its test suite
`test_orders.py`; all tests currently pass.

Refactor `process_order` in `orders.py`: extract the subtotal-and-discount
computation into a new top-level function named exactly `compute_total(items,
discount_pct)` that returns a float. Update `process_order` to call
`compute_total` instead of computing the total inline, keeping its
validation, its `"$X.XX"` return format, and all existing behavior
unchanged.

Do not modify `test_orders.py`. Do not run `pip install` or access the
network. Run the tests yourself before you finish to confirm they still
pass.
