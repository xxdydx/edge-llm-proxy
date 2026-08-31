from invoices import invoice_total, is_overdue


def test_invoice_total():
    assert invoice_total([{"price": 10, "qty": 2}, {"price": 5, "qty": 1}]) == 25


def test_is_overdue():
    assert is_overdue(31) is True
    assert is_overdue(29) is False
