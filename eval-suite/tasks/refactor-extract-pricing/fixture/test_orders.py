import pytest
from orders import process_order


def test_simple_total():
    items = [{"price": 10.0, "qty": 2}]
    assert process_order(items) == "$20.00"


def test_applies_discount():
    items = [{"price": 10.0, "qty": 2}]
    assert process_order(items, discount_pct=0.1) == "$18.00"


def test_empty_items_raises():
    with pytest.raises(ValueError):
        process_order([])


def test_negative_price_raises():
    with pytest.raises(ValueError):
        process_order([{"price": -1.0, "qty": 1}])
