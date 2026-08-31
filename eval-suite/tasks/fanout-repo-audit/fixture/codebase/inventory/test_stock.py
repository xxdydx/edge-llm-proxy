from stock import reorder_needed, apply_restock, days_of_supply


def test_reorder_needed():
    assert reorder_needed(5, 10) is True
    assert reorder_needed(15, 10) is False


def test_apply_restock():
    assert apply_restock(5, 20) == 25


def test_days_of_supply():
    assert days_of_supply(100, 25) == 4.0
    assert days_of_supply(100, 0) is None
