from discounts import bulk_discount_rate


def test_below_tier():
    assert bulk_discount_rate(5) == 0.0


def test_low_tier():
    assert bulk_discount_rate(10) == 0.05


def test_mid_tier_boundary_included():
    assert bulk_discount_rate(50) == 0.10


def test_mid_tier():
    assert bulk_discount_rate(75) == 0.10


def test_top_tier_boundary_included():
    assert bulk_discount_rate(100) == 0.20
