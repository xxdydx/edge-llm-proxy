from shipping_us import estimate_shipping_days as us_estimate
from shipping_eu import estimate_shipping_days as eu_estimate


def test_us_and_eu_agree():
    assert us_estimate(1600) == eu_estimate(1600)


def test_basic_estimate():
    assert us_estimate(0) == 1
    assert us_estimate(800) == 2


def test_express_is_faster():
    assert us_estimate(1600, express=True) < us_estimate(1600, express=False)


def test_minimum_one_day():
    assert eu_estimate(0, express=True) == 1
