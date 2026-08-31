from mod_b import preferred_phone


def test_prefers_mobile():
    assert preferred_phone({"mobile": "555-1000", "home": "555-2000"}) == "555-1000"


def test_falls_back_to_home():
    assert preferred_phone({"mobile": None, "home": "555-2000"}) == "555-2000"


def test_missing_mobile_key():
    assert preferred_phone({"home": "555-2000"}) == "555-2000"


def test_no_numbers_returns_none():
    assert preferred_phone({"mobile": None, "home": None}) is None
