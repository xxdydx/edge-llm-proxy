from records import first_valid_email


def test_skips_missing_key():
    records = [{"name": "no email field"}, {"email": "User@Example.com"}]
    assert first_valid_email(records) == "user@example.com"


def test_skips_none_email():
    records = [{"email": None}, {"email": "  Second@Example.com  "}]
    assert first_valid_email(records) == "second@example.com"


def test_skips_empty_string():
    records = [{"email": ""}, {"email": "Third@Example.com"}]
    assert first_valid_email(records) == "third@example.com"


def test_returns_none_when_no_valid_email():
    records = [{"name": "a"}, {"email": None}, {"email": ""}]
    assert first_valid_email(records) is None
