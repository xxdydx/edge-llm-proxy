import pytest
from validation import validate_title, validate_status, validate_role


def test_validate_title_strips_whitespace():
    assert validate_title("  Bug  ") == "Bug"


def test_validate_title_rejects_empty():
    with pytest.raises(ValueError):
        validate_title("   ")


def test_validate_status_rejects_unknown():
    with pytest.raises(ValueError):
        validate_status("archived")


def test_validate_role_rejects_unknown():
    with pytest.raises(ValueError):
        validate_role("guest")
