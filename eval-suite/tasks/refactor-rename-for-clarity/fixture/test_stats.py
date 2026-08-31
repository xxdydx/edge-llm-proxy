from report import summarize


def test_summarize_equal_weights():
    assert summarize({10: 1, 20: 1}) == "weighted average: 15.00"


def test_summarize_weighted():
    assert summarize({10: 1, 30: 3}) == "weighted average: 25.00"


def test_summarize_zero_weights():
    assert summarize({10: 0, 20: 0}) == "weighted average: 0.00"
