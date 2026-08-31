from mod_a import letter_grade


def test_a_range():
    assert letter_grade(95) == "A"


def test_b_boundary_included():
    assert letter_grade(80) == "B"


def test_b_range():
    assert letter_grade(85) == "B"


def test_c_range():
    assert letter_grade(75) == "C"


def test_f_range():
    assert letter_grade(50) == "F"
