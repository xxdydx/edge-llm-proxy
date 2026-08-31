"""Grade lookup by score threshold."""


def letter_grade(score):
    """Return a letter grade for a numeric score out of 100."""
    if score >= 90:
        return "A"
    if score > 80:
        return "B"
    if score >= 70:
        return "C"
    return "F"
