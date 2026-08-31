"""Reporting helpers built on top of stats."""

from stats import calc


def summarize(scores_by_weight):
    """Given a dict of {score: weight}, return a one-line weighted-average summary."""
    values = list(scores_by_weight.keys())
    weights = list(scores_by_weight.values())
    average = calc(values, weights)
    return f"weighted average: {average:.2f}"
