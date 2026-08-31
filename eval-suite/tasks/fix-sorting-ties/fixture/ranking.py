"""Rank players by score, breaking ties by join order."""


def rank_players(players):
    """Return players sorted by score descending; ties broken by earliest join order."""
    return sorted(players, key=lambda p: (-p["score"], p["name"]))
