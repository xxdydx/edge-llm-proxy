from ranking import rank_players


def test_orders_by_score_descending():
    players = [
        {"name": "a", "score": 10, "joined_order": 1},
        {"name": "b", "score": 20, "joined_order": 2},
    ]
    result = rank_players(players)
    assert [p["name"] for p in result] == ["b", "a"]


def test_ties_broken_by_join_order_not_name():
    players = [
        {"name": "zed", "score": 15, "joined_order": 1},
        {"name": "amy", "score": 15, "joined_order": 2},
    ]
    result = rank_players(players)
    assert [p["name"] for p in result] == ["zed", "amy"]


def test_three_way_tie():
    players = [
        {"name": "c", "score": 5, "joined_order": 3},
        {"name": "a", "score": 5, "joined_order": 1},
        {"name": "b", "score": 5, "joined_order": 2},
    ]
    result = rank_players(players)
    assert [p["name"] for p in result] == ["a", "b", "c"]
