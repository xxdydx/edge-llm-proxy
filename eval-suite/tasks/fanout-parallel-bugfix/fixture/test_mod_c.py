from mod_c import sort_tasks


def test_orders_by_priority_descending():
    tasks = [
        {"title": "a", "priority": 1, "due": 5},
        {"title": "b", "priority": 3, "due": 2},
    ]
    result = sort_tasks(tasks)
    assert [t["title"] for t in result] == ["b", "a"]


def test_ties_broken_by_due_date_not_title():
    tasks = [
        {"title": "zeta", "priority": 2, "due": 1},
        {"title": "alpha", "priority": 2, "due": 5},
    ]
    result = sort_tasks(tasks)
    assert [t["title"] for t in result] == ["zeta", "alpha"]
