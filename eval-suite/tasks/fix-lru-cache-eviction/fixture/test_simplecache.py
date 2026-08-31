from simplecache import LRUCache


def test_basic_put_get():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2


def test_evicts_least_recently_used_on_put():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # evicts "a": least recently used
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_get_refreshes_recency():
    cache = LRUCache(3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.get("a")  # touching "a" should protect it from eviction
    cache.get("a")
    cache.put("d", 4)  # should evict "b", the true least-recently-used
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4


def test_get_missing_key_returns_none():
    cache = LRUCache(2)
    assert cache.get("missing") is None
