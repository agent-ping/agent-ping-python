from agentping._queue import BoundedQueue


def test_put_and_drain():
    q = BoundedQueue(maxsize=10)
    for i in range(5):
        q.put({"i": i})
    assert len(q) == 5
    items = q.drain(10)
    assert [it["i"] for it in items] == [0, 1, 2, 3, 4]
    assert len(q) == 0


def test_drop_oldest_when_full():
    q = BoundedQueue(maxsize=3)
    for i in range(5):
        q.put({"i": i})
    assert len(q) == 3
    assert q.dropped == 2
    items = q.drain(10)
    assert [it["i"] for it in items] == [2, 3, 4]


def test_push_front_for_retry():
    q = BoundedQueue(maxsize=10)
    q.put({"i": 0})
    q.put({"i": 1})
    q.push_front([{"i": -1}])
    items = q.drain(10)
    assert [it["i"] for it in items] == [-1, 0, 1]


def test_push_front_drops_newest_when_full():
    q = BoundedQueue(maxsize=2)
    q.put({"i": 0})
    q.put({"i": 1})
    q.push_front([{"i": -1}])
    items = q.drain(10)
    assert [it["i"] for it in items] == [-1, 0]
    assert q.dropped == 1
