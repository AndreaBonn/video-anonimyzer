"""Tests for web.sse_stream."""

import queue

from person_anonymizer.web.sse_stream import TIMEOUT_MESSAGE, format_sse_event, iter_sse

MAX_DURATION = 7200
HEARTBEAT = 60
HEARTBEAT_FRAME = ": heartbeat\n\n"


class ScriptedQueue:
    """Queue stand-in: each get() returns the next scripted item or raises Empty."""

    def __init__(self, *items):
        self._items = list(items)
        self.timeouts = []

    def get(self, timeout):
        self.timeouts.append(timeout)
        item = self._items.pop(0)
        if item is queue.Empty:
            raise queue.Empty
        return item


def _clock(*values):
    it = iter(values)
    return lambda: next(it)


def test_format_sse_event_exact_frame():
    assert format_sse_event("x", {"a": 1}) == 'event: x\ndata: {"a": 1}\n\n'


def test_events_are_framed_until_none_sentinel():
    q = queue.Queue()
    q.put({"type": "progress", "data": {"pct": 5}})
    q.put(None)
    q.put({"type": "late", "data": {}})

    frames = list(iter_sse(q, clock=_clock(0), max_duration_s=MAX_DURATION, heartbeat_s=HEARTBEAT))

    assert frames == ['event: progress\ndata: {"pct": 5}\n\n']


def test_idle_before_deadline_yields_heartbeat_and_keeps_streaming():
    q = ScriptedQueue(queue.Empty, {"type": "done", "data": {}}, None)

    frames = list(
        iter_sse(q, clock=_clock(0, 60), max_duration_s=MAX_DURATION, heartbeat_s=HEARTBEAT)
    )

    assert frames == [HEARTBEAT_FRAME, "event: done\ndata: {}\n\n"]
    assert q.timeouts == [HEARTBEAT, HEARTBEAT, HEARTBEAT]


def test_idle_past_deadline_yields_heartbeat_then_timeout_and_stops():
    q = ScriptedQueue(queue.Empty, {"type": "never", "data": {}})

    frames = list(
        iter_sse(
            q,
            clock=_clock(0, MAX_DURATION + 1),
            max_duration_s=MAX_DURATION,
            heartbeat_s=HEARTBEAT,
        )
    )

    assert frames == [
        HEARTBEAT_FRAME,
        format_sse_event("timeout", {"message": TIMEOUT_MESSAGE}),
    ]


def test_elapsed_exactly_at_limit_is_not_a_timeout():
    q = ScriptedQueue(queue.Empty, None)

    frames = list(
        iter_sse(
            q,
            clock=_clock(0, MAX_DURATION),
            max_duration_s=MAX_DURATION,
            heartbeat_s=HEARTBEAT,
        )
    )

    assert frames == [HEARTBEAT_FRAME]
