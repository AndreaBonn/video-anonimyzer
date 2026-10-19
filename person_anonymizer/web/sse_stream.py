"""Framing and iteration of Server-Sent Events, independent of Flask."""

import json
import queue
from collections.abc import Callable, Iterator

TIMEOUT_MESSAGE = "Connessione SSE scaduta dopo 2h"


def format_sse_event(event_type: str, data: dict) -> str:
    """Format one SSE event frame."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def iter_sse(
    events: queue.Queue,
    clock: Callable[[], float],
    max_duration_s: float,
    heartbeat_s: float,
) -> Iterator[str]:
    """Yield SSE frames from ``events`` until a ``None`` sentinel or timeout.

    An idle queue yields a heartbeat comment every ``heartbeat_s`` seconds;
    once the idle check finds more than ``max_duration_s`` elapsed, a
    ``timeout`` event is yielded and iteration stops.

    Parameters
    ----------
    events : queue.Queue
        Queue of ``{"type": ..., "data": ...}`` dicts; ``None`` ends the stream.
    clock : Callable[[], float]
        Monotonic clock, injected so tests need no real waiting.
    max_duration_s : float
        Maximum stream duration, checked on heartbeats.
    heartbeat_s : float
        Seconds to wait on the queue before sending a heartbeat.
    """
    start = clock()
    while True:
        try:
            event = events.get(timeout=heartbeat_s)
        except queue.Empty:
            yield ": heartbeat\n\n"
            if clock() - start > max_duration_s:
                yield format_sse_event("timeout", {"message": TIMEOUT_MESSAGE})
                return
            continue
        if event is None:
            return
        yield format_sse_event(event["type"], event["data"])
