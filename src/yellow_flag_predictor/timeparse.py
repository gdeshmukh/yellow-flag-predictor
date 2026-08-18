"""Parsing of timing-vendor clock strings into float seconds.

Timing exports write elapsed times in a variable number of fields with no zero
padding, so the same column mixes ``52.755``, ``1:37.228`` and ``23:45:12.345``.
Two properties of that format break naive parsers and cost the legacy project
both of its cleaning notebooks:

* Sub-minute laps (Lime Rock runs ~52 s) emit a bare ``SS.fff`` with no colon.
* Session clocks past 24 h are inconsistent within a single file: the 2020
  Rolex 24 export writes ``24:01:17.937`` for fourteen rows and then wraps to
  ``1:28.075`` meaning 24:01:28.075.

Field count alone therefore cannot disambiguate a wrapped value, so
:func:`parse_session_series` resolves the wrap by monotonicity against the
running clock rather than by string shape.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

_CLOCK = re.compile(r"^\s*(\d+)(?::(\d{1,2}))?(?::(\d{1,2}))?(?:\.(\d{1,6}))?\s*$")

SECONDS_PER_DAY = 86_400.0


class ClockParseError(ValueError):
    """Raised when a clock string does not match any known vendor format."""


def parse_clock(value: str) -> float:
    """Parse ``SS.fff``, ``M:SS.fff`` or ``H:MM:SS.fff`` into seconds.

    Unlike ``datetime.strptime`` this imposes no upper bound on the leading
    field, so ``26:01:17.937`` parses rather than raising.
    """
    if value is None:
        raise ClockParseError("empty clock value")
    text = str(value).strip()
    if not text:
        raise ClockParseError("empty clock value")

    m = _CLOCK.match(text)
    if not m:
        raise ClockParseError(f"unrecognised clock string: {value!r}")

    a, b, c, frac = m.groups()
    parts = [p for p in (a, b, c) if p is not None]
    if len(parts) == 1:
        h, mi, s = 0, 0, int(parts[0])
    elif len(parts) == 2:
        h, mi, s = 0, int(parts[0]), int(parts[1])
    else:
        h, mi, s = int(parts[0]), int(parts[1]), int(parts[2])

    total = h * 3600.0 + mi * 60.0 + s
    if frac:
        total += int(frac) / (10 ** len(frac))
    return total


def parse_session_series(values: Iterable[str], wrap_slack: float = 300.0) -> list[float]:
    """Parse a whole session-time column, repairing 24 h wraps.

    A vendor clock that rolls past 24 h and a file that concatenates a fresh
    session both step backwards, so direction alone cannot tell them apart.
    A roll is identified by *size*: the backwards jump lands within
    ``wrap_slack`` of a whole number of days, because the clock kept running
    while only the printed field reset. A practice-to-race restart lands
    nowhere near a day boundary and is passed through, leaving it for
    :func:`split_sessions` to cut.
    """
    out: list[float] = []
    offset = 0.0
    peak = 0.0
    for raw in values:
        base = parse_clock(raw)
        t = base + offset
        if peak - t > WRAP_TOLERANCE:
            days = round((peak - t) / SECONDS_PER_DAY)
            if days >= 1 and abs((peak - t) - days * SECONDS_PER_DAY) <= wrap_slack:
                offset += days * SECONDS_PER_DAY
                t = base + offset
        peak = max(peak, t)
        out.append(t)
    return out


# Cars cross the line seconds apart, so genuine out-of-order rows differ by at
# most a lap. Anything beyond an hour backwards is a clock wrap.
WRAP_TOLERANCE = 3600.0


def split_sessions(
    raw_times: Sequence[float], gap: float = 900.0, reset_tolerance: float = 120.0
) -> list[tuple[int, int]]:
    """Split a row index range wherever the clock resets or idles.

    Must run on *raw* per-row clocks, before :func:`parse_session_series`
    repairs 24 h wraps: a practice-to-race reset and a 24 h wrap look identical
    to the wrap repair, so splitting first is what stops a fresh session from
    being lifted a day into the future.

    Legacy exports concatenated practice, qualifying and race into one file
    while treating the result as a single continuous session, which put a
    "caution" at 4:35:09 inside a 2 h 40 m race. Returns half-open
    ``(start, stop)`` index pairs, one per detected session.

    ``reset_tolerance`` absorbs the small backwards steps that timing feeds
    emit when two cars are scored out of order; anything larger is a restart.
    """
    if len(raw_times) == 0:
        return []
    bounds = [0]
    for i in range(1, len(raw_times)):
        went_back = raw_times[i] < raw_times[i - 1] - reset_tolerance
        idled = raw_times[i] - raw_times[i - 1] > gap
        if went_back or idled:
            bounds.append(i)
    bounds.append(len(raw_times))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
