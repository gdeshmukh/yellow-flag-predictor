"""Canonical schema for normalised timing data.

One row per car per lap, plus a separate caution-event table. Sector times are
held in a list column rather than fixed ``S01..S13`` columns because sector
count is a property of the circuit and the era, not of the sport: Lime Rock
exports 8, the 6 h Watkins Glen export 11, Daytona 13. Fixed columns are what
forced the legacy project to keep a separate copy of every notebook per year.
"""

from __future__ import annotations

from enum import StrEnum

import polars as pl


class Flag(StrEnum):
    """Flag state recorded at the moment a car crossed the line.

    Vendors spell the same condition several ways across eras, so
    :func:`yellow_flag_predictor.normalize.canonical_flag` maps onto this set.
    """

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    CHECKERED = "checkered"
    UNKNOWN = "unknown"


class SessionKind(StrEnum):
    PRACTICE = "practice"
    QUALIFYING = "qualifying"
    RACE = "race"
    UNKNOWN = "unknown"


#: One row per car per completed lap.
LAP_SCHEMA = {
    "session_id": pl.Utf8,      # "<circuit>-<year>-<kind>-<n>"
    "car": pl.Utf8,             # keep as text: "911" and "11" are distinct entries
    "car_class": pl.Utf8,
    "driver": pl.Utf8,
    "lap": pl.Int32,
    "lap_time_s": pl.Float64,
    "session_time_s": pl.Float64,   # elapsed at line crossing, wrap-repaired
    "lap_start_s": pl.Float64,      # session_time_s - lap_time_s
    "flag": pl.Utf8,
    "in_pit": pl.Boolean,
    "sectors_s": pl.List(pl.Float64),
    "trap_speed": pl.Float64,       # nullable; derived from the trap sector
}

#: One row per full-course caution. The legacy project never built this table:
#: it recorded flag state per lap but never the event, and never the cause.
#: ``cause_car`` and ``cause_kind`` are hand-annotated and cannot be recovered
#: from timing data, which is why reference/cautions.csv is version-controlled.
CAUTION_SCHEMA = {
    "session_id": pl.Utf8,
    "caution_idx": pl.Int32,
    "start_s": pl.Float64,
    "end_s": pl.Float64,
    "cause_car": pl.Utf8,       # nullable
    "cause_kind": pl.Utf8,      # contact | mechanical | debris | weather | unknown
    "source": pl.Utf8,          # derived | annotated
}

#: One row per session, so a file holding practice+qualifying+race splits cleanly.
SESSION_SCHEMA = {
    "session_id": pl.Utf8,
    "circuit": pl.Utf8,
    "year": pl.Int32,
    "kind": pl.Utf8,
    "event": pl.Utf8,
    "n_sectors": pl.Int32,
    "duration_s": pl.Float64,
    "n_cars": pl.Int32,
    "source_file": pl.Utf8,
}


def empty(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)
