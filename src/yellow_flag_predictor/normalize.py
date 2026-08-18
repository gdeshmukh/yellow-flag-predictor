"""Vendor timing exports to the canonical schema.

The export format is stable across circuits and eras apart from the sector
count and the name of the trap-speed column (``SPI`` at Daytona, ``SPS`` at
Lime Rock, ``SP3`` at Watkins Glen), so sector columns are discovered rather
than enumerated.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import polars as pl

from .schema import Flag, SessionKind
from .timeparse import ClockParseError, parse_clock, parse_session_series, split_sessions

_SECTOR_COL = re.compile(r"^S(\d{1,2})$")
_TRAP_COL = re.compile(r"^SP[A-Z0-9]?$")

_FLAG_MAP = {
    "green": Flag.GREEN,
    "yellow": Flag.YELLOW,
    "fcy": Flag.YELLOW,
    "full course yellow": Flag.YELLOW,
    "caution": Flag.YELLOW,
    "sc": Flag.YELLOW,
    "safety car": Flag.YELLOW,
    "red": Flag.RED,
    "finish": Flag.CHECKERED,
    "checkered": Flag.CHECKERED,
    "chequered": Flag.CHECKERED,
}


def canonical_flag(raw: str | None) -> str:
    if raw is None:
        return Flag.UNKNOWN
    return _FLAG_MAP.get(str(raw).strip().lower(), Flag.UNKNOWN)


def _sector_columns(header: list[str]) -> list[str]:
    cols = [(int(m.group(1)), c) for c in header if (m := _SECTOR_COL.match(c.strip()))]
    return [c for _, c in sorted(cols)]


def _trap_column(header: list[str]) -> str | None:
    for c in header:
        if _TRAP_COL.match(c.strip()):
            return c
    return None


def read_vendor_csv(
    path: str | Path,
    *,
    circuit: str,
    year: int,
    event: str,
    kind: str = SessionKind.RACE,
    min_session_rows: int = 50,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read one vendor CSV into ``(laps, sessions)``.

    A single file may hold several sessions; each is emitted with its own
    ``session_id`` and its clock rebased to zero, so a practice run can never
    contribute a lap or a caution to the race.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: no rows")

    header = list(rows[0].keys())
    sector_cols = _sector_columns(header)
    trap_col = _trap_column(header)
    if not sector_cols:
        raise ValueError(f"{path}: no sector columns (S01..SNN) found in {header}")

    # Repair 24 h rolls across the whole file first: a roll is a printing
    # artefact of one continuous clock, so it must not be read as a session
    # boundary. What survives repair is a genuine restart.
    file_clock = parse_session_series(r["Session Time"] for r in rows)
    spans = split_sessions(file_clock)
    spans = [(a, b) for a, b in spans if b - a >= min_session_rows]
    if not spans:
        spans = [(0, len(rows))]

    lap_frames: list[pl.DataFrame] = []
    session_records: list[dict] = []

    for n, (start, stop) in enumerate(spans, 1):
        chunk = rows[start:stop]
        session_id = f"{circuit}-{year}-{kind}-{n}"

        # The vendor session clock is kept as-is. It already measures elapsed
        # time from the session start, so rebasing would shift every lap by the
        # leader's first lap time.
        session_s = file_clock[start:stop]

        lap_time_s, sectors, traps = [], [], []
        for r in chunk:
            try:
                lap_time_s.append(parse_clock(r["Lap Time"]))
            except ClockParseError:
                lap_time_s.append(None)
            sectors.append([_maybe_float(r.get(c)) for c in sector_cols])
            traps.append(_maybe_float(r.get(trap_col)) if trap_col else None)

        frame = pl.DataFrame(
            {
                "session_id": [session_id] * len(chunk),
                "car": [str(r["Car"]).strip() for r in chunk],
                "car_class": [str(r.get("Class", "")).strip() for r in chunk],
                "driver": [str(r.get("Driver", "")).strip() for r in chunk],
                "lap": [_maybe_int(r.get("Lap")) for r in chunk],
                "lap_time_s": lap_time_s,
                "session_time_s": session_s,
                "flag": [canonical_flag(r.get("Flag")) for r in chunk],
                "in_pit": [str(r.get("Location", "")).strip().lower() == "pit" for r in chunk],
                "sectors_s": sectors,
                "trap_speed": traps,
            },
            schema_overrides={"lap": pl.Int32, "sectors_s": pl.List(pl.Float64)},
        ).with_columns(
            (pl.col("session_time_s") - pl.col("lap_time_s")).alias("lap_start_s")
        )

        lap_frames.append(frame)
        session_records.append(
            {
                "session_id": session_id,
                "circuit": circuit,
                "year": year,
                "kind": kind,
                "event": event,
                "n_sectors": len(sector_cols),
                "duration_s": max(session_s) - min(session_s),
                "n_cars": frame["car"].n_unique(),
                "source_file": str(path),
            }
        )

    laps = pl.concat(lap_frames) if lap_frames else pl.DataFrame()
    sessions = pl.DataFrame(session_records, schema_overrides={"year": pl.Int32, "n_sectors": pl.Int32, "n_cars": pl.Int32})
    return laps, sessions


def _maybe_float(v) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _maybe_int(v) -> int | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def sector_sum_residual(laps: pl.DataFrame) -> pl.DataFrame:
    """Per-lap disagreement between summed sectors and the reported lap time.

    A clean flying lap closes to within timing resolution. A large positive
    residual means time was spent stationary inside one sector -- a pit or
    garage stint absorbed into the lap -- which is the signal used to drop
    those laps from track-position features rather than let a parked car be
    modelled as traffic.
    """
    return laps.with_columns(
        (pl.col("lap_time_s") - pl.col("sectors_s").list.sum()).alias("sector_residual_s")
    )
