"""Caution events and the supervised target.

Flags are recorded per car at the moment it crosses the line, so a caution is
never observed directly -- it is bracketed between the last green crossing and
the first yellow one, which at Daytona pins onset to about ten seconds. That
bracket also defines the leakage boundary: the lap *ending* at onset was partly
run after the flag flew and already contains the slowdown, so features must be
built from laps completed strictly before ``onset - lap_time``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .schema import Flag


def derive_cautions(laps: pl.DataFrame, min_gap_s: float = 180.0) -> pl.DataFrame:
    """Extract caution periods from per-lap flag states.

    Yellow crossings are clustered in time; a gap shorter than ``min_gap_s``
    belongs to the same caution, since cars cross the line minutes apart and a
    single caution produces a ragged stream of yellow rows.
    """
    out = []
    for session_id, grp in laps.group_by("session_id", maintain_order=True):
        sid = session_id[0] if isinstance(session_id, tuple) else session_id
        grp = grp.sort("session_time_s")
        yellows = grp.filter(pl.col("flag") == Flag.YELLOW)["session_time_s"].to_list()
        if not yellows:
            continue

        clusters: list[list[float]] = [[yellows[0]]]
        for t in yellows[1:]:
            if t - clusters[-1][-1] > min_gap_s:
                clusters.append([t])
            else:
                clusters[-1].append(t)

        greens = grp.filter(pl.col("flag") == Flag.GREEN)["session_time_s"].to_list()
        for i, cl in enumerate(clusters):
            first_yellow = cl[0]
            # Onset lies between the last green crossing and the first yellow.
            prior_green = [g for g in greens if g < first_yellow]
            lower = max(prior_green) if prior_green else first_yellow
            after_green = [g for g in greens if g > cl[-1]]
            out.append(
                {
                    "session_id": sid,
                    "caution_idx": i,
                    "start_s": (lower + first_yellow) / 2.0,
                    "start_lower_s": lower,
                    "start_upper_s": first_yellow,
                    "end_s": min(after_green) if after_green else cl[-1],
                    "cause_car": None,
                    "cause_kind": "unknown",
                    "source": "derived",
                }
            )
    schema = {
        "session_id": pl.Utf8, "caution_idx": pl.Int32,
        "start_s": pl.Float64, "start_lower_s": pl.Float64, "start_upper_s": pl.Float64,
        "end_s": pl.Float64, "cause_car": pl.Utf8, "cause_kind": pl.Utf8, "source": pl.Utf8,
    }
    return pl.DataFrame(out, schema=schema) if out else pl.DataFrame(schema=schema)


def load_annotations(path: str | Path) -> pl.DataFrame:
    """Load hand-annotated cautions.

    Cause attribution cannot be derived from timing data -- the export records
    that a caution happened, never why -- so this file is authored by hand and
    version-controlled as reference data.
    """
    path = Path(path)
    if not path.exists():
        return pl.DataFrame(schema={"session_id": pl.Utf8, "start_s": pl.Float64,
                                    "end_s": pl.Float64, "cause_car": pl.Utf8,
                                    "cause_kind": pl.Utf8, "source": pl.Utf8})
    return pl.read_csv(path, schema_overrides={"cause_car": pl.Utf8, "session_id": pl.Utf8})


def merge_cautions(derived: pl.DataFrame, annotated: pl.DataFrame,
                   tolerance_s: float = 120.0) -> pl.DataFrame:
    """Attach hand annotations to derived cautions by onset proximity.

    Annotations that match nothing derived are kept: a session whose export
    carries no yellow rows can still have had cautions, and dropping them would
    silently understate the positive class.
    """
    if annotated.height == 0:
        return derived
    rows = derived.to_dicts()
    unmatched = []
    for a in annotated.to_dicts():
        best, best_d = None, None
        for r in rows:
            if r["session_id"] != a["session_id"]:
                continue
            d = abs(r["start_s"] - a["start_s"])
            if d <= tolerance_s and (best_d is None or d < best_d):
                best, best_d = r, d
        if best is not None:
            best["cause_car"] = a.get("cause_car") or best["cause_car"]
            best["cause_kind"] = a.get("cause_kind") or best["cause_kind"]
            best["source"] = "derived+annotated"
        else:
            unmatched.append({
                "session_id": a["session_id"], "caution_idx": -1,
                "start_s": a["start_s"], "start_lower_s": a["start_s"],
                "start_upper_s": a["start_s"],
                "end_s": a.get("end_s") if a.get("end_s") is not None else a["start_s"],
                "cause_car": a.get("cause_car"), "cause_kind": a.get("cause_kind") or "unknown",
                "source": "annotated",
            })
    merged = pl.DataFrame(rows, schema=derived.schema) if rows else derived
    if unmatched:
        merged = pl.concat([merged, pl.DataFrame(unmatched, schema=derived.schema)])
    return merged.sort(["session_id", "start_s"])


def make_labels(
    cautions: pl.DataFrame,
    session_id: str,
    duration_s: float,
    *,
    step_s: float = 20.0,
    horizon_s: float = 300.0,
) -> pl.DataFrame:
    """Discrete-time hazard target on a fixed grid.

    ``y = 1`` where a caution begins in ``(t, t + horizon_s]``. Bins already
    under caution are marked ``at_risk = False`` and must be dropped before
    fitting: a caution cannot begin while one is running, so scoring those bins
    inflates the apparent negative class and flatters every metric.

    ``horizon_s`` is the decision window. Five minutes is roughly the span over
    which a strategist can still choose to pit before a likely caution; a
    60 s horizon (the legacy choice) is too short to act on and starves an
    already tiny positive class.
    """
    sess = cautions.filter(pl.col("session_id") == session_id).sort("start_s")
    starts = sess["start_s"].to_list()
    windows = list(zip(sess["start_s"].to_list(), sess["end_s"].to_list()))

    grid = [t for t in _frange(0.0, duration_s, step_s)]
    y, at_risk, tt_next = [], [], []
    for t in grid:
        under = any(a <= t <= b for a, b in windows)
        nxt = [s for s in starts if s > t]
        dt = (nxt[0] - t) if nxt else None
        y.append(int(dt is not None and dt <= horizon_s))
        at_risk.append(not under)
        tt_next.append(dt)
    return pl.DataFrame(
        {"session_id": [session_id] * len(grid), "t": grid, "y": y,
         "at_risk": at_risk, "time_to_next_caution_s": tt_next},
        schema={"session_id": pl.Utf8, "t": pl.Float64, "y": pl.Int8,
                "at_risk": pl.Boolean, "time_to_next_caution_s": pl.Float64},
    )


def _frange(start: float, stop: float, step: float):
    t = start
    while t <= stop:
        yield t
        t += step
