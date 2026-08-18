"""Assembly of the per-session feature matrix.

One row per grid time. Every column is causal by construction -- see
:mod:`yellow_flag_predictor._causal` -- and the label is attached last so that a feature
can never be derived from it.
"""

from __future__ import annotations

import polars as pl

from ..labels import make_labels
from ..tracks import Circuit
from .consistency import lap_variance_series
from .pit import car_activity, pit_stops, restop_pressure_series, time_since_pit_series
from .traffic import concentration_series

#: Columns offered to a model. ``tsp_std`` is deliberately absent: it tracks
#: ``tsp_max`` at r = 0.99, so including both leaves neither coefficient
#: identifiable. Run ``yellow-flag-predictor diagnose`` after changing this list.
#: Identifiers, diagnostics and the label are
#: deliberately excluded; ``coverage`` is included because a density computed
#: from part of the field means something different from the same number
#: computed from all of it.
FEATURE_COLUMNS = [
    "density_max",
    "density_mean",
    "coverage",
    "tsp_mean",
    "tsp_max",
    "tsp_frac_fresh",
    "restop_pressure",
    "recent_stops",
    "lap_std_mean",
    "lap_std_max",
    "class_lap_std_mean",
    "elapsed_frac",
    "time_since_last_caution_s",
]


#: The subset a pit-count timing feed can supply. Training against this list
#: produces a model that scores every live update; training against the full
#: list produces a better-informed model that a sector-less feed cannot run.
#: Which to use is a deployment decision, not a modelling one.
LIVE_FEATURE_COLUMNS = [
    c for c in FEATURE_COLUMNS
    if c not in ("density_max", "density_mean", "coverage")
]


def build_feature_matrix(
    laps: pl.DataFrame,
    cautions: pl.DataFrame,
    session_id: str,
    duration_s: float,
    circuit: Circuit,
    *,
    step_s: float = 20.0,
    horizon_s: float = 300.0,
    lap_lag_s: float = 110.0,
) -> pl.DataFrame:
    """Build the labelled feature matrix for one session."""
    laps = laps.filter(pl.col("session_id") == session_id)
    grid = [i * step_s for i in range(int(duration_s // step_s) + 1)]

    stops = pit_stops(laps)
    activity = car_activity(laps)
    frame = (
        concentration_series(laps, grid, circuit)
        .join(time_since_pit_series(stops, grid, activity), on="t", how="left")
        .join(restop_pressure_series(stops, grid), on="t", how="left")
        .join(lap_variance_series(laps, grid, lag_s=lap_lag_s), on="t", how="left")
    )

    starts = (
        cautions.filter(pl.col("session_id") == session_id).sort("start_s")["start_s"].to_list()
    )
    # Before the first caution the clock runs from the session start, not from
    # nothing: "no caution yet, 3 hours in" is a measurement, and emitting null
    # would drop every such row from the fit.
    since = []
    for t in grid:
        prior = [s for s in starts if s <= t]
        since.append(t - max(prior) if prior else t)

    frame = frame.with_columns(
        pl.Series("elapsed_frac", [t / duration_s for t in grid]),
        pl.Series("time_since_last_caution_s", since, dtype=pl.Float64),
        pl.lit(session_id).alias("session_id"),
    )

    labels = make_labels(
        cautions, session_id, duration_s, step_s=step_s, horizon_s=horizon_s
    )
    return frame.join(labels, on=["session_id", "t"], how="inner")
