"""Field instability from lap-time dispersion.

A field whose lap times are spreading is a field under stress -- traffic,
weather, tyres going off, drivers on the edge. The legacy project measured this
as a rolling per-driver standard deviation averaged across drivers, which is
kept, with two corrections: laps run under caution are excluded (a caution
trivially inflates dispersion, which would leak the outcome into the feature),
and dispersion is measured per class before averaging, because a DPi and a GTD
lapping seconds apart is a class difference, not instability.
"""

from __future__ import annotations

import math

import polars as pl

from .._causal import completed_before


def lap_variance_series(
    laps: pl.DataFrame, grid: list[float], *, window_s: float = 600.0,
    lag_s: float = 0.0,
) -> pl.DataFrame:
    """Rolling within-car lap-time dispersion, averaged over the field."""
    green = laps.filter(
        (pl.col("flag") == "green") & (~pl.col("in_pit")) & pl.col("lap_time_s").is_not_null()
    ).sort("session_time_s")

    times = green["session_time_s"].to_list()
    cars = green["car"].to_list()
    classes = green["car_class"].to_list()
    lts = green["lap_time_s"].to_list()

    rows = []
    for t in grid:
        hi = completed_before(t, lag_s)
        lo = hi - window_s
        by_car: dict[str, list[float]] = {}
        by_class: dict[str, list[float]] = {}
        for i, ts in enumerate(times):
            if lo <= ts <= hi:
                by_car.setdefault(cars[i], []).append(lts[i])
                by_class.setdefault(classes[i], []).append(lts[i])
        stds = [_std(v) for v in by_car.values() if len(v) >= 3]
        cls_stds = [_std(v) for v in by_class.values() if len(v) >= 3]
        rows.append({
            "t": t,
            "lap_std_mean": (sum(stds) / len(stds)) if stds else None,
            "lap_std_max": max(stds) if stds else None,
            "class_lap_std_mean": (sum(cls_stds) / len(cls_stds)) if cls_stds else None,
            "n_green_laps_window": sum(len(v) for v in by_car.values()),
        })
    return pl.DataFrame(rows)


def _std(v: list[float]) -> float:
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
