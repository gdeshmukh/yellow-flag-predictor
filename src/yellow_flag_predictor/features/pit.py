"""Pit-cycle features.

Two distinct ideas, both carried over from the legacy project:

* **Time since pit.** Risk concentrates at two points in a stint -- immediately
  after a stop, when cold tyres rejoin a field at speed, and near the end of a
  fuel window, when stops bunch and cars are lapped mid-cycle.
* **Restop pressure.** A car that pits again far sooner than a fuel window
  explains is carrying damage or a problem. The legacy project scored this with
  a step function; it is kept as a smooth decay so that a 599 s and a 601 s gap
  do not differ by 20 points.
"""

from __future__ import annotations

import math

import polars as pl


def pit_stops(laps: pl.DataFrame) -> dict[str, list[float]]:
    """Session-time of each car's pit entries, ascending.

    Keyed on the lap that was *completed* in the pits, so the timestamp is when
    the stop was observed, not when it began.
    """
    stops: dict[str, list[float]] = {}
    pit = laps.filter(pl.col("in_pit")).sort("session_time_s")
    for car, grp in pit.group_by("car", maintain_order=True):
        key = car[0] if isinstance(car, tuple) else car
        stops[key] = grp["session_time_s"].to_list()
    for car in laps["car"].unique().to_list():
        stops.setdefault(car, [])
    return stops


def car_activity(laps: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """First and last time each car was seen crossing the line."""
    out: dict[str, tuple[float, float]] = {}
    for car, grp in laps.group_by("car"):
        key = car[0] if isinstance(car, tuple) else car
        ts = grp["session_time_s"]
        out[key] = (float(ts.min()), float(ts.max()))
    return out


def active_cars(
    activity: dict[str, tuple[float, float]], t: float, grace_s: float = 600.0
) -> list[str]:
    """Cars still circulating at ``t``.

    A car that has retired must leave the field summaries. Left in, its time
    since last pit grows without bound and becomes a proxy for elapsed race
    time -- which is exactly what happened here: ``tsp_max`` correlated with
    ``elapsed_frac`` at r = 1.000000, making both coefficients meaningless.

    Retirement is only observable by absence, so it is inferred causally: a car
    that has not completed a lap within ``grace_s`` is treated as out. The grace
    period must exceed a long pit or garage stop, or cars will drop out and
    return.
    """
    return [c for c, (first, last) in activity.items()
            if first <= t and t - last <= grace_s]


def time_since_pit_series(
    stops: dict[str, list[float]],
    grid: list[float],
    activity: dict[str, tuple[float, float]] | None = None,
    session_start: float = 0.0,
    grace_s: float = 600.0,
) -> pl.DataFrame:
    """Field-level summaries of minutes since each car's last stop.

    A car that has not yet stopped is measured from the session start, which is
    correct: it is on its first stint.

    The legacy project summed a Gaussian density over these values across cars
    and called the total a risk score. That total is not a probability and
    scales with car count, so it cannot be compared across races. Summaries are
    emitted here instead, and the shape of the risk is applied separately in
    :mod:`yellow_flag_predictor.risk`.
    """
    rows = []
    for t in grid:
        cars = active_cars(activity, t, grace_s) if activity is not None else list(stops)
        ages = []
        for car in cars:
            prior = [s for s in stops.get(car, []) if s <= t]
            last = max(prior) if prior else session_start
            ages.append((t - last) / 60.0)
        if not ages:
            rows.append({"t": t, "tsp_mean": None, "tsp_max": None,
                         "tsp_min": None, "tsp_std": None, "tsp_frac_fresh": None})
            continue
        n = len(ages)
        mean = sum(ages) / n
        var = sum((a - mean) ** 2 for a in ages) / n
        rows.append({
            "t": t,
            "tsp_mean": mean,
            "tsp_max": max(ages),
            "tsp_min": min(ages),
            "tsp_std": math.sqrt(var),
            # Share of the field within 5 minutes of a stop: cold tyres in traffic.
            "tsp_frac_fresh": sum(1 for a in ages if a <= 5.0) / n,
        })
    return pl.DataFrame(rows)


def restop_pressure_series(
    stops: dict[str, list[float]], grid: list[float], *, window_s: float = 1800.0,
    scale_s: float = 600.0,
) -> pl.DataFrame:
    """Field-level weight of abnormally short stop intervals.

    Each consecutive pair of stops contributes ``exp(-gap / scale_s)``, so a
    quick return to the pits counts heavily and a normal fuel-window stop counts
    for almost nothing. Only pairs whose second stop falls in the trailing
    ``window_s`` contribute, which keeps the series a description of the current
    race state rather than a cumulative total that only ever grows.

    The legacy version summed over the whole race to date and then took a
    derivative to recover a rate; a trailing window measures the same thing
    without the differencing.
    """
    rows = []
    for t in grid:
        total, n_recent = 0.0, 0
        for ss in stops.values():
            prior = [s for s in ss if s <= t]
            for a, b in zip(prior, prior[1:]):
                if t - b <= window_s:
                    total += math.exp(-(b - a) / scale_s)
                    n_recent += 1
        rows.append({"t": t, "restop_pressure": total, "recent_stops": n_recent})
    return pl.DataFrame(rows)
