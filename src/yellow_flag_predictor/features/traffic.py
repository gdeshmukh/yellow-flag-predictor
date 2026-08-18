"""Traffic density from reconstructed track position.

A car's position is reconstructed from its sector split times: sector *k* of a
lap is occupied from ``lap_start + sum(s[:k-1])`` to ``lap_start + sum(s[:k])``.
Counting cars per sector and dividing by sector length gives a density that is
comparable between a 2352 ft sector and a 120 ft one.

Two corrections to the legacy version, both material:

* It normalised counts using Daytona's sector-length map inside notebooks
  operating on other circuits. Geometry now comes from
  :mod:`yellow_flag_predictor.tracks`, keyed by the session's circuit.
* Its counts were a silent undercount. Because an interval spans one sector but
  is anchored to the previous lap's crossing, a pit or garage stint leaves a
  hole rather than a stretch, so a parked car simply vanishes from the field --
  at Daytona 2020 the reconstruction saw a mean of 33.6 cars against ~35.9
  running, losing 6.6% of all car-time, concentrated exactly in the pit and
  attrition sequences a caution model cares about. ``coverage`` is now emitted
  alongside the density so the deficit is visible to the model instead of being
  folded silently into the counts.
"""

from __future__ import annotations

import heapq

import polars as pl

from ..tracks import Circuit


def _intervals(laps: pl.DataFrame, n_sectors: int, max_residual_s: float):
    """Build ``(enter_s, exit_s, car, sector)`` for every sector run.

    Laps whose sector times do not account for the lap time contain stationary
    time -- a pit box or a garage stop absorbed into one sector -- and are
    dropped. Leaving them in models a parked car as moving traffic, which at
    Daytona 2020 put one car in sector S01 for 22 continuous minutes.
    """
    out = []
    for row in laps.iter_rows(named=True):
        sect, lap_t, start = row["sectors_s"], row["lap_time_s"], row["lap_start_s"]
        if sect is None or start is None or lap_t is None:
            continue
        # A zero sector is a dropped timing loop, not a zero-length sector.
        # Every cumulative position after it is wrong, so the lap is unusable
        # for locating the car even though its total may still look plausible.
        if len(sect) != n_sectors or any(s is None or s <= 0 for s in sect):
            continue
        total = sum(sect)
        if total <= 0 or abs(lap_t - total) > max_residual_s:
            continue
        cum = start
        for k, s in enumerate(sect, 1):
            out.append((cum, cum + s, row["car"], k))
            cum += s
    out.sort(key=lambda r: r[0])
    return out


def _sweep(spans, grid):
    """Yield the active span set at each grid time.

    Spans are consumed in enter order and retired from a heap keyed on exit, so
    the whole series costs one pass rather than a scan per grid point.
    """
    spans = sorted(spans, key=lambda r: r[0])
    i, active = 0, []  # active: heap of (exit, payload...)
    for t in grid:
        while i < len(spans) and spans[i][0] <= t:
            s = spans[i]
            heapq.heappush(active, (s[1], s[2:]))
            i += 1
        while active and active[0][0] <= t:
            heapq.heappop(active)
        yield t, active


def concentration_series(
    laps: pl.DataFrame,
    grid: list[float],
    circuit: Circuit,
    *,
    max_residual_s: float = 5.0,
) -> pl.DataFrame:
    """Per-grid-time traffic density summaries.

    Returns the maximum and mean density over sectors (cars per 1000 ft), the
    number of cars located, and ``coverage`` -- located cars as a fraction of
    cars actually circulating -- so that a density computed from two thirds of
    the field is distinguishable from the same density computed from all of it.
    """
    n_sectors = circuit.n_sectors
    lengths = circuit.sector_lengths_ft
    grid = sorted(grid)

    located = {
        t: list(active) for t, active in _sweep(_intervals(laps, n_sectors, max_residual_s), grid)
    }
    running_spans = [
        (r["lap_start_s"], r["session_time_s"], r["car"])
        for r in laps.iter_rows(named=True)
        if r["lap_start_s"] is not None and r["session_time_s"] is not None
    ]
    running = {t: len({p[0] for _, p in active}) for t, active in _sweep(running_spans, grid)}

    rows = []
    for t in grid:
        counts = [0] * (n_sectors + 1)
        seen = set()
        for _exit, (car, k) in located[t]:
            counts[k] += 1
            seen.add(car)
        dens = [counts[k] / lengths[k - 1] * 1000.0 for k in range(1, n_sectors + 1)]
        live = running[t]
        rows.append({
            "t": t,
            "density_max": max(dens),
            "density_mean": sum(dens) / len(dens),
            "cars_located": len(seen),
            "cars_running": live,
            "coverage": (len(seen) / live) if live else None,
        })
    return pl.DataFrame(rows)
