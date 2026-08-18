"""The congregated risk score.

A single scalar per grid time, aggregating every car's stint state into one
number. This is the legacy project's central idea and it is kept, with its
units repaired.

The legacy form was ``sum over cars of (N(x; 0, 11.07) + N(x; 38.13, 4.65))``
where ``x`` is minutes since that car's last stop. The bimodal shape encodes a
real observation -- risk peaks just after a stop, on cold tyres in traffic, and
again near the end of a fuel window when stops bunch -- and that shape is worth
preserving. Summing a probability density across cars is not:

* A density is not a probability. ``N(0; 0, 11.07)`` is 0.036 per minute; the
  value has units of inverse minutes and is not bounded by one.
* The sum scales with the number of cars, so a 38-car Rolex 24 and a 17-car
  Lime Rock race are not on the same scale and cannot share a threshold or a
  fitted coefficient.
* The peak heights are set by the standard deviations, so the narrow fuel-window
  mode (sigma 4.65) is weighted ~2.4x the broad post-stop mode (sigma 11.07) as
  a pure artefact of normalisation rather than a claim about racing.

:func:`congregated_risk` fixes all three: each mode is scaled to unit peak so
its weight is stated explicitly, per-car values land in [0, 1], and the field is
aggregated by mean rather than sum so the score is comparable across races.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass(frozen=True)
class RiskMode:
    """One bump in the stint-risk profile, at unit peak height."""

    center_min: float
    width_min: float
    weight: float = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.weight * np.exp(-0.5 * ((x - self.center_min) / self.width_min) ** 2)


@dataclass(frozen=True)
class RiskProfile:
    """A set of modes over minutes-since-last-pit-stop.

    Defaults carry the legacy centres and widths. Their *weights* are the
    tunable part and default to equal, because the legacy relative weighting was
    an artefact of density normalisation rather than a fitted quantity.
    """

    modes: tuple[RiskMode, ...] = field(
        default_factory=lambda: (
            RiskMode(center_min=0.0, width_min=11.07, weight=1.0),   # cold tyres rejoining
            RiskMode(center_min=38.13, width_min=4.65, weight=1.0),  # fuel window bunching
        )
    )

    def per_car(self, minutes_since_pit: np.ndarray) -> np.ndarray:
        x = np.asarray(minutes_since_pit, dtype=float)
        total = np.zeros_like(x)
        for m in self.modes:
            total = np.maximum(total, m(x))
        return total


DEFAULT_PROFILE = RiskProfile()


def congregated_risk(
    stops: dict[str, list[float]],
    grid: list[float],
    *,
    profile: RiskProfile = DEFAULT_PROFILE,
    session_start: float = 0.0,
    activity: dict[str, tuple[float, float]] | None = None,
    grace_s: float = 600.0,
) -> pl.DataFrame:
    """Field-aggregated stint risk on the grid.

    ``risk_mean`` is the score to use: bounded in [0, 1] and independent of car
    count, so a threshold set on one race means the same thing at another.
    ``risk_sum`` reproduces the legacy scale-dependent quantity and is emitted
    only so the two can be compared directly.
    """
    from .features.pit import active_cars

    rows = []
    for t in grid:
        cars = sorted(active_cars(activity, t, grace_s)) if activity is not None else sorted(stops)
        ages = []
        for car in cars:
            prior = [s for s in stops.get(car, []) if s <= t]
            last = max(prior) if prior else session_start
            ages.append((t - last) / 60.0)
        if not ages:
            rows.append({"t": t, "risk_mean": None, "risk_max": None,
                         "risk_sum": None, "n_cars": 0})
            continue
        per = profile.per_car(np.array(ages))
        rows.append({
            "t": t,
            "risk_mean": float(per.mean()),
            "risk_max": float(per.max()),
            "risk_sum": float(per.sum()),
            "n_cars": len(ages),
        })
    return pl.DataFrame(rows)
