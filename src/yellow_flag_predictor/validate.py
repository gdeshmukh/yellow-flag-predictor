"""Does the risk score carry signal?

With six cautions in the reference race, no fitted model is defensible. What
*is* defensible is testing the score's premise directly, against a null built
from the same race, and reporting the result with an honest p-value.

Every test here compares an observed statistic to a permutation null drawn from
green-flag moments in the same session. That controls for everything a race
does over its own length -- fuel windows, attrition, darkness -- without
assuming any of it. With this few events the permutation test is not merely
better than a parametric one, it is the only one whose assumptions hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .features.pit import active_cars, car_activity, pit_stops
from .risk import DEFAULT_PROFILE, RiskProfile


def stint_ages_at(
    laps: pl.DataFrame, times: list[float], grace_s: float = 600.0
) -> list[np.ndarray]:
    """Minutes since last pit stop for every running car, at each given time."""
    stops = pit_stops(laps)
    activity = car_activity(laps)
    out = []
    for t in times:
        ages = []
        for car in active_cars(activity, t, grace_s):
            prior = [s for s in stops.get(car, []) if s <= t]
            ages.append((t - (max(prior) if prior else 0.0)) / 60.0)
        out.append(np.array(ages))
    return out


def green_times(
    cautions: pl.DataFrame, duration_s: float, *, step_s: float = 20.0,
    buffer_s: float = 300.0, warmup_s: float = 600.0,
) -> np.ndarray:
    """Grid times under green, excluding the run-up to any caution.

    The buffer matters: sampling a null from the two minutes before a caution
    would put the very state being tested into the null, and would understate
    any real effect.
    """
    windows = list(zip(cautions["start_s"].to_list(), cautions["end_s"].to_list()))
    grid = np.arange(warmup_s, duration_s, step_s)
    keep = []
    for t in grid:
        if any(a - buffer_s <= t <= b + buffer_s for a, b in windows):
            continue
        keep.append(t)
    return np.array(keep)


@dataclass
class PermutationResult:
    statistic: str
    observed: float
    null_mean: float
    null_std: float
    p_value: float
    n_events: int
    n_null: int

    def as_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "observed": round(self.observed, 5),
            "null_mean": round(self.null_mean, 5),
            "null_std": round(self.null_std, 5),
            "z": round((self.observed - self.null_mean) / self.null_std, 3)
            if self.null_std > 0 else None,
            "p_value": round(self.p_value, 5),
            "n_events": self.n_events,
            "n_null_draws": self.n_null,
        }


def _permutation_p(observed: float, null: np.ndarray) -> float:
    """One-sided p with the observed value included in the null.

    Including it bounds p away from zero, which is the honest floor when the
    event count is small: six events cannot support a p below ~1/(draws+1).
    """
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def risk_elevation(
    laps: pl.DataFrame,
    cautions: pl.DataFrame,
    duration_s: float,
    *,
    profile: RiskProfile = DEFAULT_PROFILE,
    lead_s: float = 120.0,
    n_draws: int = 10_000,
    seed: int = 0,
) -> PermutationResult:
    """Is the congregated risk score higher just before a caution than at a
    random green moment?

    The score is evaluated ``lead_s`` before onset, so it measures anticipation
    rather than observation. The null draws sets of the same size from green
    moments, which is what makes the comparison fair with six events.
    """
    rng = np.random.default_rng(seed)
    onsets = [t - lead_s for t in cautions["start_s"].to_list() if t - lead_s > 0]
    nulls = green_times(cautions, duration_s)

    obs_ages = stint_ages_at(laps, onsets)
    observed = float(np.mean([profile.per_car(a).mean() for a in obs_ages if len(a)]))

    null_ages = stint_ages_at(laps, [float(t) for t in nulls])
    null_scores = np.array([profile.per_car(a).mean() for a in null_ages if len(a)])

    k = len(obs_ages)
    draws = np.array([rng.choice(null_scores, size=k, replace=False).mean()
                      for _ in range(n_draws)])
    return PermutationResult(
        statistic=f"mean congregated risk {lead_s:.0f}s before onset",
        observed=observed, null_mean=float(draws.mean()), null_std=float(draws.std()),
        p_value=_permutation_p(observed, draws), n_events=k, n_null=n_draws,
    )


def feature_elevation(
    features: pl.DataFrame, column: str, *, n_draws: int = 10_000, seed: int = 0
) -> PermutationResult | None:
    """Same test for any feature column, using the labelled matrix.

    Positive bins are those whose horizon contains a caution onset; the null is
    drawn from at-risk bins that do not.
    """
    d = features.filter(pl.col("at_risk")).drop_nulls(subset=[column])
    pos = d.filter(pl.col("y") == 1)[column].to_numpy()
    neg = d.filter(pl.col("y") == 0)[column].to_numpy()
    if len(pos) == 0 or len(neg) == 0:
        return None
    rng = np.random.default_rng(seed)
    observed = float(pos.mean())
    draws = np.array([rng.choice(neg, size=len(pos), replace=False).mean()
                      for _ in range(n_draws)])
    return PermutationResult(
        statistic=f"mean {column} in positive bins",
        observed=observed, null_mean=float(draws.mean()), null_std=float(draws.std()),
        p_value=_permutation_p(observed, draws), n_events=len(pos), n_null=n_draws,
    )


def empirical_stint_profile(
    laps: pl.DataFrame, cautions: pl.DataFrame, duration_s: float,
    *, lead_s: float = 120.0, bins: int = 24, max_min: float = 60.0,
) -> pl.DataFrame:
    """Where in a stint cars actually are when a caution starts.

    This is the direct test of the legacy profile's shape. Its two modes -- at
    0 and 38.13 minutes since a stop -- were hand-entered with no recorded
    provenance. The ratio column is what matters: above 1 means cars at that
    stint age are over-represented at caution onset relative to green running,
    which is the claim the risk score encodes.
    """
    onsets = [t - lead_s for t in cautions["start_s"].to_list() if t - lead_s > 0]
    obs = np.concatenate(stint_ages_at(laps, onsets)) if onsets else np.array([])
    nulls = green_times(cautions, duration_s)
    base = np.concatenate(stint_ages_at(laps, [float(t) for t in nulls]))

    edges = np.linspace(0, max_min, bins + 1)
    o, _ = np.histogram(obs, bins=edges)
    b, _ = np.histogram(base, bins=edges)
    o_frac = o / o.sum() if o.sum() else o.astype(float)
    b_frac = b / b.sum() if b.sum() else b.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(b_frac > 0, o_frac / b_frac, np.nan)
    return pl.DataFrame({
        "stint_age_min_low": edges[:-1],
        "stint_age_min_high": edges[1:],
        "n_at_caution": o,
        "frac_at_caution": o_frac,
        "frac_green": b_frac,
        "ratio": ratio,
    })


def profile_from_ratio(
    table: pl.DataFrame, *, min_count: int = 5, clip: float = 3.0
) -> "RiskProfile":
    """Build a risk profile from the observed over-representation ratio.

    The legacy profile asserted two modes, at 0 and 38.13 minutes. This derives
    the shape instead: each bin with enough observations becomes a mode whose
    weight is its ratio, so the profile says what the data says rather than what
    was assumed. Bins below ``min_count`` are dropped -- a ratio computed from
    two observations is noise, and at this event count it would otherwise
    dominate.
    """
    from .risk import RiskMode, RiskProfile

    modes = []
    for row in table.iter_rows(named=True):
        r = row["ratio"]
        if row["n_at_caution"] < min_count or r is None or not np.isfinite(r) or r <= 1.0:
            continue
        centre = (row["stint_age_min_low"] + row["stint_age_min_high"]) / 2
        half_width = (row["stint_age_min_high"] - row["stint_age_min_low"]) / 2
        modes.append(RiskMode(center_min=centre, width_min=max(half_width, 1.0),
                              weight=min(r, clip)))
    if not modes:
        raise ValueError("no bin exceeded the baseline; nothing to fit")
    peak = max(m.weight for m in modes)
    return RiskProfile(modes=tuple(
        RiskMode(m.center_min, m.width_min, m.weight / peak) for m in modes
    ))


def loo_profile_test(
    laps: pl.DataFrame,
    cautions: pl.DataFrame,
    duration_s: float,
    *,
    lead_s: float = 120.0,
    bins: int = 12,
    max_min: float = 60.0,
    n_draws: int = 10_000,
    seed: int = 0,
) -> dict:
    """Leave-one-caution-out test of a data-fitted stint profile.

    A profile fitted on all six cautions and then scored against those same six
    is circular: the modes are placed *where the cautions are*, so elevation is
    guaranteed and means nothing. Here each caution is held out, the profile is
    fitted on the rest, and the held-out event is scored against a green-flag
    null. Only this number is evidence.

    With six events the test has very little power -- it can fail to detect a
    real effect easily -- so a null result here is not proof of absence. A
    positive result would be worth acting on; a negative one means *not yet
    known*, and the answer is more races.
    """
    rng = np.random.default_rng(seed)
    nulls = green_times(cautions, duration_s)
    rows = cautions.sort("start_s").to_dicts()

    held_scores, null_dists = [], []
    for i in range(len(rows)):
        train = pl.DataFrame([r for j, r in enumerate(rows) if j != i],
                             schema=cautions.schema)
        held = rows[i]
        try:
            prof = profile_from_ratio(
                empirical_stint_profile(laps, train, duration_s, lead_s=lead_s,
                                        bins=bins, max_min=max_min)
            )
        except ValueError:
            continue

        t = held["start_s"] - lead_s
        if t <= 0:
            continue
        ages = stint_ages_at(laps, [t])[0]
        if not len(ages):
            continue
        held_scores.append(float(prof.per_car(ages).mean()))

        null_ages = stint_ages_at(laps, [float(x) for x in nulls])
        null_dists.append(np.array([prof.per_car(a).mean() for a in null_ages if len(a)]))

    if not held_scores:
        return {"error": "no caution could be held out"}

    observed = float(np.mean(held_scores))
    draws = np.array([
        float(np.mean([rng.choice(nd) for nd in null_dists])) for _ in range(n_draws)
    ])
    return {
        "statistic": "mean out-of-sample risk under a leave-one-out fitted profile",
        "observed": round(observed, 5),
        "null_mean": round(float(draws.mean()), 5),
        "null_std": round(float(draws.std()), 5),
        "z": round((observed - draws.mean()) / draws.std(), 3) if draws.std() > 0 else None,
        "p_value": round(_permutation_p(observed, draws), 5),
        "n_events": len(held_scores),
        "note": "six events: low power. A null result means not yet known, not no effect.",
    }


def feature_elevation_blocked(
    features: pl.DataFrame,
    cautions: pl.DataFrame,
    column: str,
    *,
    horizon_s: float = 300.0,
    n_draws: int = 20_000,
    seed: int = 0,
) -> PermutationResult | None:
    """Event-level permutation test. Use this one, not :func:`feature_elevation`.

    Bin-level permutation treats the ~15 positive bins belonging to one caution
    as 15 independent observations. They are not: consecutive bins are twenty
    seconds apart and differ almost not at all, so the bin-level test reports a
    sample size it does not have and returns p-values that are far too small.

    Here each caution contributes one number -- the mean of ``column`` over its
    own run-up -- and the null draws contiguous green blocks of the same length.
    The effective sample size is the number of cautions, which is the honest
    figure and is why six is such a hard constraint.
    """
    d = features.filter(pl.col("at_risk")).drop_nulls(subset=[column])
    if d.height == 0:
        return None
    t = d["t"].to_numpy()
    v = d[column].to_numpy()
    starts = cautions["start_s"].to_list()

    per_event = []
    for s in starts:
        m = (t > s - horizon_s) & (t <= s)
        if m.sum():
            per_event.append(float(v[m].mean()))
    if not per_event:
        return None

    neg = d.filter(pl.col("y") == 0)
    tn, vn = neg["t"].to_numpy(), neg[column].to_numpy()
    width = horizon_s
    block_means = []
    for anchor in tn:
        m = (tn > anchor - width) & (tn <= anchor)
        if m.sum():
            block_means.append(float(vn[m].mean()))
    block_means = np.array(block_means)
    if len(block_means) < len(per_event):
        return None

    rng = np.random.default_rng(seed)
    observed = float(np.mean(per_event))
    draws = np.array([
        rng.choice(block_means, size=len(per_event), replace=False).mean()
        for _ in range(n_draws)
    ])
    return PermutationResult(
        statistic=f"event-mean {column} over the {horizon_s:.0f}s before onset",
        observed=observed, null_mean=float(draws.mean()), null_std=float(draws.std()),
        p_value=_permutation_p(observed, draws), n_events=len(per_event), n_null=n_draws,
    )
