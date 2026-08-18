import numpy as np
import polars as pl
import pytest

from yellow_flag_predictor import validate as V
from yellow_flag_predictor.labels import derive_cautions


def test_green_times_exclude_caution_neighbourhoods(daytona):
    laps, sessions = daytona
    c = derive_cautions(laps)
    g = V.green_times(c, float(sessions["duration_s"][0]), buffer_s=300.0)
    for a, b in zip(c["start_s"].to_list(), c["end_s"].to_list()):
        assert not ((g >= a - 300) & (g <= b + 300)).any()


def test_permutation_p_is_bounded_below_by_draw_count():
    """Six events cannot support an arbitrarily small p."""
    null = np.zeros(999)
    assert V._permutation_p(1.0, null) == pytest.approx(1 / 1000)


def test_event_level_test_is_more_conservative_than_bin_level(daytona):
    """The whole point of the blocked test.

    Bins twenty seconds apart within one caution's run-up are near-duplicates,
    so the bin-level test overstates its sample size. The event-level p must be
    larger for a feature the bin-level test calls significant.
    """
    from yellow_flag_predictor.dataset import build

    features, _, cautions = build("reference/sources.csv", "reference/cautions.csv")
    binned = V.feature_elevation(features, "restop_pressure", n_draws=2000)
    blocked = V.feature_elevation_blocked(features, cautions, "restop_pressure",
                                          n_draws=2000)
    assert binned is not None and blocked is not None
    assert blocked.p_value > binned.p_value
    assert blocked.n_events < binned.n_events


def test_fitted_profile_needs_an_above_baseline_bin():
    empty = pl.DataFrame({
        "stint_age_min_low": [0.0], "stint_age_min_high": [5.0],
        "n_at_caution": [1], "frac_at_caution": [0.1],
        "frac_green": [0.5], "ratio": [0.2],
    })
    with pytest.raises(ValueError, match="nothing to fit"):
        V.profile_from_ratio(empty)
