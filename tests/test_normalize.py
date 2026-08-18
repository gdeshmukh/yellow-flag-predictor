import polars as pl
import pytest

from yellow_flag_predictor.normalize import canonical_flag, sector_sum_residual
from yellow_flag_predictor.schema import Flag


def test_flag_vocabulary_collapses_vendor_spellings():
    for raw in ("Yellow", "FCY", "full course yellow", "SC", "caution"):
        assert canonical_flag(raw) == Flag.YELLOW
    assert canonical_flag("Finish") == Flag.CHECKERED
    assert canonical_flag(None) == Flag.UNKNOWN
    assert canonical_flag("banana") == Flag.UNKNOWN


def test_no_rows_are_lost(daytona):
    """The wrapped tail must survive: splitting on it silently dropped 14 laps."""
    laps, _ = daytona
    assert laps.height == 27526


def test_single_session_and_full_duration(daytona):
    laps, sessions = daytona
    assert sessions.height == 1
    assert sessions["n_sectors"][0] == 13
    assert laps["session_time_s"].max() == pytest.approx(86532.968)


def test_clock_is_ordered_to_within_feed_ties(daytona):
    """Rows are chronological apart from sub-tenth-second scoring ties.

    107 of 27,525 steps go backwards, by at most 99 ms -- two cars scored out
    of order at the line, not disorder. Anything larger would mean the wrap
    repair had mis-lifted a row.
    """
    laps, _ = daytona
    st = laps["session_time_s"].to_list()
    backsteps = [st[i - 1] - st[i] for i in range(1, len(st)) if st[i] < st[i - 1]]
    assert max(backsteps) < 0.5


def test_lap_start_precedes_crossing(daytona):
    laps, _ = daytona
    d = laps.drop_nulls(["lap_start_s", "session_time_s"])
    assert (d["lap_start_s"] <= d["session_time_s"]).all()


def test_clean_laps_have_sectors_summing_to_lap_time(daytona):
    """Green flying laps close to timing resolution.

    Laps that do not close contain stationary time -- a pit box or garage stop
    absorbed into one sector -- and are what the density builder drops.
    """
    laps, _ = daytona
    r = (
        sector_sum_residual(laps)
        .filter((~pl.col("in_pit")) & (pl.col("flag") == "green"))
        # A zero sector is a dropped timing loop; such laps cannot close and
        # are excluded from position reconstruction for the same reason.
        .filter(pl.col("sectors_s").list.min() > 0)
    )
    close = r.filter(pl.col("sector_residual_s").abs() <= 0.5).height
    assert close / r.height > 0.99
