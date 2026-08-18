import pytest

from yellow_flag_predictor.timeparse import (
    ClockParseError,
    parse_clock,
    parse_session_series,
    split_sessions,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0.430", 0.430),
        ("52.755", 52.755),          # sub-minute lap: no colon at all
        ("1:37.228", 97.228),
        ("23:45:12.345", 85512.345),
        ("24:01:17.937", 86477.937),  # hour field past 24
        ("26:00:00.000", 93600.0),
    ],
)
def test_parse_clock(text, expected):
    assert parse_clock(text) == pytest.approx(expected)


def test_rejects_garbage():
    for bad in ("", "  ", "abc", "1:2:3:4.5"):
        with pytest.raises(ClockParseError):
            parse_clock(bad)


def test_two_digit_wrapped_minute():
    """The input that crashed the legacy cleaner.

    Its rollover hack concatenated a hard-coded ``'0'`` to zero-pad a
    single-digit minute, so any race whose classified tail ran past 9m59s after
    the 24 h mark raised ValueError.
    """
    assert parse_clock("10:15.123") == pytest.approx(615.123)


def test_wrap_is_detected_by_day_sized_jump():
    got = parse_session_series(["23:59:58.895", "24:01:17.937", "1:28.075", "2:12.968"])
    assert got == pytest.approx([86398.895, 86477.937, 86488.075, 86532.968])
    assert all(b > a for a, b in zip(got, got[1:]))


def test_session_restart_is_not_treated_as_a_wrap():
    """A practice-to-race reset must not be lifted a day into the future."""
    got = parse_session_series(["1:00:00.000", "1:30:00.000", "0:52.700", "1:45.400"])
    assert got[2] == pytest.approx(52.7)
    assert got[3] == pytest.approx(105.4)


def test_split_sessions_cuts_on_restart():
    spans = split_sessions([1200.0, 3600.0, 5400.0, 52.7, 110.4, 165.9], gap=10_000.0)
    assert spans == [(0, 3), (3, 6)]


def test_split_sessions_tolerates_out_of_order_crossings():
    assert split_sessions([1000.0, 1030.0, 1005.0, 1060.0], gap=10_000.0) == [(0, 4)]
