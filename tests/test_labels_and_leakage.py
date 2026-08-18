import polars as pl
import pytest

from yellow_flag_predictor.labels import derive_cautions, make_labels, merge_cautions


def test_derives_the_six_known_cautions(daytona):
    laps, _ = daytona
    c = derive_cautions(laps)
    assert c.height == 6
    assert (c["end_s"] > c["start_s"]).all()


def test_onset_bracket_is_tight(daytona):
    """Onset is bracketed by the last green and first yellow crossing.

    Cars cross seconds apart, so the flag is pinned to roughly ten seconds --
    which is what makes a five-minute horizon meaningful.
    """
    laps, _ = daytona
    c = derive_cautions(laps)
    bracket = (c["start_upper_s"] - c["start_lower_s"]).max()
    assert bracket < 30.0


def test_bins_under_caution_are_not_at_risk(daytona):
    laps, sessions = daytona
    c = derive_cautions(laps)
    lab = make_labels(c, sessions["session_id"][0], sessions["duration_s"][0])
    for row in c.iter_rows(named=True):
        mid = (row["start_s"] + row["end_s"]) / 2
        during = lab.filter((pl.col("t") - mid).abs() < 10)
        assert not during["at_risk"].any()


def test_positive_label_precedes_its_caution(daytona):
    """A positive bin must sit strictly before the onset it predicts.

    This is the leakage guard: if a bin at or after onset were labelled
    positive, the model would be scored on observing a caution rather than
    anticipating one.
    """
    laps, sessions = daytona
    c = derive_cautions(laps)
    horizon = 300.0
    lab = make_labels(c, sessions["session_id"][0], sessions["duration_s"][0],
                      horizon_s=horizon)
    starts = c["start_s"].to_list()
    for t in lab.filter(pl.col("y") == 1)["t"].to_list():
        assert any(t < s <= t + horizon for s in starts)


def test_annotations_attach_by_onset_proximity(daytona):
    """Legacy hand-annotations must land on the cautions they describe."""
    laps, _ = daytona
    c = derive_cautions(laps)
    ann = pl.DataFrame({
        "session_id": ["daytona-2020-race-1"] * 2,
        "start_s": [27940.0, 36360.0],
        "end_s": [28930.0, 37349.0],
        "cause_car": ["38", "19"],
        "cause_kind": ["stopped_car"] * 2,
        "source": ["legacy-annotation"] * 2,
    })
    merged = merge_cautions(c, ann)
    assert merged.height == 6, "annotations must attach, not duplicate"
    assert set(merged.drop_nulls("cause_car")["cause_car"]) == {"38", "19"}


def test_unmatched_annotation_is_kept(daytona):
    """A session whose export has no yellow rows can still have had cautions."""
    laps, _ = daytona
    c = derive_cautions(laps)
    ann = pl.DataFrame({
        "session_id": ["daytona-2020-race-1"],
        "start_s": [50000.0], "end_s": [50600.0],
        "cause_car": ["99"], "cause_kind": ["contact"], "source": ["manual"],
    })
    merged = merge_cautions(c, ann)
    assert merged.height == 7
    assert "annotated" in set(merged["source"])
