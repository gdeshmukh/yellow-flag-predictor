import pytest

from yellow_flag_predictor.tracks import Circuit, get


def test_daytona_sectors_close_on_the_lap_distance():
    """The registry's whole purpose: geometry that does not sum is a typo."""
    get("daytona").validate()


def test_trap_sector_implies_the_vendor_speed_constant():
    """``SPI`` in the 2020 export is exactly ``(900/11) / S12``.

    The 120 ft trap sector is where that constant comes from, which is why the
    trap sector and the speed column are the same measurement twice.
    """
    d = get("daytona")
    assert d.trap_length_ft() == 120.0
    assert d.trap_length_ft() * 3600 / 5280 == pytest.approx(900 / 11)


def test_bad_geometry_is_rejected():
    bogus = Circuit(key="x", name="x", lap_miles=3.56, sector_lengths_ft=(100.0, 200.0))
    with pytest.raises(ValueError, match="sectors sum to"):
        bogus.validate()


def test_unknown_circuit_names_the_registered_ones():
    with pytest.raises(KeyError, match="registered"):
        get("nowhere")
