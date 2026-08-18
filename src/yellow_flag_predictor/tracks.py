"""Circuit and timing-loop geometry.

Sector lengths convert a car count per sector into a density, which is the
only form of the traffic feature that compares across sectors of different
length. The legacy project carried this as a bare dict inside a notebook and
reused Daytona's 13-sector map inside the Lime Rock and Watkins Glen notebooks,
silently normalising one circuit's counts by another's geometry.

Lengths are feet, measured between timing loops. ``validate`` is what makes the
registry trustworthy: a map whose sectors do not sum to the published lap
distance is a transcription error, not a rounding difference.
"""

from __future__ import annotations

from dataclasses import dataclass

FEET_PER_MILE = 5280.0


@dataclass(frozen=True)
class Circuit:
    key: str
    name: str
    lap_miles: float
    sector_lengths_ft: tuple[float, ...]
    #: Index of the sector used as a speed trap, if the vendor derives a speed
    #: column from it. At Daytona sector 12 is a 120 ft loop and the exported
    #: ``SPI`` is exactly ``(120 * 3600 / 5280) / S12`` -- the same measurement
    #: twice, so the trap sector and the speed column must never both be fed to
    #: a model as independent inputs.
    trap_sector: int | None = None

    @property
    def n_sectors(self) -> int:
        return len(self.sector_lengths_ft)

    @property
    def lap_feet(self) -> float:
        return self.lap_miles * FEET_PER_MILE

    def validate(self, tolerance: float = 0.02) -> None:
        """Raise if sector lengths disagree with the published lap distance."""
        total = sum(self.sector_lengths_ft)
        if self.lap_feet <= 0:
            raise ValueError(f"{self.key}: lap_miles must be positive")
        err = abs(total - self.lap_feet) / self.lap_feet
        if err > tolerance:
            raise ValueError(
                f"{self.key}: sectors sum to {total:.1f} ft but lap is "
                f"{self.lap_feet:.1f} ft ({err:.1%} off, tolerance {tolerance:.1%})"
            )

    def trap_length_ft(self) -> float | None:
        if self.trap_sector is None:
            return None
        return self.sector_lengths_ft[self.trap_sector - 1]


# Daytona's per-sector lengths are the one piece of geometry the legacy project
# measured; they are carried over verbatim and then checked against the
# published 3.56 mi road course by Circuit.validate.
DAYTONA_SECTORS_2020 = (
    1876.0, 1751.33, 1768.67, 2352.0, 1488.0, 1475.0, 1427.0,
    1428.0, 1457.0, 1484.0, 1283.0, 120.0, 886.83,
)

REGISTRY: dict[str, Circuit] = {
    "daytona": Circuit(
        key="daytona",
        name="Daytona International Speedway (road course)",
        lap_miles=3.56,
        sector_lengths_ft=DAYTONA_SECTORS_2020,
        trap_sector=12,
    ),
}


def get(key: str) -> Circuit:
    try:
        return REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "(none)"
        raise KeyError(f"unknown circuit {key!r}; registered: {known}") from None


def register(circuit: Circuit) -> None:
    """Add a circuit, validating geometry before it can be used."""
    circuit.validate()
    REGISTRY[circuit.key] = circuit
