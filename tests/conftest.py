from pathlib import Path

import pytest

#: The 2020 Rolex 24 export the project was originally built on. It lives
#: outside the repo because raw data is never committed, so every test that
#: needs it skips cleanly when it is absent.
LEGACY_DAYTONA = Path("/home/gaurav/projects/Daytona24/data/2020_jan.csv")


@pytest.fixture(scope="session")
def daytona_csv() -> Path:
    if not LEGACY_DAYTONA.exists():
        pytest.skip(f"reference export not present at {LEGACY_DAYTONA}")
    return LEGACY_DAYTONA


@pytest.fixture(scope="session")
def daytona(daytona_csv):
    from yellow_flag_predictor.normalize import read_vendor_csv

    return read_vendor_csv(daytona_csv, circuit="daytona", year=2020, event="Rolex 24")
