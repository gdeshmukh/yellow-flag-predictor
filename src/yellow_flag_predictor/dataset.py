"""Building the modelling dataset from a source manifest.

The manifest (``reference/sources.csv``) is the only record of where data came
from; the data itself is never committed. Rebuilding from the manifest is the
reproducibility guarantee, which is what the legacy project lost when 83 of its
88 referenced intermediate files disappeared.
"""

from __future__ import annotations

import csv
from pathlib import Path

import polars as pl

from .features.build import build_feature_matrix
from .labels import derive_cautions, load_annotations, merge_cautions
from .normalize import read_vendor_csv
from .risk import DEFAULT_PROFILE, RiskProfile, congregated_risk
from .features.pit import car_activity, pit_stops
from .tracks import get as get_circuit


def _resolve(raw: str, manifest_path: Path) -> Path:
    """Locate a manifest entry.

    Tried in order: as given (absolute, or relative to the working directory),
    then relative to the manifest itself. Paths are kept relative in the
    manifest so a checkout is portable between machines.
    """
    candidates = [Path(raw), manifest_path.parent / raw]
    for c in candidates:
        if c.exists():
            return c.resolve()
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"manifest entry {raw!r} not found; tried:\n  {tried}")


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        return [r for r in csv.DictReader(fh) if r.get("path", "").strip()]


def build(
    manifest_path: str | Path,
    annotations_path: str | Path | None = None,
    *,
    step_s: float = 20.0,
    horizon_s: float = 300.0,
    profile: RiskProfile = DEFAULT_PROFILE,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return ``(features, sessions, cautions)`` across every manifest entry.

    Sessions whose export contains no yellow flags still contribute rows: they
    are genuine negative evidence about how a race runs without a caution, and
    dropping them would bias the base rate upward.
    """
    manifest_path = Path(manifest_path)
    annotations = load_annotations(annotations_path) if annotations_path else None

    feats, sess_all, caut_all = [], [], []
    for entry in read_manifest(manifest_path):
        src = _resolve(entry["path"], manifest_path)
        circuit = get_circuit(entry["circuit"])
        laps, sessions = read_vendor_csv(
            src, circuit=entry["circuit"], year=int(entry["year"]),
            event=entry["event"], kind=entry.get("kind", "race"),
        )
        cautions = derive_cautions(laps)
        if annotations is not None and annotations.height:
            cautions = merge_cautions(cautions, annotations)

        for row in sessions.iter_rows(named=True):
            sid, dur = row["session_id"], row["duration_s"]
            if row["n_sectors"] != circuit.n_sectors:
                raise ValueError(
                    f"{sid}: export has {row['n_sectors']} sectors but circuit "
                    f"{circuit.key!r} is registered with {circuit.n_sectors}"
                )
            fm = build_feature_matrix(
                laps, cautions, sid, dur, circuit, step_s=step_s, horizon_s=horizon_s
            )
            grid = fm["t"].to_list()
            slaps = laps.filter(pl.col("session_id") == sid)
            risk = congregated_risk(pit_stops(slaps), grid, profile=profile,
                                    activity=car_activity(slaps))
            feats.append(fm.join(risk, on="t", how="left"))

        sess_all.append(sessions)
        caut_all.append(cautions)

    features = pl.concat(feats, how="vertical_relaxed") if feats else pl.DataFrame()
    sessions = pl.concat(sess_all, how="vertical_relaxed") if sess_all else pl.DataFrame()
    cautions = pl.concat(caut_all, how="vertical_relaxed") if caut_all else pl.DataFrame()
    return features, sessions, cautions
