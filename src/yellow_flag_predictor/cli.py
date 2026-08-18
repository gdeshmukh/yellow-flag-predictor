"""Command line entry points."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from . import dataset, model
from .features.build import FEATURE_COLUMNS, LIVE_FEATURE_COLUMNS

app = typer.Typer(add_completion=False, help="IMSA caution-risk modelling.")
console = Console()

DEFAULT_MANIFEST = Path("reference/sources.csv")
DEFAULT_ANNOTATIONS = Path("reference/cautions.csv")
DEFAULT_OUT = Path("data/features/features.parquet")


@app.command()
def build(
    manifest: Path = DEFAULT_MANIFEST,
    annotations: Path = DEFAULT_ANNOTATIONS,
    out: Path = DEFAULT_OUT,
    step: float = typer.Option(20.0, help="Grid resolution in seconds."),
    horizon: float = typer.Option(300.0, help="Predict a caution starting within this many seconds."),
) -> None:
    """Normalise every source in the manifest and write the feature matrix."""
    features, sessions, cautions = dataset.build(
        manifest, annotations if annotations.exists() else None,
        step_s=step, horizon_s=horizon,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(out)
    sessions.write_parquet(out.with_name("sessions.parquet"))
    cautions.write_parquet(out.with_name("cautions.parquet"))

    t = Table(title="Sessions")
    for c in ("session_id", "event", "n_sectors", "duration_s", "n_cars"):
        t.add_column(c)
    for r in sessions.iter_rows(named=True):
        t.add_row(r["session_id"], r["event"], str(r["n_sectors"]),
                  f"{r['duration_s']:.0f}", str(r["n_cars"]))
    console.print(t)

    at_risk = features.filter(pl.col("at_risk"))
    console.print(
        f"[bold]{features.height}[/] grid rows, [bold]{at_risk.height}[/] at risk, "
        f"[bold]{at_risk['y'].sum()}[/] positive "
        f"(base rate {at_risk['y'].mean():.4f}), "
        f"[bold]{cautions.height}[/] cautions across "
        f"[bold]{sessions.height}[/] sessions"
    )
    console.print(f"wrote {out}")


@app.command()
def evaluate(
    features: Path = DEFAULT_OUT,
    kind: str = typer.Option("linear", help="'linear' (OLS) or 'logistic'."),
    live_only: bool = typer.Option(False, help="Restrict to live-available features."),
) -> None:
    """Leave-one-race-out evaluation against the base rate."""
    df = pl.read_parquet(features)
    cols = [c for c in (LIVE_FEATURE_COLUMNS if live_only else FEATURE_COLUMNS)
            if c in df.columns]
    console.print_json(json.dumps(model.evaluate(df, kind=kind, columns=cols), indent=2))


@app.command()
def train(
    features: Path = DEFAULT_OUT,
    kind: str = typer.Option("linear"),
    out: Path = Path("data/features/model.json"),
    live_only: bool = typer.Option(
        False, help="Restrict to features a live timing feed can supply."
    ),
) -> None:
    """Fit on all data and write coefficients for the live scorer."""
    df = pl.read_parquet(features)
    cols = [c for c in (LIVE_FEATURE_COLUMNS if live_only else FEATURE_COLUMNS)
            if c in df.columns]
    res = model.fit(df, kind=kind, columns=cols)
    out.parent.mkdir(parents=True, exist_ok=True)
    scaler = res.estimator.named_steps["scale"]
    payload = {
        "kind": kind,
        "columns": res.columns,
        "coefficients": res.coefficients,
        "intercept": res.intercept,
        # Coefficients are fitted on standardised inputs, so the live scorer
        # needs the same centring and scaling to reproduce the fit.
        "standardise": {
            "mean": [float(v) for v in scaler.mean_],
            "scale": [float(v) for v in scaler.scale_],
        },
    }
    out.write_text(json.dumps(payload, indent=2))

    t = Table(title=f"{kind} coefficients (standardised inputs)")
    t.add_column("feature")
    t.add_column("coefficient", justify="right")
    for k, v in sorted(res.coefficients.items(), key=lambda kv: -abs(kv[1])):
        t.add_row(k, f"{v:+.5f}")
    console.print(t)
    console.print(f"intercept {res.intercept:+.5f}")
    console.print(f"wrote {out}")


@app.command()
def validate(
    features: Path = DEFAULT_OUT,
    horizon: float = typer.Option(300.0),
    draws: int = typer.Option(20_000),
) -> None:
    """Test every feature for elevation before cautions, at event level."""
    from . import validate as V

    f = pl.read_parquet(features)
    c = pl.read_parquet(features.with_name("cautions.parquet"))
    cols = [x for x in FEATURE_COLUMNS + ["risk_mean", "risk_max"] if x in f.columns]

    rows = []
    for col in cols:
        r = V.feature_elevation_blocked(f, c, col, horizon_s=horizon, n_draws=draws)
        if r:
            d = r.as_dict()
            d["feature"] = col
            rows.append(d)
    table = pl.DataFrame(rows).sort("z", descending=True)

    t = Table(title=f"elevation before cautions ({rows[0]['n_events']} events, event-level permutation)")
    for c_ in ("feature", "observed", "null mean", "z", "p"):
        t.add_column(c_, justify="right")
    for r in table.iter_rows(named=True):
        z = r["z"] or 0.0
        colour = "green" if z > 1.0 else ("red" if z < -1.0 else "white")
        t.add_row(r["feature"], f"{r['observed']:.4g}", f"{r['null_mean']:.4g}",
                  f"[{colour}]{z:+.2f}[/]", f"{r['p_value']:.4f}")
    console.print(t)
    console.print(
        f"[yellow]{rows[0]['n_events']} events is the effective sample size. "
        "Nothing here can reach significance; the ranking is a prior for what to "
        "test once more races are ingested, not a result.[/]"
    )


@app.command()
def diagnose(
    features: Path = DEFAULT_OUT,
    threshold: float = typer.Option(0.99, help="Report feature pairs above this |r|."),
) -> None:
    """Report near-duplicate features before they corrupt a fit."""
    df = pl.read_parquet(features)
    dupes = model.collinearity(df, threshold=threshold)
    if dupes.height == 0:
        console.print(f"[green]no feature pair exceeds |r| = {threshold}[/]")
        return
    t = Table(title=f"collinear feature pairs (|r| >= {threshold})")
    for c in ("feature_a", "feature_b", "r"):
        t.add_column(c)
    for r in dupes.iter_rows(named=True):
        t.add_row(r["feature_a"], r["feature_b"], f"{r['r']:+.6f}")
    console.print(t)
    console.print("[yellow]drop one of each pair: their coefficients are not identifiable.[/]")


@app.command()
def features_list() -> None:
    """Show the columns offered to the model."""
    for c in FEATURE_COLUMNS:
        console.print(f"  {c}")


if __name__ == "__main__":
    app()
