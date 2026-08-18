# yellow-flag-predictor

Predicting full-course cautions in IMSA endurance races from timing data.

The question: at any moment of a race, how likely is a caution in the next few
minutes? A strategist who knows that pits differently.

This is a rebuild. Two earlier attempts (2021) got as far as feature engineering
and a hand-tuned live score; neither ever tested whether the score worked. See
[FINDINGS.md](FINDINGS.md) for what they established and what they got wrong.

## State

Working: ingestion and normalisation of vendor timing exports, caution
extraction, the feature set, the risk score, and the validation harness.

Not started: multi-race ingestion (the blocker), live scoring.

**The headline result is negative.** On the one race currently ingested, the
risk score is *lower* before cautions than at random green-flag moments
(z = −1.25). No feature reaches significance. That is not a surprise and not a
dead end — with six caution events the test has almost no power. The point of
the harness is that this answer will change as races are added, and it will
change honestly.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'

.venv/bin/yellow-flag-predictor build       # normalise sources -> feature matrix
.venv/bin/yellow-flag-predictor validate    # test features for signal
.venv/bin/yellow-flag-predictor diagnose    # check for collinear features
.venv/bin/yellow-flag-predictor train       # fit the linear model
.venv/bin/yellow-flag-predictor evaluate    # leave-one-race-out (needs >= 2 races)
```

Sources are listed in `reference/sources.csv`. Raw data is never committed; the
manifest plus the CLI reproduces every artifact.

## How it fits together

```
reference/sources.csv     what to ingest, and from where
reference/cautions.csv    hand-annotated caution causes (not derivable)
        |
   normalize.py           vendor CSV -> canonical laps + sessions
        |                 handles 0/1/2-field clocks, 24h wraps, session splits
   labels.py              flag transitions -> caution events -> hazard target
        |
   features/              traffic density, pit cycle, lap dispersion
   risk.py                the congregated risk score
        |
   validate.py            does any of it carry signal?
   model.py               linear / logistic hazard, leave-one-race-out
```

## The target

A discrete-time hazard: *does a caution begin in the next 5 minutes?*, on a
20-second grid. Bins already under caution are marked `at_risk = False` and
dropped — a caution cannot begin while one is running, and scoring those bins
would pad the negative class with rows whose answer is structurally no.

Five minutes is roughly the window in which a strategist can still act. The
2021 version used 60 seconds, which is too short to be useful and starves an
already tiny positive class.

## Adding a race

1. Put the export somewhere outside the repo, add a row to
   `reference/sources.csv`.
2. If the circuit is new, add it to `tracks.py` with sector lengths in feet.
   `Circuit.validate()` refuses geometry that does not sum to the lap distance.
3. `build`, then `validate`.

Every added race increases the effective sample size by its caution count —
which is the only thing that will move this project forward.

## Testing

```bash
.venv/bin/python -m pytest tests -q
```

Tests that need the reference export skip cleanly without it. The invariants
worth knowing about: sector times sum to lap time for 99.8% of clean green
laps, caution onset is bracketed to under 30 seconds, positive labels strictly
precede the caution they predict, and the event-level permutation test is
always more conservative than the bin-level one.
