# Findings

What the two 2021 attempts established, what they got wrong, and what carried over.

## The two predecessors

**Daytona24** (Jan 2021) — 2020 Rolex 24, 27,526 laps. Cleaning and sector
bookkeeping only. Computes caution times into a local variable and never saves
them. Not my code; the repo was transferred to my account and all three commits
are Adithya Shanmugam's. No licence, so it is all-rights-reserved by him — this
rebuild is clean-room and shares no code with it.

**NortheastGrandPrix** (Jul 2021) — the real attempt. Four features, a
hand-tuned risk score, a Selenium scraper against Al Kamel live timing, and a
live matplotlib readout. It ran: `scores.json` holds four real points from a
live session.

Both are archived. Nothing was ported; the ideas were reimplemented.

## Corrections to what I thought this project was

**There was never a regression.** Grepping both repos for sklearn, statsmodels,
`LinearRegression`, `train_test_split`, `curve_fit`, `.fit()`, xgboost or torch
returns nothing. The only statistics import anywhere is `scipy.stats.norm`. No
train/test split, no cross-validation, no metric, no baseline. The risk-score
constants were hand-entered. Fitting a model was the intended next step, never
a completed one.

**The label supply is seven caution periods.** Daytona 2020 has 6. The three
committed Lime Rock CSVs have 0, 1 and 0. That is everything, across both repos.

**The notebooks and their committed data are from different races.** The
`src/2019/` notebooks were run against a 6-hour DPi race (stored output: 6,301
rows, 11 sectors, clock to 06:01:59) while `data/race_data/2019.csv` is 2h40m
Lime Rock, GTLM+GTD, 8 sectors. `FixData2019.ipynb`'s stored output is the
resulting crash: `ValueError: time data '52.755' does not match format`, because
sub-minute Lime Rock laps have no colon and the parser was written for Daytona.
`MAXTIME` is variously 43200, 21719 and 9749 across notebooks in the same
year-folder.

**83 of 88 referenced data files no longer exist.** Only the raw exports
survive. Every intermediate, and all weather data, is gone.

## Ideas that carried over

| Idea | Verdict |
|---|---|
| Sector-cumulative reconstruction of track position | Sound. Rebuilt. |
| Traffic density as cars per 1000 ft | Sound — length-normalisation is the right form. Rebuilt. |
| Time since last pit stop as a risk driver | Sound as a feature. The *shape* assumed for it was wrong (below). |
| Unscheduled-restop pressure ("sus score") | Best idea in either repo. Rebuilt as a smooth decay over a trailing window. Currently the strongest-ranked feature. |
| Rolling lap-time dispersion | Sound. Rebuilt, with caution laps excluded and per-class normalisation. |
| Weather covariates | Sound, but the data is lost. Not yet rebuilt. |
| Hand-annotated caution causes | Irreplaceable. Salvaged to `reference/cautions.csv`. |

The five Daytona annotations (`cars = [38, 19, 74, 19, 47]` at
`times = [27940, 36360, 65440, 68460, 70540]`) each match a caution derived
independently from flag transitions to within 15 seconds, which is what makes
them trustworthy. The rest could not be attributed to a session and are parked
in `reference/legacy_annotations_unresolved.csv`.

## Bugs found, and the guards that now prevent them

**`SPI` was not a feature.** In the Daytona export `SPI` is exactly
`(900/11) / S12` — zero residual across all 26,285 rows with `S12 > 0`. It is
the speed through sector 12, a 120-foot trap loop. Feeding both to a model is
perfect collinearity. Confirmed independently by the legacy `distances` dict,
which lists sector 12 as 120 feet, and by the sector map summing to 18,796.83 ft
against a published 18,796.8 ft lap.
*Guard:* `tracks.py` records the trap sector; `diagnose` reports any feature
pair above |r| = 0.99. It immediately caught a fresh instance — `tsp_max`
correlated with `elapsed_frac` at exactly 1.000000, because retired cars were
never dropped from the field, so "time since last pit" became a proxy for
elapsed race time.

**The density artifact undercounted, and I initially had this backwards.**
Because intervals are one sector long and anchored to the previous crossing, a
multi-hour garage stint leaves a *hole*, not an occupancy. Car 4 appears in 0 of
the 1,409 timesteps of its 7.8-hour stop. The reconstruction saw a mean of 33.6
cars against ~35.9 running — 6.6% of race car-time missing, concentrated in
exactly the pit and attrition sequences a caution model cares about.
*Guard:* `coverage` is emitted as a column, so a density computed from part of
the field is distinguishable from the same number computed from all of it.

**The 24-hour rollover worked by luck.** The legacy fix concatenated a
hard-coded `'0'` to zero-pad a single-digit minute; the 2020 tail stopped at
`2:12.968`, about eight minutes short of the two-digit boundary that raises
`ValueError`.
*Guard:* clock parsing is field-count agnostic, and a wrap is identified by the
backwards jump landing within seconds of a whole day — which also distinguishes
it from a practice-to-race restart.

**Zero sectors are dropped timing loops.** 1,443 of 1,483 green non-pit laps
that fail to close have a zero sector. Excluding them, 99.83% close within 0.5s.
Every cumulative position after a zero is wrong.
*Guard:* such laps are excluded from position reconstruction and asserted in
tests.

## The risk score, tested

The 2021 score was `sum over cars of N(x; 0, 11.07) + N(x; 38.13, 4.65)`, where
`x` is minutes since that car's last stop. Two modes: cold tyres rejoining, and
a fuel window around 38 minutes.

Three problems with the form, all fixed in `risk.py`: a density is not a
probability, the sum scales with car count so no threshold transfers between
races, and the relative height of the two modes was set by their standard
deviations rather than by any claim about racing.

The bigger problem is the content. Measuring where cars actually are in their
stints when a caution starts, against green-flag running in the same race:

| stint age (min) | over-representation |
|---|---|
| 0–5 | **0.57** |
| 10–15 | 1.22 |
| 15–20 | **1.50** |
| 20–25 | 1.33 |
| 35–40 | **0.79** |

Both hand-picked modes sit where cars are *under*-represented at caution onset.
The observed bump is a broad one around 15–25 minutes into a stint. That
explains the negative result: the score peaks where cautions are least likely.

Fitting the profile from data lifts it to p = 0.041 — but that is in-sample,
fitting modes where the cautions are. Held out one caution at a time, it
collapses to z = 0.45, p = 0.32. There is no out-of-sample evidence yet.

## Feature ranking

Event-level permutation, 6 events, mean over the 5 minutes before onset:

| feature | z | p |
|---|---|---|
| recent_stops | +1.10 | 0.134 |
| restop_pressure | +1.10 | 0.141 |
| time_since_last_caution_s | +0.67 | 0.247 |
| density_mean | +0.48 | 0.404 |
| lap_std_mean | +0.34 | 0.276 |
| **risk_mean** | **−1.25** | 0.901 |

Nothing is significant and nothing can be at this sample size. The ranking is a
prior for what to test first when more races land, not a result.

A note on why this is the right test: the bin-level version of the same test
reports p = 0.0002 for `restop_pressure`. That is wrong. The ~15 positive bins
belonging to one caution are twenty seconds apart and nearly identical, so the
bin-level test claims a sample size it does not have — inflating significance by
roughly 700×. The effective n is the number of cautions.

## What would actually move this forward

More races, and nothing else. Every other question is downstream of the sample
size. Priorities in order:

1. Multi-race ingestion from the IMSA/Al Kamel historical archive. Sector-level
   exports with flag states, as many seasons as are available.
2. Re-run `validate`. With ~50 cautions the ranking above becomes testable.
3. Only then fit a model, leave-one-race-out, against the base rate.
4. Weather, which the 2021 version had and lost.
5. Live scoring — last, and only if something survives step 3.

Two constraints to design around when the data arrives: the DPi→GTP (LMDh)
transition and the GTLM/GTD→GTD Pro reshuffle mean pre-2023 and post-2023 races
may not be exchangeable, and sector counts differ per circuit and era, which is
why sectors are a list column rather than fixed `S01..S13`.
