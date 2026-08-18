# Reference data

Hand-authored facts that cannot be derived from timing exports. Version-controlled,
unlike everything under `data/`.

## cautions.csv

Caution cause attribution. A timing export records *that* a caution happened — the
flag state per car per lap — but never *why*. Cause must be read off race control
logs, timing-and-scoring notes or race video, once, by hand.

The five Daytona 2020 rows are recovered from hardcoded constants in the legacy
`NortheastGrandPrix` notebooks (`cars = [38, 19, 74, 19, 47]` against
`times = [27940, 36360, 65440, 68460, 70540]`, written to a since-deleted
`20_stopped_car_pit.csv`). Each matches a caution independently derived from flag
transitions in the 2020 Rolex 24 export to within 15 seconds, which is what
justifies trusting them. The sixth Daytona caution (onset ≈ 17006 s) has no
recorded cause.

`cause_kind` vocabulary: `contact`, `stopped_car`, `mechanical`, `debris`,
`weather`, `unknown`.

## legacy_annotations_unresolved.csv

Caution times and cause cars found hardcoded in the legacy notebooks that **cannot
currently be attributed to a session**. The legacy repo's `src/2017`, `src/2018` and
`src/2019` folders are named by year but their notebooks were run against at least
three different event types — `MAXTIME` is variously 43200 s (12 h), 21719 s (6 h)
and 9749 s (2.7 h) — and the committed Lime Rock CSVs are not the data the notebooks
produced their stored output from. These rows are preserved because they represent
manual work that would otherwise be lost, but they must not be used as labels until
the source session is identified.
