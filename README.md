# Yellow flag predictor

This project is a notebook-first look at pit-cycle risk and full-course yellow
periods in IMSA timing data.

Start with [race_viewer.ipynb](race_viewer.ipynb). Set `SELECTED_RACE` and
`VIEW_AT_HOUR` in the first code cell, then run the notebook. It shows the raw
timing data, a per-car risk table at the selected time, and the aggregate field
risk alongside observed yellow periods.

The notebook discovers timing exports from the Daytona24, NortheastGrandPrix,
Mid-Ohio, LagunaSeca, Glen, and Detroit repositories when they are sibling
directories under `projects`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab race_viewer.ipynb
```

The current score recreates the earlier pit-stint experiment. It is not a
trained probability model.

## IMSA PDF data

The source archive is [Al Kamel's IMSA timing results](https://imsa.results.alkamelcloud.com/).
The dataset covers 118 WeatherTech SportsCar Championship races from 2016
through the available 2026 season. Because the archive layout changes between
seasons, `data/imsa_pdf_catalog.csv` records every discovered final-release PDF
and its exact source URL. Tests, the Roar, and per-hour summary reports are not
part of the parsed dataset; endurance events use the highest published `Hour N`
release.

The repository tracks the parsed CSVs and validation manifests. The 2,641
source PDFs remain available locally under each race's ignored `pdfs/`
directory for manual cross-checks, but are not required after parsing.

```text
data/
  imsa_pdf_catalog.csv
  <year>/
    <track>/
      <race-date>/
        pdfs/
        analysis_by_lap.csv
        time_cards.csv
        entries.csv
        results.csv
        results_by_class.csv
        fastest_laps_by_driver.csv
        drive_time.csv
        drive_time_totals.csv
        pit_stops.csv
        flags_analysis.csv
        sources.csv
        parse_audit.csv
        report_parse_audit.csv
```

The race directory prevents collisions when the championship visits one track
twice in the same season, as it did at Sebring in 2020. All tables parsed from
that race stay together. A file is present only when IMSA published the
corresponding report. Four early races lack an Analysis by Lap PDF. Four races
without an overall Results PDF instead have `results_by_class.csv`; the two 2017
endurance Drive Time reports use the totals-only schema in
`drive_time_totals.csv`.

Values are preserved as printed, including report inconsistencies. Structural
extraction failures block a table, while cross-report disagreements remain as
warnings in the audit CSVs. Weather reports are vector graphs and are retained
as PDFs without guessed CSV values. Starting grids, leader sequences, fastest
lap sequences, and hourly reports are intentionally excluded.

The rebuild tools are maintained separately in the private
`gdeshmukh/imsa-pdf-pipeline` repository. No consolidated lap table is included
on this branch; that derived work is deferred to a separate branch.
