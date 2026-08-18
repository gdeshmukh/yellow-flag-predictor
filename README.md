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
