# Pluvium

Hour-by-hour rain forecasts for every town and village in the Caribbean, three days ahead. For each place Pluvium
answers the questions people actually ask:

* will it rain, hour by hour, and is that drizzle, showers, steady rain, heavy rain or a thunderstorm?
* when will it most likely start, in local time and with a window, and when will it be heaviest?
* how long will it last, and how much will fall (median and a bad case)?
* is lightning active nearby right now, and could it flood?

Pluvium runs on the NatureV1 / iHELIX engine (geometric attention on the sphere). **Pluvium-500** is the full-size
model: 57.9M parameters ("500" is its name, not its parameter count). It is trained on one GPU by a single resumable
program.

> Research model, not an official forecast. Follow your national meteorological service.

## Install

```bash
pip install pluvium
```

## Use a trained model

```bash
pluvium forecast --country JM            # every place; tables in ./pluvium-forecast, a report on screen
pluvium place "Port Antonio" --country JM  # hour by hour for one place
pluvium learn                              # check past forecasts against airport reports; update the corrections
pluvium info                               # what the published model is and how it scored
```

`pluvium forecast` writes:

| File | Contents |
|---|---|
| `towns.csv` | One row per place: chance of rain now and in 6/12/24/48/72 h, heavy rain, thunder, main weather, likely start and window (local time), heaviest hour, 24 h totals (median, 90th percentile, chance of 10/25/50/100 mm), wettest 3 h, rain over the river catchment, flood risk and reason, lightning now, cause, and a one-sentence summary |
| `parishes.csv` | One row per parish or province, with a headline |
| `towns.geojson` | The same as points, for maps |
| `hourly.npz` | Hourly chance of rain, of each intensity and of thunder, and expected rain, per place |
| `forecast.json`, `report.txt` | Metadata and a readable digest |

From Python:

```python
from pluvium.cli import fetch_model
from pluvium.forecast import Forecaster, place_timeline

fc = Forecaster(fetch_model("CorxLabs/Pluvium-500"))
result = fc.run(out_dir="forecast")
print(place_timeline(fc.towns, result["summary"], "Mandeville", country="JM", hours=24))
```

## Train Pluvium-500 on one GPU

`notebooks/rtx6000_pluvium.py` is one copy-paste cell for an NVIDIA RTX PRO 6000 Blackwell (96 GB). Set
`GPU_HOURS` and a Hub token and run it. The program then:

1. builds the Caribbean geography and the town list;
2. stages ERA5, CMORPH and airport reports;
3. computes the climatology;
4. trains stage A on reanalysis;
5. trains stage B on real GFS runs, fetched in the background;
6. calibrates and scores on a held-out year;
7. publishes to the Hub;
8. makes the live forecast.

Everything expensive is mirrored to the Hub, so re-running the cell continues where the last session stopped.

```bash
GPU_HOURS=24 HF_TOKEN=... HF_REPO=you/Pluvium-500 pluvium program
```

## How it works

The full explanation is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), every module's purpose is in
[docs/MODULES.md](docs/MODULES.md), and what can go wrong in operation (and what Pluvium does about it) is in
[docs/FAILURE_MODES.md](docs/FAILURE_MODES.md). In short:

* **Inputs**:
  * the global models' fields: GFS, ECMWF IFS and AIFS live, ERA5 in training;
  * convective indices (K-index, lifted index, mid-level humidity, shear, upper divergence);
  * rain falling now, from the satellite;
  * the last three hours of airport reports;
  * terrain, coasts, sun and season.
* **Engine**: iHELIX blocks on a Caribbean mesh of 6,075 points (about 11 km over land). Each block combines
  geodesic attention with per-head radii from 15 to 500 km, content-addressed index attention, and a recurrence
  along the forecast's 43 frames.
* **Physics in the loop**:
  * mountain rain (Smith–Barstad linear theory, re-solved every hour with learned parameters);
  * sea-breeze convergence over each island;
  * fronts and shear lines with arrival times;
  * the rain falling now, advected along the steering wind for the first 12 hours.
* **Outputs**: per town and hour, an *intensity chain* (dry, light, moderate, heavy) gives every timing and amount
  answer consistently. Heads for thunder, drizzle/rain/showers and the cause complete it.
* **Training** scores the satellite's and the airports' reports through each instrument's learned confusion
  matrix (a differentiable hidden Markov model), so neither instrument's errors are taught as truth.
* **Live**: one run per driving model, mixed. Calibrated on held-out data. Corrected online from what the
  airports report afterwards.
* **Version 3 depth**, in its own subpackages:
  * `atmos/`: CAPE/CIN, column humidity, Saharan air, tropical waves, cyclone rain, sea breezes, diurnal clocks,
    terrain lift;
  * `engine/`: terrain patches, lead modulation, temporal and neighbour attention, experts, uncertainty heads;
  * `training/`: rare-event curriculum, averaged weights, augmentation, distillation;
  * `prob/`: copulas, conformal intervals, extreme-value tails, scores;
  * `guard/`: input repair, freshness, drift, output repair, fallback;
  * `hydro/`: soil wetness, runoff, flood peak timing;
  * `insight/`: attribution, error analysis, case replays;
  * `render/`: maps, bulletins, CAP alerts;
  * `ops/`: hourly service and JSON API.

## Benchmark

`pluvium bench` trains architectures from scratch on identical data with identical budgets. It scores them on a
held-out period against climatology, persistence and the driving model's rain. Results for version 1 against
version 2 are in [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Data and attribution

ERA5 (Copernicus Climate Change Service), CMORPH CDR and ISD (NOAA NCEI), GFS (NOAA NCEP), ECMWF open data
(CC-BY-4.0), GOES-East imagery and lightning (NOAA), METAR (NOAA Aviation Weather Center), GeoNames (CC-BY-4.0),
geoBoundaries (CC-BY-4.0), Terrarium elevation tiles (Mapzen and its sources).

Code: Apache-2.0, CorX Labs.
