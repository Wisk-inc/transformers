# Pluvium

Rain forecasts for every town and village, in any country between 60 S and 60 N: hour by hour for three days, and
day by day for 14. It started in Jamaica and the Caribbean. For each place Pluvium answers the questions people
actually ask:

* will it rain, hour by hour, and is that drizzle, showers, steady rain, heavy rain or a thunderstorm?
* when will it most likely start, in local time and with a window, and when will it be heaviest?
* how long will it last, and how much will fall (median and a bad case)?
* is lightning active nearby right now, and could it flood?

There are three products in the one package.

Pluvium Blend forecasts each day's rain out to 14 days. It reads every operational system at once: ECMWF ENS, HRES
and AIFS, NOAA GEFS, GFS and NOAA's run of GraphCast, and Pluvium's own network where it runs. It needs no GPU. It
says when a forecast is beyond anything it was trained on, and gives each amount in the place's own terms ("about
once in 6 years").

Pluvium Cyclone forecasts where every tropical cyclone in the North Atlantic and the eastern North Pacific will go,
from the same six systems: a track, a circle that should hold the centre two times in three, and the chance of the
centre passing near each place. It forecasts the track only, and needs no GPU.

Pluvium's network runs on the NatureV1 / iHELIX engine (geometric attention on the sphere) and forecasts hour by
hour. Pluvium-500 is the full-size model, with 57.9M parameters ("500" is its name, not its parameter count). It is
trained on one GPU by a single resumable program.

> Research model, not an official forecast. Follow your national meteorological service and, for hurricanes, the
> National Hurricane Center.

## One command

```bash
pip install pluvium
pluvium now --country JM
```

`pluvium now` does whatever is still missing. It sets the country up (borders, terrain, every town), which takes
seconds. It then needs a trained blend. If one has been published for the country in the Hub repository
(`CorxLabs/Pluvium-500`, folder `blends/`), it is downloaded; otherwise one is trained on the most recent year
whose rain has been published, which takes about two hours, nearly all of it downloads. Last, it reads the newest
run of each of the six systems for the country, a few megabytes, and prints the forecast for the largest towns.
The second call takes about a minute.

```bash
pluvium now --country JM --place "Port Antonio"    # the places whose name contains this
pluvium now --country JM --learn                   # first learn from the days whose rain has been published since
pluvium now --country DO,HT --storms               # two countries, and every tropical cyclone alive now
pluvium blend publish --root ~/pluvium-JM --repo you/your-repo   # share the blend you trained
```

Nobody has to fetch or convert the six systems' output by hand. Pluvium asks each public archive for the rain
fields over the country and nothing else. A system whose newest run is not out yet is read from its previous run,
and one that is missing altogether is left out.

In Python:

```python
from pluvium.quick import forecast, brief
got = forecast("JM")
print(brief(got["result"], got["names"], got["population"]))
```

## The step-by-step commands

`pluvium now` wraps these, and each can be run on its own.

```bash
pluvium setup --country TT --root ~/pluvium-TT     # area, terrain, provinces and every town (Trinidad and Tobago)
pluvium blend train --root ~/pluvium-TT            # learn Pluvium Blend from a year of archived runs (CPU, ~2 h)
pluvium blend forecast --root ~/pluvium-TT         # 14 days of daily rain chances for every town
pluvium blend learn --root ~/pluvium-TT            # update the weights from the days whose rain is now published
pluvium program --root ~/pluvium-TT --country TT   # train Pluvium's own network for the country (GPU)
```

`pluvium blend forecast` writes one row per place and day: the chance of 1, 10, 25 and 50 mm; the median, 90th
percentile and expected rain; how many raw systems call for 50 mm; an alert tier (watch, warning, extreme); whether
the forecast is beyond the training record; and how rare the expected amount is at that place.

`--country` takes ISO codes, one or several (`DO,HT` for Hispaniola), and `--box south,north,west,east` sets the
extent by hand; large countries should be split into regions. Bulletins come in each country's languages: English,
Spanish or French, and Jamaican Patois for Jamaica. Live "rain now" comes from GOES-East or GOES-West wherever they
see (the Americas and the Pacific). Elsewhere Pluvium forecasts without it, which its network was trained to do.
The set-up has been run from live data for Jamaica, Trinidad and Tobago, Sri Lanka (eastern hemisphere) and Ghana
(which straddles longitude 0).

Nothing in Pluvium is tied to Jamaica. The engine works on the sphere, and every source it learns from is global:
terrain, borders, towns, satellite rain, airports and the driving models.

### The rain record arrives late

Pluvium Blend learns from NOAA's CMORPH satellite rain record, and NOAA publishes that record about four and a
half months after the fact. In November 2026 its newest complete month was May. Two things follow. A blend can be
trained only on runs that are about five months old, so `pluvium now` picks the most recent year the record
covers by itself, and `pluvium blend train` with a later period stops and says which period will work. And
`pluvium blend learn` has nothing to learn from on most days; when NOAA publishes another month, each call works
through two weeks of it.

## The upstream weights

Two of the six systems are AI models whose weights are public: GraphCast, which NOAA fine-tunes and runs (first as
GraphCastGFS, since May 2026 as AIGFS), and ECMWF's AIFS. A copy of those weights can be kept beside Pluvium:

```bash
pluvium upstream list                                    # which models, how large, under which licence
pluvium upstream fetch                                   # copy them to ~/.cache/pluvium/upstream, checked
HF_TOKEN=... pluvium upstream mirror --repo you/your-repo    # keep them in a Hub repository, under upstream/
```

Every copy is unchanged and carries its publisher's licence (CC BY 4.0 for DeepMind's and ECMWF's, CC0 for
NOAA's) and a note of where and when it was copied from, with the SHA-256 of each file.

Pluvium does not run these models. It reads the forecasts NOAA and ECMWF make with them. Running them needs a
large GPU and that day's global analysis from NOAA or ECMWF, which no copy of the weights can replace, and the
other four systems are physical models with no weights at all. The copies are there because a feed can be renamed
or switched off, as NOAA's GraphCastGFS feed was on 5 May 2026.

## Hurricane tracks

```bash
pluvium cyclone --root ~/storms                          # every storm alive now: track, cone, passage chances
pluvium cyclone --root ~/storms --time 2025-10-25T22     # as it would have been issued then
```

It reads the National Hurricane Center's public decks and sea-level pressure from six systems, follows the storm in
every ensemble member, and combines the tracks with weights learned from the forecasts that have already verified.
For each storm it writes a bulletin and a data file: the track to five days, the circle that should hold the centre
two times in three at each lead, the chance of the centre passing within 150 km of each listed place, and scenarios
when the tracks split.

## The hourly network

```bash
pluvium forecast --country JM            # every place; tables in ./pluvium-forecast, a report on screen
pluvium place "Port Antonio" --country JM  # hour by hour for one place
pluvium learn                              # check past forecasts against airport reports; update the corrections
pluvium info                               # what the published model is and how it scored
```

These use a trained network from the Hub repository, which exists once the training program below has been run
with a write token. `pluvium forecast` writes:

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

## Train

Every command that trains something, ready to copy, is in [docs/TRAINING.md](docs/TRAINING.md).
`notebooks/pluvium_everything.py` is one notebook cell that does all of it: the package, the Hub repository with
the code, the papers and the upstream weights, the network if a GPU is there, the rain forecast for a country and
the hurricane tracks. `notebooks/pluvium_blend_cell.py` is the shorter cell for the rain forecast alone, on a CPU.

### Pluvium-500 on one GPU

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
[docs/FAILURE_MODES.md](docs/FAILURE_MODES.md). Four short papers are in [docs/papers](docs/papers/README.md), and
the ideas that were built, tested and dropped are in [docs/WHAT_DID_NOT_WORK.md](docs/WHAT_DID_NOT_WORK.md).

Pluvium Blend, in short:

* each system is post-processed by its own small network, trained on every heavy-rain day with weights that keep
  the chances unbiased;
* a pool weighs them by threshold band and lead, fitted on forecasts the networks did not train on;
* online weights move towards whoever has been right, as days verify;
* a forecast beyond anything in the training period is handed back, in part, to what the raw systems say, in the
  places where that improved the forecast of storms withheld from training;
* scenarios arrange the calibrated amounts in space like the ECMWF ensemble's members.

Pluvium Cyclone, in short: a tracker on sea-level pressure, a consensus whose weights minimise the expected error
given the systems' error covariance so far, a circle sized from verified errors and the systems' spread, and a
cloud of tracks from which passage chances are read.

Pluvium's network takes five kinds of input: the global models' fields (GFS, ECMWF IFS and AIFS live, ERA5 in
training); convective indices (K-index, lifted index, mid-level humidity, shear, upper divergence); the rain
falling now, from the satellite; the last three hours of airport reports; and terrain, coasts, sun and season.

Its engine is a stack of iHELIX blocks on a Caribbean mesh of 6,075 points, about 11 km apart over land. Each
block combines geodesic attention with per-head radii from 15 to 500 km, content-addressed index attention, and a
recurrence along the forecast's 43 frames. Four pieces of physics are computed inside the model: mountain rain
(Smith–Barstad linear theory, re-solved every hour with learned parameters), sea-breeze convergence over each
island, fronts and shear lines with arrival times, and the rain falling now advected along the steering wind for
the first 12 hours.

For each town and hour the output is an intensity chain (dry, light, moderate, heavy) from which every timing and
amount answer is read, so that the answers agree with each other. Further heads give thunder, the kind of rain
(drizzle, rain, showers) and the cause. Training scores the satellite's and the airports' reports through each
instrument's learned confusion matrix, a differentiable hidden Markov model, so that neither instrument's errors
are taught as truth. Live, there is one run per driving model and the runs are mixed, calibrated on held-out data
and corrected online from what the airports report afterwards.

The deeper parts of version 3 sit in their own subpackages:

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

## Benchmarks

Losses are reported beside wins in every one of these.

### The second generation of Pluvium Blend

See [docs/FRONTIER_BENCHMARK.md](docs/FRONTIER_BENCHMARK.md). The first full-year benchmark, below, found three
weak spots: 50 mm, winter in mid-latitude climates, and structure at long range. Version 0.3 was aimed at them and
scored on the same year by the same rules, with every part of the design scored on its own.

<!-- frontier-summary:start -->
On the test year, in 7 places (the same year the first Blend was scored on):

- At 50 mm, Brier skill improved on the first Blend in 13 of 28 place and lead-group cells, and in 7 of 7 on days 1-3. Mean skill over the places on days 1-3 went from +0.1901 to +0.2133. The new Blend leads every other system at 50 mm in 15 of 28 cells (the first led in 14).
- In the places scored after the design was final (Jamaica and the US Great Plains), 50 mm skill improved on the first Blend in 4 of 8 cells, and the new Blend leads every other system in 3 (the first: 0).
- Against each post-processed system at 50 mm, 79 of 161 comparisons are better with 95% confidence and 0 are worse.
- At 1 mm, which is rain or no rain, the new Blend leads in 24 of 28 cells (the first: 26), and improved on the first in 13. On days 1-7, 81 of 84 comparisons with single post-processed systems are better with 95% confidence and 0 are worse.
- The CRPS leads in 25 of 28 cells (was 23), the RMSE in 21 of 28 (was 20) and the anomaly correlation in 18 of 28 (was 16).
- Days 11-14 got slightly worse. There the new Blend leads every other system at 1 mm in 4 of 7 places (the first: 6) and at 50 mm in 2 (the first: 5). Mean Brier skill at 1 mm on those days went from +0.0394 to +0.0390. The pool alone scores +0.0396: the online weights for that lead group learn from few forecasts that verify late, and cost a little.
- The online weights were scored with each day's rain known two days later, and the satellite record is published about 140 days late. With that delay, mean Brier skill on days 1-3 over 7 places is +0.3585 at 1 mm and +0.2081 at 50 mm. The figures above correspond to +0.3591 and +0.2133, and with no online weights the Blend scores +0.3583 and +0.2055.
- Each place's staged storms kept the record-storm hand-over in Jamaica, the Caribbean, the Maritime Continent, the US Great Plains and the Southern Andes and switched it off in the Alps and the US Pacific Northwest. That verdict had the sign of what the hand-over then did in the test year in 7 of 7 places (2 of the 2 scored after the rule was final). The out-of-fold audit tried first would have been right in 4 of 7.
- On days 11-14 one coupled scenario has 0.88 of the weather's spatial variability (raw ECMWF ENS: 0.84), and the CRPS of rain averaged over two-degree boxes is 2.26 mm against 2.93 for uncoupled draws and 2.57 for the raw ensemble (means over 7 places).
<!-- frontier-summary:end -->

![Heavy rain, place by place](docs/img/frontier_heavy_rain.png)

### Hurricane tracks

See [docs/CYCLONE_BENCHMARK.md](docs/CYCLONE_BENCHMARK.md). Pluvium Cyclone is scored against the National
Hurricane Center's best tracks, beside the Center's official forecast and guidance, on the forecasts all of them
made. Settings were chosen on three seasons and tested once on a fourth (East Pacific 2025, 167 forecasts of 18
storms).

On that test season it had a lower error than ECMWF HRES at all 8 leads from 12 to 120 hours (all 8 with 95%
confidence) and than NOAA GFS at all 8 (5 with 95% confidence). It was level with TVCN, the Center's simple
consensus. It was behind the official forecast at every lead, by 2 to 25 nautical miles (125 against 100 at five
days). On the North Atlantic 2025 development season it was ahead at every lead, but its settings were chosen with
that season in view, so that result should not be relied on. Google DeepMind's cyclone model is better than both.

The circle meant to hold the centre two times in three held it 70% of the time. The chance of the centre passing
within 150 km of a place scored a Brier skill of 0.47, against 0.42 for the raw cloud of ensemble members.

![Track error by lead](docs/img/cyclone_track_error.png)

### The first full year, 14 days, six regions

See [docs/FULL_YEAR_BENCHMARK.md](docs/FULL_YEAR_BENCHMARK.md). This is version 0.2 of Pluvium Blend against ECMWF
ENS, HRES and AIFS and NOAA GEFS, GFS and GraphCast, from real archived runs only. It is scored on April 2025 to
March 2026, every season, in the Caribbean, the Maritime Continent, the Alps, the US Pacific Northwest, the US
Great Plains and the southern Andes. Each system is scored raw and after the same post-processing.

For rain of 1 mm or more, Pluvium Blend had the highest Brier skill of every system, raw or post-processed, in all
six regions on days 1 to 10, and in 5 of 6 on days 11 to 14. Against each system after equal post-processing it was
better with 95% confidence in all 72 comparisons on days 1 to 7 at 1 mm, and never worse. On days 1 to 10 it was
best in 18 of 18 region and lead contests for Brier skill at 10 mm, CSI, ROC area and CRPS, in 17 for 25 mm, 16 for
RMSE and 14 for anomaly correlation.

It did not lead at 50 mm, where post-processed AIFS often won at short range, or in anomaly correlation beyond a
week in the mid-latitudes, where raw ensembles keep more of it. In winter on days 1 to 3 it led at 1 mm in 2 of 6
regions, was within 0.004 of the leader in three more, and was 0.025 behind post-processed ENS in the Great Plains.

For Jamaica at 8 km, with Pluvium's network as a seventh component, Pluvium Blend led at 1 mm on days 1 to 7. It
lost the amount scores in autumn to Hurricane Melissa, because it shrank a record forecast the raw systems had
right: two days ahead it gave a 66% chance of 50 mm against 83 to 100% from every raw system. The daily table
has reported since then when the raw systems agree on 50 mm or more.

### Against the operational systems, July and August 2025

See [docs/OPERATIONAL_BENCHMARK.md](docs/OPERATIONAL_BENCHMARK.md). `pluvium operational` scores Pluvium's network
against ECMWF ENS, ECMWF HRES, ECMWF AIFS, NOAA GEFS, NOAA GraphCast and NOAA GFS. It uses real archived runs
only, with no reanalysis anywhere, and calibrates every system the same causal way.

These are early results, on Jamaica's 450 land cells, with the network at its small size (0.9M parameters, about
two hours of CPU training) and driven only by GFS. In July and August 2025, with the protocol frozen before the
data were downloaded, Pluvium had the highest skill for rain of at least 1 mm on days 1 and 2: +0.127 and +0.118
against +0.087 and +0.091 for the best operational system. With 95% confidence it beat ECMWF ENS, ECMWF HRES, AIFS
and GraphCast; against GEFS and GFS the lead is not yet significant. It tied at 10 mm. Its CRPS on daily totals was
40-50% smaller than ECMWF ENS's and GEFS's.

In May and June 2025, the first test, Pluvium lost after calibration: its calibration windows held an unusually
wet April it had never trained on. At 25 and 50 mm no system, ECMWF's included, did meaningfully better than a
forecast that never says rain.

### Between versions

See [docs/BENCHMARK.md](docs/BENCHMARK.md). `pluvium bench` trains each version of the network from scratch on the
same data with the same time budget. Version 3 nearly doubles the skill of 24-hour totals over version 2 (CRPS
skill +44.6% against +23.8%, with emulated forecast errors). It also has the best hourly skill and critical success
index. Heavy-rain hours and thunder are its weak spots.

## Data and attribution

ERA5 (Copernicus Climate Change Service), CMORPH CDR and ISD (NOAA NCEI), GFS and GEFS (NOAA NCEP), NOAA's
real-time GraphCast and AIGFS, ECMWF open data: IFS, ENS and AIFS (CC BY 4.0), GOES-East imagery and lightning
(NOAA), METAR (NOAA Aviation Weather Center), GeoNames (CC BY 4.0), geoBoundaries (CC BY 4.0), Terrarium elevation
tiles (Mapzen and its sources).

Code: Apache 2.0, CorX Labs.
