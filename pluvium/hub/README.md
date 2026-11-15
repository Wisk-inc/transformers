---
license: apache-2.0
library_name: pluvium
pipeline_tag: time-series-forecasting
language:
- en
tags:
- weather
- weather-forecasting
- climate
- meteorology
- earth-science
- precipitation
- rainfall
- extreme-rainfall
- nowcasting
- probabilistic-forecasting
- tropical-cyclones
- hurricane
- multi-model-ensemble
- post-processing
- online-learning
- geospatial
- caribbean
- pytorch
- naturev1
- ihelix
- graphcast
- aifs
metrics:
- brier_score
- crps
---

# Pluvium-500

Rain forecasts for every town, in any country between 60 S and 60 N, and hurricane tracks. There are three
products in one package (`pip install pluvium`).

| Product | What it forecasts | Needs |
|---|---|---|
| Pluvium Blend | The chance of 1, 10, 25 and 50 mm on each of the next 14 days, from six operational systems at once | A CPU |
| Pluvium Cyclone | Where every tropical cyclone in the North Atlantic and the eastern North Pacific will go: track, a circle that should hold the centre two times in three, and the chance of passing near each place | A CPU |
| Pluvium-500, the network | Rain hour by hour for three days at every town: whether it rains, when it starts, how heavy, how long, how much, thunder, flood risk | One GPU to train |

The six systems are ECMWF ENS, ECMWF HRES, ECMWF AIFS, NOAA GEFS, NOAA GFS and NOAA's run of GraphCast.

> A research model, not an official forecast. Follow your national meteorological service and, for hurricanes, the
> US National Hurricane Center.

## One command

```bash
pip install pluvium
pluvium now --country JM
```

`pluvium now` does whatever is still missing. It sets the country up (borders, terrain, every town). It looks in
this repository for a blend already trained for that country (`blends/`) and downloads it; if there is none, it
trains one on the most recent year whose rain has been published. That took 86 minutes for Jamaica on a four-core
machine, nearly all of it downloads, and left 41 MB on disk. Then it reads the newest run of each of the six systems for the country and prints the forecast for the
largest towns. The second time it takes about a minute.

`--parishes` prints one block per parish (province or state in other countries): the typical place in it and the
range across its places. `--parish "St Ann"` prints the largest towns of one parish.

For the first three days each line names the six-hour period in which rain is most likely, in the place's own
clock, and `--when` lists all four periods of each day with their chance. The chance for a period is the calibrated
chance of a wet day, spread over the day by the systems' members mixed half and half with when it usually rains at
the place. On Jamaica's wet days the mix placed the rain 7.8% better than the usual pattern alone, and the members
alone placed it 4.9% worse. It rarely changes which period is the most likely one. The measurements are in
[docs/TIMING_BENCHMARK.md](docs/TIMING_BENCHMARK.md). This needs no GPU.

`--country` takes any ISO codes (`TT`, `DO,HT`, `PH`, `KE`). The table it writes has one row per place and day,
with the place's parish: the chance of 1, 10, 25 and 50 mm; the median, 90th percentile and expected rain; how many raw systems call for 50 mm;
an alert tier (watch, warning, extreme); whether the forecast is beyond the training record; and how rare the
expected amount is at that place. `pluvium cyclone --root ~/storms` writes a bulletin for every storm alive now.

In Python:

```python
from pluvium.quick import forecast, brief
got = forecast("JM")
print(brief(got["result"], got["names"], got["population"]))
```

## The six systems, and what you have to do about them

Nothing. Pluvium does not need the six systems' output to be fetched or converted by hand. At forecast time it asks
each public archive (ECMWF open data, NOAA on AWS) for the rain fields over the country and nothing else, which
comes to a few megabytes. A system whose newest run is not out yet is read from its previous run, and a system
that is missing altogether is left out; the blend was trained to forecast from whichever are there.

Two of the six are AI models with public weights: NOAA's run of GraphCast and ECMWF's AIFS. Copies of those
weights are kept in this repository under `upstream/`, unchanged, each with its publisher's licence, and
`upstream/README.md` lists them. `pluvium upstream mirror --repo <repository>` makes or refreshes the copies.

Pluvium does not run those models. Running them needs a large GPU and that day's global analysis from NOAA or
ECMWF, which no copy of the weights can replace, and the other four systems are physical models with no weights
at all. A forecast feed can still change under a project like this one: NOAA stopped publishing its GraphCast run
as GraphCastGFS on 5 May 2026 and has published the successor, AIGFS, since. From version 0.4 the package reads
AIGFS after that date.

## What is in this repository

- All the code the three products are built from: the `pluvium` package, its tests, the benchmark runners and the
  scripts that write the result pages.
- Four technical papers in [docs/papers](docs/papers/README.md).
- The benchmarks, with losses reported beside wins:
  [rain, second full-year benchmark](docs/FRONTIER_BENCHMARK.md),
  [rain, first full-year benchmark](docs/FULL_YEAR_BENCHMARK.md),
  [hurricane tracks](docs/CYCLONE_BENCHMARK.md),
  [against operational systems, summer 2025](docs/OPERATIONAL_BENCHMARK.md).
- A record of what was tried and dropped: [docs/WHAT_DID_NOT_WORK.md](docs/WHAT_DID_NOT_WORK.md).
- Trained blends, in `blends/<country>/`, once somebody has published one with
  `pluvium blend publish --root <folder> --repo <repository>`. Pluvium opens a downloaded blend in a way that
  cannot run code.
- The upstream weights described above, in `upstream/`.
- The weights of the Pluvium-500 network, once the training program below has run with a write token. Its own
  card, with the scores it measured, is then in `NETWORK_CARD.md`. Pluvium Cyclone needs no download: what it
  learned on 2024 and 2025 ships inside the package, and it keeps learning.

## Everything in one cell

[notebooks/pluvium_everything.py](notebooks/pluvium_everything.py) is one notebook cell that installs the package,
builds this repository (code, papers, the collection, the upstream weights), trains the network if a GPU of 24 GB
or more is there, forecasts the rain for a country and tracks the hurricanes alive now. Each step can be run
again; what is already done is not done twice.

## Train the Pluvium-500 network on one GPU

Paste into one notebook cell on a machine with a GPU of 24 GB or more and about 150 GB of disk. Put a Hugging Face
write token where it says so. Checkpoints and data are mirrored here, so a session that stops can pick up where it
left off, and the finished model is published here.

```python
GPU_HOURS = 24                              # wall-clock budget for this session
HF_TOKEN = "PASTE_YOUR_ORG_TOKEN_HERE"      # a write token (Hub -> Settings -> Access Tokens)
HF_REPO = "CorxLabs/Pluvium-500"            # where data, checkpoints and the finished model are kept
ROOT = "/workspace/pluvium"                 # working folder with about 150 GB free
SIZE = "pluvium-500"                        # or pluvium-base for a smaller, faster model
COUNTRY = ""                                # ISO codes such as "JM" or "DO,HT"; empty for the whole Caribbean

import os, subprocess, sys

def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

sh(f"{sys.executable} -m pip install -q --upgrade pluvium")
probe = ("import torch; assert torch.cuda.is_available(), 'no GPU visible';"
         "x = torch.ones(8, device='cuda'); print(torch.cuda.get_device_name(0), torch.__version__, float(x @ x))")
if subprocess.run([sys.executable, "-c", probe]).returncode != 0:
    sh(f"{sys.executable} -m pip install -q --upgrade torch --index-url https://download.pytorch.org/whl/cu128")
    subprocess.run([sys.executable, "-c", probe], check=True)

env = dict(os.environ, GPU_HOURS=str(GPU_HOURS), HF_REPO=HF_REPO, PLUVIUM_ROOT=ROOT, PLUVIUM_SIZE=SIZE,
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
if not HF_TOKEN.startswith("PASTE"):
    env["HF_TOKEN"] = HF_TOKEN
else:
    print("No Hub token: nothing is mirrored, so a stopped session cannot resume and nothing is published.")
cmd = [sys.executable, "-u", "-m", "pluvium.program"] + (["--country", COUNTRY] if COUNTRY else [])
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
try:
    for line in proc.stdout:
        print(line, end="", flush=True)
finally:
    if proc.poll() is None:
        proc.terminate()
print(f"finished with exit code {proc.wait()}; the forecast is in {ROOT}/forecast/report.txt")
```

Every other training command is in [docs/TRAINING.md](docs/TRAINING.md).

## Results

### Rain: Pluvium Blend against six operational systems, a full year, 14 days

Brier skill, CSI, CRPS, RMSE and anomaly correlation against CMORPH satellite rain, April 2025 to March 2026, in the
Caribbean, the Maritime Continent, the Alps, the US Pacific Northwest, the US Great Plains, the southern Andes, and
Jamaica at 8 km. Every system is scored raw and after the same post-processing. A system counts as beaten only if
the Blend is ahead of both.

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

Full tables, every part of the design scored on its own, and what is still weak:
[docs/FRONTIER_BENCHMARK.md](docs/FRONTIER_BENCHMARK.md).

![Heavy rain, place by place](docs/img/frontier_heavy_rain.png)

### Hurricane tracks: Pluvium Cyclone against the National Hurricane Center's guidance

Settings were chosen on three seasons and tested once on a fourth (East Pacific 2025: 167 forecasts of 18 storms).
Mean track error in nautical miles for the early guidance, on the forecasts all four made.

| East Pacific 2025, the test season | 12 h | 24 h | 36 h | 48 h | 60 h | 72 h | 96 h | 120 h |
|---|---|---|---|---|---|---|---|---|
| Pluvium Cyclone | 20 | 30 | 37 | 46 | 57 | 69 | 90 | 125 |
| NHC official forecast | 18 | 27 | 34 | 40 | 46 | 52 | 73 | 100 |
| NHC consensus TVCN | 19 | 27 | 35 | 43 | 53 | 64 | 92 | 127 |
| NHC corrected consensus HCCA | 19 | 30 | 34 | 43 | 51 | 58 | 78 | 107 |

On the test season Pluvium Cyclone had a lower error than every single model as the National Hurricane Center
tracks it: lower than ECMWF HRES at all 8 leads (all 8 with 95% confidence) and than NOAA GFS at all 8 (5 with 95%
confidence). It was level with TVCN, the Center's simple consensus. It was behind the official forecast at every
lead, by 2 to 25 nautical miles, and behind Google DeepMind's cyclone model.

On a development season (North Atlantic 2025) it was ahead of the official forecast at every lead. Its settings
were chosen with that season in view, so that result should not be relied on.

The circle meant to hold the centre two times in three held it 70% of the time. The chance of the centre passing
within 150 km of a place scored a Brier skill of 0.47, against 0.42 for the raw cloud of ensemble members.

![Track error by lead](docs/img/cyclone_track_error.png)

Full tables: [docs/CYCLONE_BENCHMARK.md](docs/CYCLONE_BENCHMARK.md).

### The network against operational systems

At its small size (0.9M parameters, two hours of CPU training, driven only by GFS), on Jamaica's 450 land cells in
July and August 2025, Pluvium's network had the highest skill for rain of 1 mm or more on days 1 and 2 (+0.127 and
+0.118 against +0.087 and +0.091 for the best operational system). In May and June 2025 it lost after calibration.
Both periods are in [docs/OPERATIONAL_BENCHMARK.md](docs/OPERATIONAL_BENCHMARK.md). The full-size Pluvium-500 has
not been scored there yet.

## What it is not good at

- The rain test year is the one on which the weak spots of the first version were found. Design choices were made
  inside the training year, but a fresh year is the test that counts.
- The satellite rain record Pluvium learns from is published about four and a half months late. A blend can
  therefore be trained only on runs that are five months old, and its online weights learn that much later than
  they did in the benchmark, which gave them each day's rain after two days. The benchmark page says what that
  delay costs.
- Heavy rain is not won everywhere, and at long range every system is close to the climatology.
- Beyond a week in mid-latitudes, raw models keep a higher anomaly correlation for the expected amount.
- Record storms are still shrunk in some places. Handing a record forecast to the raw systems helps where they are
  the better guide and does harm where they overcall heavy rain. Each place decides from four storms staged in its
  own training year. Where they say no, a storm beyond the training record is forecast by the calibration alone,
  and the table flags such a forecast. The verdict can be wrong, and `pluvium blend train --handover always` or
  `never` overrules it.
- NOAA's GraphCast input changed from GraphCastGFS to AIGFS in May 2026. Every benchmark here was scored before
  that, so the Blend and the tracker have not been scored with AIGFS.
- Hurricanes are forecast by track only: no intensity, size, wind, surge or rain, and no forecast of formation. On
  the clean test the track was behind the official forecast.
- There is one truth for rain, a satellite estimate that is weak on frozen and orographic precipitation.
- Rain only: no temperature or wind. Nothing poleward of 60 degrees. The Blend gives daily totals.
- The flood rule has not been verified against flood reports.

## How it works

The network runs on the NatureV1 / iHELIX engine: geodesic attention on a mesh of the sphere, physics computed
inside the model (mountain rain, sea breezes, fronts, advection of the rain falling now), and a chain of intensity
states per town and hour from which every timing and amount is read.

Pluvium Blend post-processes each system with its own small network trained on every heavy-rain day, pools them
with weights by threshold and lead, and re-weighs them online as days verify. A forecast that lies beyond the
training record is handed back, in part, to the raw systems, in the places where that improved the forecast of
storms withheld from training.

Pluvium Cyclone tracks the storm in every ensemble member from sea-level pressure and combines the systems with
weights that minimise the expected error given their error covariance so far.

The full account is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and every module's purpose is in
[docs/MODULES.md](docs/MODULES.md).

## Papers

1. [Tail-aware pooling with online weights](docs/papers/01_blend_frontier.md)
2. [Forecasting beyond the training record](docs/papers/02_record_storms.md)
3. [Sharp scenarios from smooth chances](docs/papers/03_scenarios.md)
4. [A learning consensus for tropical cyclone tracks](docs/papers/04_pluvium_cyclone.md)

They are research notes from one person's project and have not been peer reviewed.

## Data and attribution

ERA5 (Copernicus Climate Change Service), CMORPH CDR and ISD (NOAA NCEI), GFS, GEFS and MRMS (NOAA), NOAA's
real-time GraphCast and AIGFS, ECMWF open data: IFS, ENS and AIFS (CC BY 4.0), ATCF decks (US National Hurricane
Center), GOES imagery and lightning (NOAA), METAR (NOAA Aviation Weather Center), GeoNames (CC BY 4.0),
geoBoundaries (CC BY 4.0), Terrarium elevation tiles (Mapzen and its sources).

The weights under `upstream/` are other publishers' work: GraphCast by Google DeepMind (CC BY 4.0), GraphCastGFS
and AIGFS by NOAA (released by NOAA under CC0; fine-tuned from GraphCast, so DeepMind's attribution is kept with
them), and AIFS by ECMWF (CC BY 4.0). Each folder carries the licence text and the notice to keep.

## Licence and citation

Everything Pluvium made, code and trained weights, is under Apache 2.0, CorX Labs. That licence does not cover the
folder `upstream/`, which holds the publishers' own licences.

```
Nathan (2026). Pluvium: rain and tropical cyclone forecasts from six operational systems.
https://pypi.org/project/pluvium/
```
