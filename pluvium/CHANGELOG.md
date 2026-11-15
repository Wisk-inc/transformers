# Changelog

## 0.3.0 (November 2026)

Aimed at the three weak spots the first full-year benchmark found (the heavy tail, winter in mid-latitude climates,
structure at long range), and adds hurricane tracks. Results, losses included, are in
[docs/FRONTIER_BENCHMARK.md](docs/FRONTIER_BENCHMARK.md) and [docs/CYCLONE_BENCHMARK.md](docs/CYCLONE_BENCHMARK.md).

### Pluvium Blend, second generation

`pluvium blend train` now fits `pluvium.frontier.FrontierBlend`. `--classic` trains the first Blend, and models
saved by 0.2 load and forecast as before.

- **Components see every heavy day** (`pluvium.tails.stratify`): every row with observed or forecast rain of 25 mm
  or more is kept, the rest thinned, with weights that keep the chances unbiased.
- **Online weights** (`pluvium.pool.online`): the components and the pool are re-weighed after every verified day.
  New command: `pluvium blend learn`.
- **Record storms** (`pluvium.tails.support`, `pluvium.tails.override`): a forecast beyond anything in the training
  period is handed back, in part, to what the raw systems say.
- **Each place decides about the hand-over** (`pluvium.tails.withheld`): at every training the four wettest days are
  withheld, forecast without having been seen, and scored with and without it. It is used only where it helped.
  Added after the benchmark showed the hand-over hurts where the raw systems overcall heavy rain.
  `--handover always` or `never` overrules the verdict. An audit of the training period's ordinary days
  (`pluvium.tails.audit`) is reported beside it and decides nothing.
- **Scenarios** (`pluvium.scenarios`): calibrated amounts arranged in space like the ECMWF ensemble's members
  (ensemble copula coupling), or like past observed days (Schaake shuffle).
- **The forecast table** has three new columns: an alert tier for 50 mm (`pluvium.tails.alerts`), a flag for a
  forecast beyond the training record, and how rare the expected amount is at the place (`pluvium.tails.returns`,
  `pluvium.tails.gpd`).

### Pluvium Cyclone (new)

`pluvium cyclone` forecasts the track of every tropical cyclone in the North Atlantic and the eastern North Pacific
from six operational systems. Track only.

- A tracker on sea-level pressure for every member of ECMWF ENS and NOAA GEFS and for ECMWF HRES, ECMWF AIFS, NOAA
  GFS and NOAA GraphCast (`pluvium.cyclone.tracker`).
- A consensus whose weights are learned through the season from the systems' error covariance
  (`pluvium.cyclone.learner`, `pluvium.cyclone.minvar`), starting from weights learned on 2024 and 2025.
- A circle that should hold the centre two times in three (`pluvium.cyclone.cone`), the chance of the centre passing
  within 150 km of each place (`pluvium.cyclone.approach`), and scenarios when the tracks split
  (`pluvium.cyclone.clusters`).

### Benchmarks

- `python -m pluvium.benchmarks.frontier_bench`: the second full-year benchmark, with every part of the design
  scored with and without the others.
- `pluvium.benchmarks.tail_bench`, `scenario_bench`, `winter_bench`: the tail and what a warning meant, coupled
  scenarios against raw ensembles, and winter against a second truth (MRMS).
- `python -m pluvium.benchmarks.cyclone_bench`: tracks against the National Hurricane Center's best tracks, beside
  its official forecast and guidance.

### Documents

- Four technical papers in [docs/papers](docs/papers/README.md).
- [docs/TRAINING.md](docs/TRAINING.md): every command that trains something, ready to copy.
- [docs/WHAT_DID_NOT_WORK.md](docs/WHAT_DID_NOT_WORK.md): the ideas that were built, tested and dropped.
- `hub/upload_to_hub.py`: puts the code, model card, papers and benchmarks on the Hugging Face Hub.

## 0.2.0

Pluvium Blend (14 days of daily rain chances from six operational systems), any-country set-up, and the first
full-year benchmark.

## 0.1.0

Pluvium's network on the NatureV1 / iHELIX engine: hour-by-hour rain for every town, trained on one GPU.
