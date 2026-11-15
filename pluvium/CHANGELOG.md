# Changelog

## 0.4.2 (November 2026)

Found by training a full year for Jamaica from an empty folder and publishing the result.

- The note uploaded beside a published blend listed the staged storms as a Python list and said the Brier score
  "changed by +49.3%" when it had fallen by that much. It now names the days, gives both scores and says which way
  is better. The training log's line uses the same sign as the yearly audit's: the change in the score, so a
  negative number is an improvement.
- `pluvium upstream mirror` with some models named rewrote `upstream/README.md` listing only those. The index now
  lists every model in the repository.
- Running `pluvium upstream mirror` or `pluvium blend publish` a second time no longer prints the Hub library's
  line about an empty commit.
- First-time training was described as two hours and 5 GB. Measured for Jamaica on a four-core machine it took
  86 minutes and left 41 MB on disk, and the documents now say so.
- The one cell reports a step that took under 100 seconds in seconds.

## 0.4.1 (November 2026)

Found by running the Hub steps against a real account.

- A token the Hub refuses now gives one plain line in place of a traceback, in `pluvium blend publish`,
  `pluvium upstream mirror` and `hub/upload_to_hub.py`. The common case is a fine-grained token that was not given
  the organisation it is asked to write to, and the message says how to add it.
- `pluvium blend publish` no longer prints the Hub library's warning about card metadata for the note it uploads
  beside a blend.

## 0.4.0 (November 2026)

One command from nothing to a forecast, copies of the upstream AI models' weights, and three faults found by
running the system live.

### One command

- `pluvium now --country JM` sets the country up, downloads a trained blend from the Hub repository if one is
  published there, trains one if not, reads the newest runs and prints the forecast for the largest towns
  (`pluvium.quick`). `--place`, `--learn`, `--storms` and `--handover` adjust it. In Python:
  `pluvium.quick.forecast("JM")`.
- `pluvium blend publish --root FOLDER --repo REPO` uploads a trained blend to `blends/<country>/`, so that the
  next person's `pluvium now` is a download.
- Blend files now hold tensors and plain values only. A blend that was downloaded is opened by a reader that cannot
  run code, and is not used if it does not open that way. Files written by 0.3 still load when they were made on the
  same machine.

### Upstream weights

- `pluvium upstream list`, `fetch` and `mirror --repo REPO` (`pluvium.upstream`) copy the published weights of the
  AI models whose forecasts Pluvium reads: DeepMind's GraphCast, NOAA's GraphCastGFS and AIGFS, and ECMWF's AIFS
  Single 0.2.1, 1.0, 1.1 and 2.0. The copies are unchanged and checked, and each carries its publisher's licence
  (CC BY 4.0, or CC0 for NOAA's) and a note of its source with the SHA-256 of every file. Pluvium does not run these
  models; it reads the forecasts NOAA and ECMWF make with them.

### Faults found live, and fixed

- NOAA stopped publishing its GraphCast run as GraphCastGFS on 5 May 2026 and has published the successor, AIGFS,
  since. Version 0.3 read nothing for that input after that date and forecast from the other five systems. The
  rain reader and the hurricane tracker now read AIGFS from 6 May 2026.
- NOAA publishes the satellite rain record Pluvium learns from about four and a half months late. Training on a
  period the record did not cover failed with an error that explained nothing, and `pluvium blend learn` never
  found a day to learn from. Now the training period defaults to the most recent year the record covers, a period
  without rain stops with a message that names one that will work, and `blend learn` works through newly published
  months two weeks a call.
- The benchmark gave the online weights each day's rain after two days, which that record cannot do. The test
  year was scored again with the rain 140 days late, and the result is in
  [docs/FRONTIER_BENCHMARK.md](docs/FRONTIER_BENCHMARK.md) and paper 1.

### The Hub

- `hub/upload_to_hub.py` also creates a collection in the organisation, holding the model and the papers it
  builds on.
- `pluvium program` publishes the network's own card as `NETWORK_CARD.md` and no longer replaces a card the
  repository already has.
- `notebooks/pluvium_everything.py` is one cell that does all of it.

### Documents

- The papers, the model card and the readme were revised against Wikipedia's field guide to signs of AI writing
  (`scripts/check_style.py` holds the checks, and the test suite runs them). Every reference is now cited where it
  is used and carries its DOI or arXiv number, each checked against Crossref or arXiv.

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
