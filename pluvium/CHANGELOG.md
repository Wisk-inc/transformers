# Changelog

## 0.5.5 (November 2026)

The third GPU run trained, then stopped on a missing file. Its log was on the Hub this time, so the cause is known.

- Training ran on the 96 GB card for the first time: steps were sized by measuring (one forecast 41.9 GB, two
  81.4 GB), a step that ran out of memory was made smaller, and steps were logged. Then a data loader asked for
  `mesh_2016.npy` and it was not there.
- An earlier run's uploads had been refused half way, so the Hub held 2016's `targets` file and not its `mesh`
  file. The stager asked only the targets file whether a year was done. It now asks both, and the program checks
  that every data file is there before training starts.
- After that failure the run did not end for an hour and a half, it waited for the background fetch of stage B.
  That fetch is stopped now when a run ends or fails.
- The first real step ran out of memory at a size that had measured 83.9 GB on a 94 GB card, right after three
  larger sizes had been tried and failed, and the run went on with one forecast at a time. I think the failed
  tries had not given their memory back yet, the log does not prove it. Memory is now collected and freed before
  anything is given up, the same step is tried once more, then the extra target cells go, and a forecast goes
  last. What PyTorch says about a step that runs out of memory is written to the log from now on.
- A long scoring pass at the end says where it is every five minutes.

## 0.5.4 (November 2026)

The GPU run is made hard to stop.

- The second GPU run stopped as well, right after it measured one forecast at 38.6 GB on the 96 GB card. The old
  batch rule then tried three forecasts, and it moved that batch to the GPU outside the lines that catch running out
  of memory. Now a batch that cannot fit is not tried at all, and the move is inside those lines.
- A step size that fails for another reason than memory (a CUDA library refusing a tensor that large) is taken as
  too large. It is written down beside the checkpoint, so the run started after it does not try that size again.
- One batch that cannot be trained on is left out and said, the run goes on. Five in a row stop it.
- `pluvium program` now also starts the run again when it was killed (the machine ran out of RAM) or when it
  printed nothing for two hours. A run started again draws its batches differently, so it does not meet the batch
  that stopped it at the same step.
- The last 400 lines of a run that stopped are kept in `logs/attempt-N.txt` in the working folder and in the Hub
  repository, with anything that looks like a token blanked. Both earlier runs stopped with an error nobody saw the
  last line of.

## 0.5.3 (November 2026)

No change to the forecast. The pages were rewritten.

- The readme is short now and in my own words: what it is, is it standalone, can it tell the time, and tables for
  the rest. The page on PyPI and the card on the Hub are that same text.
- The old readme is `docs/GUIDE.md`, the long manual, and the old card is `docs/MODEL_CARD.md`. Nothing was thrown away.
- The readme says plainly that Pluvium-500 is not finished: the first GPU run stopped at 138 minutes and the hour
  by hour forecast is not published yet.
- `scripts/check_style.py` also reads Python files now (comments, docstrings, messages), and the tests run it on
  all of them. It found six phrases in 215 files, they are reworded.
- `docs/STYLE.md` has the rules the pages are written by.

## 0.5.2 (November 2026)

- The staged data went up to the Hub one commit per file, about 150 at once. The Hub takes at most 128 commits an
  hour in a repository, so most were refused with `429 Too Many Requests`, the log filled with those refusals, and
  the checkpoints of the following hour could be refused too. Training was not affected, but a new session could not
  resume from the Hub's copy. The data now go up in commits of 60 files, and only the files the Hub does not hold
  already.
- The automatic restart of 0.5.1 also covers runs with the old batch rule (`VRAM = "fastest"`), and its last try is
  one forecast per step (`--batch 1`). On the RTX PRO 6000 one Pluvium-500 forecast took 38.6 GB at its peak, so a
  step holds two at most.

## 0.5.1 (November 2026)

The GPU training starts itself again when it stops.

- The first GPU run of 0.5.0 (an RTX PRO 6000, 96 GB) downloaded its data, began training, and stopped after 138
  minutes with an error inside the network's attention. The cell showed only "ended with exit code 1", and the
  error's own last line was not seen, so the cause is not confirmed. Three weaknesses that could each stop a run
  this way are fixed.
- Steps were sized on the first issue time, whatever its weather. A step needs more memory when it rained, because
  the loss scores amounts only where rain fell and each intensity only where it occurred, so a step sized on a dry
  day could run out on a stormy one. Steps are now sized on the rainiest issue time, which the curriculum already
  weighs. A size found the old way is measured again when a run resumes.
- After running out of memory, training gave up extra target cells first, and only those. When the encoder was what
  ran out, that did not help, and the run stopped after eight tries. Now each try gives up forecasts and target cells
  together, up to sixteen tries.
- The CUDA libraries report running out of memory in their own words (`CUBLAS_STATUS_ALLOC_FAILED`, `CUDA error: out
  of memory`). These stopped the run; now they shrink the step like PyTorch's own out-of-memory error.
- When a run on a GPU stops with any error, `pluvium program` starts it again from its last checkpoint in a fresh
  process: first with steps sized to 10 points less of the card (85% instead of 95%), then with the batch rule of
  0.4.2. The data are on disk by then, so a second try is training again within minutes. `PLUVIUM_RETRIES=0` turns
  this off.
- A command that fails in the one-cell notebook is reported with its last line, which is where Python puts the
  error, instead of only its exit code.

## 0.5.0 (November 2026)

When in the day it rains, for the first three days. No GPU needed.

- `pluvium now` names, for each place and each of the first three days, the six-hour period in which rain is most
  likely, in the place's own clock. `--when` lists all four periods of each day with their chance, `--no-when`
  skips the work (about 40 seconds), and every period of every place is written to `blend_<date>_periods.csv`.
- The chance for a period is the blend's calibrated chance of a wet day, times the chance that the period is wet
  given a wet day. That second number is a half-and-half mix of the systems' members and of when it usually rains
  at the place, from six-hour satellite rain over the blend's training year (`pluvium.timing`).
- Checked for Jamaica on 61 past runs (docs/TIMING_BENCHMARK.md). On wet days the mix placed the rain 7.8% better
  than the usual pattern alone (90% interval 3.6% to 11.9%). The members alone placed it 4.9% worse, so a blend
  with no record of the usual pattern shows no periods. The mix named the wettest period no more often than the
  usual pattern does (66.7% against 66.5%). The chance printed for a period is only as good as the daily chance: a
  weak blend scored on runs it had not seen was 5.8% worse than the usual chance of rain in that period.
- `pluvium blend timing --root ...` adds the usual pattern (`timing.npz`, 6 KB for Jamaica) to a blend that has
  none. `pluvium blend train` makes it, `pluvium blend publish` shares it, and a blend downloaded before its
  repository had the file gets it on the next run.
- `pluvium now --days 3` read every system out to day 14 and threw eleven days away. It now reads as far as asked.
- `python -m pluvium.benchmarks.timing_bench --root ...` repeats the check for another country.

## 0.4.3 (November 2026)

The GPU training starts, and fills the card.

- The GPU training never started from a notebook cell. Every cell ran `python -m pluvium.program`, and
  `pluvium/program.py` had no entry point, so in every release up to 0.4.2 that command ended at once with exit code
  0. The cell then reported a finished program. Nothing was trained and the GPU stayed idle. `pluvium program` on
  the command line always worked. The module now has its entry point, the cells start the program through
  `pluvium program`, and they stop with an error if no stage was recorded. A test starts every module the cells and
  documents start, in a real process.
- The program held out 2022 by default, and the hourly ERA5 archive it reads ends on 31 December 2021. Staging 2022
  stopped the run with an IndexError, after every training year had been downloaded. It now trains on 2015 to 2020
  and holds out 2021, and reads the archive's first and last year when it starts. Found by running the program end
  to end for the first time, on a CPU at its smallest size.
- Before, training kept the batch that was fastest per forecast, tried no batch above 32, and always used gradient
  checkpointing. Speed per forecast stops rising at a small batch, so on a large GPU most of the memory stayed
  empty. Now each step is sized to 95% of the card by measuring (`pluvium.training.fit`): gradient checkpointing on
  or off, whichever trains more forecasts per second; as many forecasts per step as fit; and, with what is left,
  more of the area's cells scored per forecast.
- The learning rate is unchanged up to a batch of 8 and grows with the square root of the batch above it, to at
  most twice.
- A step that runs out of memory no longer stops the run: the step is made smaller and training carries on.
- The batches waiting in the machine's RAM are kept within 30% of it. A batch sized to a large GPU is large in RAM
  too, and every loader worker holds several.
- Held-out forecasts are scored at the usual cells whatever training is sized to, and their mean no longer depends
  on how many are scored at a time.
- Every progress line shows `VRAM used/total`. `PLUVIUM_VRAM_SHARE`, `PLUVIUM_VRAM=fastest` and `PLUVIUM_BATCH`
  (or `--vram-share`, `--vram`, `--batch`) change the rule. A checkpoint from an earlier version is sized again
  when it resumes.
- Checked on simulated cards and on a CPU training run with a stand-in memory meter. Not yet run on a GPU by its
  authors.
- A failure in stage B no longer stops the program: the model trained in stage A is calibrated, scored and
  released. A progress line is printed after the first step and at least every five minutes.

Parishes.

- `pluvium now --country JM --parishes` prints one block per parish (province or state elsewhere): the typical place
  in it, and the range across its places. `--parish "St Ann"` prints the largest towns of one parish, or with
  `--parishes` its summary alone. `St`, `St.` and `Saint` are read alike.
- The forecast table has an `area` column after the place, holding its parish. GeoNames spells Jamaica's parishes
  three ways (`Portland Parish`, `St. Elizabeth`, `Clarendon`); they are written one way.

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
