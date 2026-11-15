# How Pluvium works

Pluvium forecasts rain for every town and village in the Caribbean, hour by hour, for three days. It answers the
questions people actually ask: *will it rain here, when does it start, will it be drizzle, showers, heavy rain or a
thunderstorm, how long will it last, how much will fall, and could it flood?*

This document explains the engine, what each part is for, how it is trained and checked, and how it compares with
the best-known AI rain models.

## 1. The problem, and why the global AI models are not enough

| System | What it does well | Where it falls short for Caribbean towns |
|---|---|---|
| GraphCast, GenCast, AIFS, FuXi (global AI weather models) | Medium-range large-scale weather, tracks of tropical cyclones | Trained on mean errors, they smear rain (the "double penalty" / spectral bias), under-forecast extremes, and misplace the daily cycle: GraphCast concentrates tropical rain too close to local midday. Their grid is 25 km or coarser. |
| MetNet-3 (Google) | 0-24 h rain at 1-2 km, learns from sparse station data | Built on dense radar truth over the USA and Europe; the Caribbean has almost no open radar. Not available for the region. |
| NowcastNet, DGMR, pySTEPS (radar nowcasting) | 0-3 h extrapolation of radar echoes; NowcastNet adds physics-conditioned evolution | Need radar; skill fades after a few hours. |
| CorrDiff and other downscalers | Sharp km-scale fields from coarse forecasts | Expensive; trained where high-resolution truth exists. |
| National models (GFS, ECMWF) | The physics of the whole atmosphere | 9-25 km grids, convection parameterised, hourly timing of island showers is weak. |

Pluvium is built for what the Caribbean has: coarse forecasts from the global models, satellites (CMORPH in
training, GOES-East live), about 250 airports reporting the weather every hour, and lightning from space.

## 2. The engine (NatureV1 / iHELIX)

```
 driving model fields (ERA5 in training; GFS, ECMWF IFS, AIFS live)   ->  coarse hourly + upper-air channels
 convective environment (K-index, lifted index, RH700, shear, ...)    ->  (version 2)
                    |  cross-attention, 12 nearest grid points
 Caribbean mesh: 6,075 points, ~11 km over land, ~80 km at sea, one frame per forecast step (43 frames)
   + rain already falling (satellite), sun, season, island-heating convergence, mountain (orographic) rain
   + airport reports of the last 3 hours, read in by cross-attention (version 2)
                    |
 iHELIX processor: 12 blocks of
   geodesic attention (each head its own radius, 15-500 km, distances on the sphere)
   index attention (content-addressed: "find the places that look like this one", anywhere)
   gated delta-rule recurrence along the forecast's frames (memory through time)
                    |
 decoder: every town (or 8 km cell) reads the mesh around it, with its own terrain, coast and climate
   + advected rain: the last hour's rain carried along the steering wind to this point (version 2)
                    |
 heads per hour:
   intensity chain   P(next state | this state) over dry / light / moderate / heavy (version 2; dry/wet in v1)
   amounts           lognormal within each intensity range (v1: a free lognormal mixture)
   thunder           chance of thunder (version 2)
   kind              drizzle / rain / showers (version 2)
   cause             cold front or shear line, tropical wave, tropical cyclone, afternoon storms, trade-wind showers
```

The engine is 57.9M parameters at the Pluvium-500 size. "500" names the model, not its parameter count.

### Why a chain of intensity states

Most AI rain models output a separate number for every hour. Pluvium's network writes, for every hour and town,
a 4 x 4 transition matrix: from dry, light, moderate or heavy rain now to each of them in the next hour. One chain
then gives every answer consistently:

* the chance of rain in each hour (its marginal distribution);
* the hour rain is most likely to **start** (first entry into a wet state from dry), with a window;
* the hour it turns **heavy**, and how long rain will **last** (first return to dry);
* the chance of **any** rain, or any heavy rain, in the next 6, 12, 24, 48 or 72 hours (computed exactly, not
  assuming the hours are independent);
* **totals** and the wettest 1, 3 and 6 hours, by sampling paths of the chain with amounts from each state.

### Training through the instruments' eyes

Neither truth source is perfect. The satellite smears a downpour over an 8 km box and misses light showers; an
airport sees one spot. So training scores the observed sequence of classes through a hidden Markov model in which
each source has its own learned **confusion matrix** (from the true state to the class it reports). A satellite
miss is not taught to the model as "it was dry". The forward algorithm is exact and differentiable; it is checked
against brute-force enumeration in the tests.

### Physics given, not learned

* **Orographic rain** (Smith-Barstad linear theory) solved by FFT on 2 km tiles for every forecast hour, with its
  time scales and moisture parameters learned: rain on windward slopes, shadow in the lee.
* **Island-heating convergence**: where the sea breezes meet over each island in the afternoon, carried downwind.
* **Fronts and shear lines**, with arrival times at every point.
* **Convective indices** (version 2): K-index, total totals, theta-e difference, lifted index, 700 hPa humidity
  (the Saharan air layer chokes summer showers), moisture-flux convergence, 850-200 hPa shear and 200 hPa
  divergence.
* **Advection** (version 2): for the first 12 hours, the rain falling now is carried along the steering wind, the
  core idea of radar nowcasting, done here with satellite rain because the region has little open radar.

## 3. Version 3: deeper physics, a deeper engine, and training for rare events

Version 3 (the default; `-v2` and `-v1` rebuild the earlier versions for comparison) adds four layers of
depth, each aimed at a specific weakness.

**Theoretical physics as inputs** (`pluvium/atmos/`):

* exact moist thermodynamics (Bolton 1980) and the CAPE, CIN and LCL of an 850 hPa parcel;
* column relative humidity and the exponential law of tropical rain (Bretherton et al. 2004), with
  vertically integrated moisture-flux convergence;
* the Saharan air layer index (dry, warm mid-levels with an easterly jet);
* tropical-wave trough axes at 700 hPa and their arrival times;
* cyclone detection with an R-CLIPER-shaped rain prior and shear asymmetry;
* the sea-breeze front's passage from gravity-current theory;
* the land and sea diurnal clocks;
* forced ascent at two terrain scales, and the Froude number for blocking.

**A deeper decoder** (`pluvium/engine/`):

* a terrain-patch encoder reads 35 km of elevation and coastline around every point;
* lead-time FiLM, and learned shrinkage of the transitions towards climatology as lead grows;
* attention along each point's 40 future frames;
* attention among neighbouring decoded points;
* a regime mixture of experts;
* head dropout sampled at forecast time for epistemic uncertainty.

**Training for what matters** (`pluvium/training/`):

* a rare-event curriculum that draws heavy-rain and thunder windows several times as often, with tempered
  importance weights;
* averaged (EMA) weights;
* physical augmentation: satellite gaps, airport report errors, driving-model moisture bias;
* distillation into small models.

**Probability and hydrology** (`pluvium/prob/`, `pluvium/hydro/`):

* a spatial copula for coherent area and catchment samples;
* conformal intervals for totals;
* generalised Pareto tails and return periods;
* Brier decomposition, ROC, ETS, SEDI and the fractions skill score;
* antecedent wetness, curve-number runoff and unit-hydrograph flood timing.

**Operations** (`pluvium/guard/`, `pluvium/ops/`, `pluvium/render/`):

* input repair, freshness checks, drift detection and output repair;
* a statistical fallback;
* an hourly operational cycle with atomic publishing and a JSON API;
* maps, bulletins in English and Jamaican Patois, and CAP 1.2 alerts.

Every module and its purpose is listed in [MODULES.md](MODULES.md). What can go wrong in operation, and what the
system does about it, is in [FAILURE_MODES.md](FAILURE_MODES.md).

## 3a. Pluvium Blend: every operational system in one forecast

Pluvium's network reads one driving model at a time. Pluvium Blend reads all six operational systems at once:
ECMWF ENS, NOAA GEFS, ECMWF HRES, ECMWF AIFS, NOAA GraphCast and NOAA GFS. Where it runs, it also reads Pluvium's
network. It then writes one calibrated forecast of the day's rain for every place, out to 14 days.

It is built in two layers.

1. **Components.** A small network turns forecasts into the chance of at least 0.2, 0.5, 1, 2 ... 150 mm. The chances
   fall with the threshold by construction, so they never contradict each other. One copy is trained on each system
   alone, which makes that system's post-processed forecast. Another copy reads every system at once.
   * **Inputs per system and place:** the members' mean and spread, the chances of 1, 10 and 25 mm, and the same over
     everything within half a degree. A `present` flag marks a system that is missing (GraphCast stops at day 10).
   * **Context:** the climatology of the place and month, and the lead.
   * **Training:** the sum of the Brier scores at the thresholds. Integrated over thresholds, the Brier score is the
     CRPS, so one fit serves yes/no questions, amounts and the whole distribution.
2. **The pool** (`StackPool`). It averages the components' log-odds with one weight per component, per band of
   thresholds (light, moderate, heavy) and changing with lead. It then sharpens by the weights' total and adds an
   offset per threshold. A missing component simply drops out.

The pool has about 70 parameters. They are fitted on forecasts each component made for weeks it was not trained on:
two folds of alternate weeks, split by the day forecast, so no component is ever judged on truth it learned from. A
pool this small cannot overfit, and it can always fall back to its best component.

Why two layers: in validation inside the training year, one network over all six systems at once was no better than
the best single system, because it learned combinations that did not carry over to a new season. The pool was better
than that network in about 20 of 24 comparisons. Bagging, hiding whole systems during training, and separate networks
for short and long leads were tried and gave mixed or worse results. See the full-year benchmark for the record.

**A fair rival for every system:** each system's component is that system's post-processed forecast, scored on its
own. Pluvium Blend has to beat each system after equal treatment as well as its raw output.

Training a blend for a new country needs a year of archived runs and satellite truth: about two hours of downloads
and under an hour of CPU (`pluvium blend train`).

### The second generation (version 0.3)

The first full-year benchmark found three weak spots: the heavy tail, winter in mid-latitude climates, and
structure at long range. `pluvium blend train` now fits a design aimed at them (`pluvium.frontier.FrontierBlend`).
Each part was chosen inside the training year and is scored on its own in
[FRONTIER_BENCHMARK.md](FRONTIER_BENCHMARK.md).

1. **Components see every heavy day** (`pluvium.tails.stratify`). A network trained on a uniform sample of rows sees
   few days with 50 mm. Each component is now trained on every row with observed or forecast rain of 25 mm or more,
   with the remaining rows thinned. Each row carries a weight equal to one over its chance of being kept, so the
   chances the network learns are still right for an ordinary day.
2. **The pool is fitted on the same rows** (`pluvium.pool.gated`). It keeps its form: weights per component and band
   of thresholds, changing with lead.
3. **Online weights** (`pluvium.pool.online`). The components and the pool are treated as experts. After every day
   whose rain is known, each expert's weight moves by how much better or worse it did than the mixture, separately
   for four lead groups and three bands of thresholds. A small share of weight always returns to the starting
   weights, so no expert is written off for good. This is how the forecast follows a model upgrade or a season in
   which one system pulls ahead. The weights are part of the saved model and move with `pluvium blend learn`. The
   benchmark gave the weights each day's rain 48 hours later. Live, the rain is known when NOAA publishes it, about
   four and a half months later, so the weights follow that far behind; the benchmark page shows what the delay
   costs.
4. **Record storms** (`pluvium.tails.support`, `pluvium.tails.override`). Calibration learned on ordinary years pulls
   a record forecast towards normal. The model keeps the edge of what it was trained on: the 99.9th percentile of the
   raw systems' consensus amount. Inside that edge nothing changes. Beyond it, trust moves from the calibrated chances
   to the raw systems' own, in proportion to how far outside the forecast is and how many systems agree.
5. **Each place decides about the hand-over** (`pluvium.tails.withheld`). Handing over helps where calibration
   shrinks a storm the raw systems had right. It hurts where the raw systems call for heavy rain far more often than
   it falls. Which case holds is tested at every training by staging record storms: the four wettest days of the
   training period are withheld, a forecast is trained without them, and they are then forecast with and without the
   hand-over. The hand-over is used only if it lowered the Brier score at 50 mm on those storms. The verdict is
   printed and saved with the model, and the owner can overrule it (`pluvium blend train --handover always` or
   `never`). This step was added after the first places of the benchmark showed that handing over everywhere hurts
   in some of them. A first remedy, an audit of the training period's ordinary days, proved unreliable and was
   dropped as the switch (`pluvium.tails.audit`, still reported). The benchmark tells both stories.
6. **Scenarios** (`pluvium.scenarios`). A calibrated chance per place is smooth at long range, as it should be. For
   maps and area totals, the calibrated amounts at each place are arranged across the scenarios in the same order as
   the ECMWF ensemble's members (ensemble copula coupling). Each scenario then has the right amounts place by place
   and the ensemble's arrangement in space.
7. **Words for the heavy end** (`pluvium.tails.alerts`, `pluvium.tails.returns`). The forecast table gives an alert
   tier for 50 mm (watch at a 10% chance, warning at 30%, extreme at 60%), flags a forecast that lies beyond the
   training record, and says how rare the expected amount is at that place ("about once in 6 years"), from a
   generalised Pareto fit to the area's daily record. Beyond four times the length of that record it says only
   "rarer than once in that many years".

What was tried and dropped is in [WHAT_DID_NOT_WORK.md](WHAT_DID_NOT_WORK.md).

### One command, shared blends and the upstream weights (version 0.4)

`pluvium now --country XX` (`pluvium.quick`) strings the steps together and does only what is missing: it sets the
country up, gets a trained blend, reads the newest runs and writes the forecast. A trained blend is small (a few
hundred kilobytes), so one can be shared. `pluvium blend publish` uploads it to a Hub repository under
`blends/<area>/`, and `pluvium now` looks there before it trains. A downloaded blend is used only if it is for the
same box and opens with a reader that cannot run code (`pluvium.blendlive.read_state`); blend files are written as
tensors and plain values for that reason.

The training period is no longer fixed. A blend learns from the rain that fell after each forecast, and the
satellite record arrives about four and a half months late, so `pluvium.quick.recent_year` asks NOAA's bucket how
far the record goes (`pluvium.sources.cmorph_daily.truth_until`) and takes the year that ends 15 days before that.

Two of the six systems are AI models with public weights: NOAA's run of GraphCast and ECMWF's AIFS.
`pluvium.upstream` lists where each publisher keeps them and under which licence, copies them unchanged with their
checksums, and can keep the copies in a Hub repository under `upstream/`. Pluvium does not run them. It reads the
forecasts NOAA and ECMWF make with them, because running them needs a large GPU and that day's global analysis,
which only those centres produce. The copies guard against a different risk: a feed being renamed or switched off,
as NOAA's GraphCastGFS was in May 2026 (the reader now follows it to its successor, AIGFS).

## 3b. Any country

An `Area` (`pluvium/area.py`) says where a deployment works:
* its countries (ISO codes);
* where forecasts are made (their outline from geoBoundaries, plus a margin);
* where the mesh and the driving fields reach (six degrees more on every side, so weather can arrive);
* the bulletin languages;
* the satellite that sees it.

Geographies save their area and their own satellite-rain window. Windows work in both hemispheres and across
longitude 0. `pluvium setup --country XX` builds everything for a country.

## 3c. Live data

| Feed | Used for |
|---|---|
| GFS (NOAA), ECMWF IFS and AIFS open data | The driving fields; one forecast per model, mixed (disagreement widens the windows) |
| GFS ensemble (GEFS, 31 members at 0.5 degrees, optional: `--ensemble 31`) | More members in the mixture: the spread of the start window and the totals then reflects the atmosphere's own uncertainty; the members together weigh as much as two deterministic models |
| GOES-East rain rate (or infrared when that product is unavailable) | Rain falling now, and the field that is advected |
| MRMS Caribbean radar (NOAA, 1 km every 2 minutes, Puerto Rico and the Virgin Islands) | Rain now where radar reaches, replacing the satellite estimate |
| METAR from ~80 Caribbean airports (aviationweather.gov) | The last three hours of weather read into the mesh; the truth that online learning checks forecasts against |
| GOES-East Geostationary Lightning Mapper | Lightning within 10 km of each town in the last 15 minutes |

## 3d. Pluvium Cyclone: where a hurricane goes

`pluvium cyclone` forecasts the track of every tropical cyclone in the North Atlantic and the eastern North Pacific
from the same six operational systems. It forecasts track only.

1. **Decks** (`pluvium.cyclone.atcf`, `storms`). The National Hurricane Center's public decks say which storms exist
   and where each was at every hour, as known at the time.
2. **Fields** (`fields`). Sea-level pressure every 12 hours to 120 hours, from every member of ECMWF ENS and NOAA
   GEFS and from ECMWF HRES, ECMWF AIFS, NOAA GFS and NOAA GraphCast.
3. **Tracker** (`detect`, `tracker`, `tracks`). From the real-time position, each member is followed: the nearest
   closed low near where the storm should be, refined below the grid spacing. A member that loses the storm stops.
4. **Consensus that learns** (`minvar`, `learner`, `online`, `consensus`). The six tracks are averaged with weights
   that minimise the expected error given the systems' error covariance so far. The covariance is updated after
   every forecast that has verified, against the position known in real time. A new system starts as an average
   member. Within a storm, a system that has done badly on that storm counts for less.
5. **Early guidance** (`interp`). The newest complete run is 12 hours old when an advisory is written. Its track is
   read 12 hours further along and slid to the observed position.
6. **Uncertainty** (`cone`, `cloud`, `approach`, `clusters`). A circle per lead that should hold the centre two times
   in three, learned from verified errors and widened when the systems disagree. A cloud of tracks: every system's
   track with the ensembles' perturbations, weighted, centred on the consensus and sized by verified errors. From
   the cloud: the chance of the centre passing within 150 km of each listed place, and scenarios when the tracks
   split.
7. **Live** (`live`, `report`). Each call replays the last ten days of runs on top of what was learned in 2024 and
   2025, which ships with the package, and writes a bulletin and a data file per storm.

How it scored against the National Hurricane Center's guidance is in [CYCLONE_BENCHMARK.md](CYCLONE_BENCHMARK.md).

## 4. Products

Per town: the chance of rain each hour; words for each hour ("light drizzle", "moderate showers", "heavy rain",
"thunderstorms", "torrential rain"); the likely start time and window in local time; the heaviest hour; thunder
chance; chance of any rain and of heavy rain in each window; median and 90th-percentile totals; the wettest
3 hours; **flood risk** with its reason (rain routed through the town's river catchment from a D8 flow network on
the terrain, plus slope; a rule, not yet verified against flood reports); and the cause. Per parish: a headline.
Formats: CSV, GeoJSON, JSON and NPZ, and a text report.

## 5. Learning from mistakes

Every forecast's hourly chances at the airports are archived. `pluvium learn` reads what the airports reported
since and updates an online corrector per parish, lead time and part of the day (logistic recalibration and an
amount multiplier). The corrector is pulled back towards "no correction" at every step, so it needs repeated
evidence to move and relaxes when a pattern ends. A parish without an airport borrows its country's correction.
The network's weights are retrained by the one-GPU program, which resumes from the Hugging Face Hub.

## 6. How skill is measured

Every score is computed the same way for Pluvium and for three references: climatology (that place, month and
hour), persistence, and the driving model's own rain. Scores include the Brier score of hourly chances; the
critical success index; skill for any rain in 6/12/24 hours; the error of the predicted start hour; the ranked
probability score over intensity classes; the Brier score for heavy hours; CRPS of 24-hour totals; and thunder at
airports against each airport's thunder climatology. Calibration is fitted on one half of the held-out period and
scores are reported on the other half. They are reported with the reanalysis as a perfect forecast (an upper
bound) and with emulated forecast errors (closer to live).

`pluvium bench` trains competing architectures from scratch on identical data with identical budgets and reports
all of these side by side.

`pluvium operational` is the stricter test. It puts Pluvium against the operational systems: ECMWF ENS, ECMWF HRES,
ECMWF AIFS, NOAA GEFS, NOAA GraphCast and NOAA GFS. Only real archived runs are used, with no reanalysis anywhere.
Pluvium is driven by that morning's GFS run, after fine-tuning on earlier GFS runs. Every system is calibrated the
same causal way, on the 30 runs before each test day. Results are in [OPERATIONAL_BENCHMARK.md](OPERATIONAL_BENCHMARK.md)
and the version comparison is in [BENCHMARK.md](BENCHMARK.md).

`python -m pluvium.benchmarks.fullyear` is the widest test: Pluvium Blend against the same six systems for a full year
(April 2025 to March 2026, every season), days 1 to 14.
* **Places:** six regions on the systems' own 0.25-degree grid (tropics, mid-latitude mountains, coasts and plains, both
  hemispheres), and Jamaica at 8 km with Pluvium's network.
* **Equal treatment:** every system is scored raw and post-processed by the same network trained on it alone.
* **Training:** quarterly refits on everything verified by then.
* **Scores:** Brier skill at 1, 10, 25 and 50 mm against the training year's climatology, CSI, ROC area, CRPS, RMSE,
  anomaly correlation and sharpness.
* **Uncertainty:** a block bootstrap over runs against each system.

Results are in [FULL_YEAR_BENCHMARK.md](FULL_YEAR_BENCHMARK.md).

`python -m pluvium.benchmarks.frontier_bench` scores the second generation on the same year, places and rules, and
adds three tests of its own:
* **the tail**: the threshold-weighted CRPS above 25 mm against the raw consensus, and what each alert tier meant
  (how often it was issued, how often the rain came, what share of heavy days it caught);
* **scenarios**: coupled scenarios against the raw ECMWF ensemble and against the same amounts with no arrangement,
  by the CRPS of rain averaged over two-degree boxes, the energy score and the variogram score;
* **winter against a second truth**: the two US places again for December to February against MRMS, the US radar
  and gauge analysis.

Every part of the design is scored with and without the others, and the design as first written down is scored
beside the final one. Results are in [FRONTIER_BENCHMARK.md](FRONTIER_BENCHMARK.md).

`python -m pluvium.benchmarks.cyclone_bench` scores hurricane tracks against the National Hurricane Center's best
tracks, beside the Center's own official forecast and guidance, on the forecasts all of them made. Settings were
chosen on three seasons and tested once on a fourth. Results are in [CYCLONE_BENCHMARK.md](CYCLONE_BENCHMARK.md).

## 7. Honest limits

* Truth is satellite rain at 8 km and point reports at airports; neither sees every shower.
* Live "rain now" comes from GOES infrared when the rain-rate product is unavailable, which is cruder than the
  CMORPH used in training.
* Tropical cyclones are rare in any training period.
* Pluvium Blend does not lead everywhere. Version 0.3 narrowed the gap at 50 mm a day but did not close it in every
  place. Beyond a week in the mid-latitudes raw models keep a higher anomaly correlation of the expected amount. The
  current list, from the data, is in [FRONTIER_BENCHMARK.md](FRONTIER_BENCHMARK.md#what-is-still-weak).
* The year version 0.3 was tested on is the year its weak spots were found on. Its design was chosen inside the
  training year, and one step (each place deciding about the hand-over) was settled after five of the seven places
  had been scored.
* Where a place's staged storms switch the record-storm hand-over off, a storm beyond the training record is
  forecast by the calibration alone, which shrinks it. The verdict rests on four storms and can be wrong.
* Pluvium Cyclone forecasts the track only: no intensity, size, wind, surge or rain, and no formation. On its clean
  test season it was behind the National Hurricane Center's official forecast at every lead.
* Pluvium forecasts rain and cyclone tracks: not temperature, wind or heat waves, and not the poles (its truth stops
  at 60 degrees of latitude).
* The flood rule needs verification against flood reports before anyone relies on it.
* A research model: not an official forecast.
