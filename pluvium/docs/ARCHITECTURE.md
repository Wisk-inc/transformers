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

## 3b. Live data

| Feed | Used for |
|---|---|
| GFS (NOAA), ECMWF IFS and AIFS open data | The driving fields; one forecast per model, mixed (disagreement widens the windows) |
| GFS ensemble (GEFS, 31 members at 0.5 degrees, optional: `--ensemble 31`) | More members in the mixture: the spread of the start window and the totals then reflects the atmosphere's own uncertainty; the members together weigh as much as two deterministic models |
| GOES-East rain rate (or infrared when that product is unavailable) | Rain falling now, and the field that is advected |
| MRMS Caribbean radar (NOAA, 1 km every 2 minutes, Puerto Rico and the Virgin Islands) | Rain now where radar reaches, replacing the satellite estimate |
| METAR from ~80 Caribbean airports (aviationweather.gov) | The last three hours of weather read into the mesh; the truth that online learning checks forecasts against |
| GOES-East Geostationary Lightning Mapper | Lightning within 10 km of each town in the last 15 minutes |

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

## 7. Honest limits

* Truth is satellite rain at 8 km and point reports at airports; neither sees every shower.
* Live "rain now" comes from GOES infrared when the rain-rate product is unavailable, which is cruder than the
  CMORPH used in training.
* Tropical cyclones are rare in any training period.
* The flood rule needs verification against flood reports before anyone relies on it.
* A research model: not an official forecast.
