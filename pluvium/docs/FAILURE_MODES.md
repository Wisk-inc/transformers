# Failure modes: what can go wrong when Pluvium runs, and what it does about it

A forecast system fails in operation in ways no benchmark shows. This table lists the failures that matter for
Pluvium. For each it gives how the failure is detected and what the system does, with the module responsible.

## Inputs

| Failure | Detected by | What happens | Module |
|---|---|---|---|
| A driving model changes units (Celsius for Kelvin, hPa for Pa, percent for fraction, g/kg for kg/kg) | Medians outside physical ranges | Converted; noted in the forecast | `guard/inputs.py` |
| A field is missing from a run | All-NaN field | Physical default and a note. If water vapour is missing, it is derived from humidity at 850/700/500 hPa | `guard/inputs.py`, `sources/nwp.py` |
| Missing or spiked frames in a run | Whole-frame gaps; a frame far from both neighbours | Interpolated in time or replaced by the neighbours' mean | `guard/inputs.py` |
| A frozen feed (the same field every hour) | Identical future frames | Flagged in the notes | `guard/inputs.py` |
| A stale driving run | Run age over 18 h | Noted; the newest usable run is always preferred | `forecast.py`, `guard/freshness.py` |
| The satellite rain-rate product is paused (as GOES RRQPE has been since late 2026) | No recent scene | Falls back to the infrared auto-estimator; the product used is named in the forecast | `sources/goes.py` |
| Satellite gaps (missing scans, sun glint) | NaN regions | The model was trained with simulated gaps, so it tolerates them | `training/augment.py` |
| Radar out of range or down | NaN outside coverage | Satellite used there; radar used only where it reaches | `sources/mrms.py` |
| Airport sensors missing (`////` in a METAR) | Slash groups | The hour is "unknown", never "dry" | `weather.py` |
| An airport report from the wrong hour | Report time minus 5 minutes defines the hour | Reports at 14:00 describe hour 13 | `sources/metar.py`, `sources/stations.py` |
| No airport reports at all | Empty METAR response | The model was trained with random whole-network dropout of reports | `dataset.py` |
| Clock or feed time errors | Observation times in the future or too old | Not used as current; noted | `guard/freshness.py` |
| Memory exhausted by global fields (three models at 0.25 degrees hold thousands of 4 MB fields) | Found in a live test, killed at 9.4 GB | Every field is cut to the domain as soon as it is decoded; a run now peaks at about 0.3 GB | `sources/nwp.py` |
| An archive throttles bursts (ECMWF's AWS bucket answers with HTTP 503) | Retries exhausted | ECMWF runs are read from Google Cloud's copy first, AWS second | `sources/baselines.py` |
| AWS answers requests for missing files with HTTP 503, not 404 | A path probe retried for seconds per step | A 404 from Google Cloud is final for probes (AIFS: 0.04 to 0.8 runs a second) | `sources/baselines.py`, `sources/archive.py` |
| A producer changes its publishing path (AIFS: experimental to AIFS Single, February 2025) | No rain messages under either path for a few days | Both paths are tried; runs in the gap are missing, and Pluvium Blend forecasts from the other systems (`present` flags) | `sources/archive.py`, `blend.py` |
| No geostationary rain product for an area (outside GOES-East and GOES-West) | The area's satellite is `none` | Forecast made without rain now, as a third of training forecasts were | `area.py`, `forecast.py` |
| An area crossing longitude 0 or the date line | Box check | Longitude 0 is handled (windows wrap); the date line is refused with a clear message | `domain.py` |

## The model

| Failure | Detected by | What happens | Module |
|---|---|---|---|
| A situation unlike training: a record hurricane, a new driving-model version with a different bias | Input channels outside their training range | The forecast is pulled towards climatology in proportion; the channels responsible are named | `guard/drift.py` |
| Overconfidence where data were thin | Dropout before the heads, sampled at forecast time | The spread of the samples widens the windows and totals | `engine/mc_heads.py` |
| Rare heavy rain under-learned | Version 2's heavy-hour score | Rare-event curriculum with importance weights. Physics priors: CAPE, column humidity, cyclone rain. GPD tails | `training/curriculum.py`, `atmos/`, `prob/extremes.py` |
| Cyclone rain never seen at a place | Too few storms in any training period | A parametric storm-rain prior with shear asymmetry is an input | `atmos/cyclones.py` |
| Wrong hour for afternoon storms (a known weakness of global AI models) | Stratified errors by local time | Diurnal clocks, sea-breeze front timing and island-heating convergence are inputs | `atmos/diurnal.py`, `atmos/sea_breeze.py` |
| Hour-to-hour jitter in start times | Start-time error | Attention along each point's future, then one chain per place | `engine/temporal_mixer.py`, `chain.py` |
| Neighbouring villages with contradicting hours | Neighbour attention | Coherent fields; correlated sampling for areas | `engine/refiner.py`, `prob/copula.py` |
| Overconfident long range | Lead-time scores | Learned shrinkage of transitions towards climatology with lead | `engine/lead_film.py` |
| One network over many systems learns combinations that do not carry over to a new season | Validation inside the training year: it lost to the best single system | Pluvium Blend pools each system's own post-processed forecast with about 70 weights fitted on weeks the components did not train on; it falls back towards its best component | `blend.py` (`StackPool`), `benchmarks/fullyear.py` |
| An operational run is late or missing in daily use | No archive file for the newest run | The previous run is used at the matching lead; a system missing altogether drops out of the pool and the others are re-weighted | `blendlive.py` |
| A storm larger than anything in the training period: calibration shrinks it (Hurricane Melissa: 66% against 83 to 100% from every raw system) | The raw consensus amount lies beyond the 99.9th percentile of the training forecasts | The forecast is handed back, in part, to what the raw systems say, in proportion to how far outside it lies and how many systems agree. The table flags it as beyond the record | `tails/support.py`, `tails/override.py`, `frontier/model.py` |
| The raw systems overcall heavy rain in a place, so handing over to them does harm | Storms staged at every training: the four wettest days are withheld, forecast without having been seen, and scored with and without the hand-over | Where the hand-over did not improve the 50 mm Brier score on them it is switched off for that place. The verdict is printed and saved with the model, and `--handover always` or `never` overrules it | `tails/withheld.py` |
| A system gets better or worse during the year (an upgrade, a season it handles badly) | Each expert's Brier score on every verified day | Online weights move towards whoever has been right, per lead group and band of thresholds; a small share always returns to the prior so nobody is written off | `pool/online.py`, `pluvium blend learn` |
| The truth for a day arrives late | A fixed delay | A day is learned from 48 hours after it ends, never earlier | `frontier/model.py` |
| Heavy rain is rare, so a uniform sample learns it from noise | Counts of 50 mm rows | Every heavy row is kept in training and the rest thinned, with weights that keep the chances unbiased | `tails/stratify.py` |
| A tropical cyclone is lost by a model's member (it weakens, merges, or leaves the grid) | No closed low near the expected position | That member's track stops there; an ensemble's mean position is used only where 40% of members still carry the storm | `cyclone/tracker.py`, `cyclone/consensus.py` |
| One system's pressure fields arrive after the others | Systems missing from a cached case | The case is re-tracked for the systems that have since arrived | `cyclone/tracks.py` |
| Hurricane consensus weights from past seasons age as models are upgraded | Errors of every forecast that has verified | Weights are learned through the season from the running error covariance, starting from a shipped prior and replaying the last ten days | `cyclone/learner.py`, `cyclone/minvar.py`, `cyclone/live.py` |
| The network cannot run (bad checkpoint, no GPU, broken inputs) | Any exception in the forecast | A statistical fallback forecast is issued, labelled as such | `guard/fallback.py`, `forecast.py` |
| Satellite truth under-reports downpours | Learned confusion matrices | Training never takes an instrument's word as truth | `losses.py`, `chain.py` |

## Outputs

| Failure | Detected by | What happens | Module |
|---|---|---|---|
| Probabilities that contradict each other (6 h more likely than 12 h, heavy above rain) | Consistency checks | Repaired before publishing; repairs counted in the notes | `guard/consistency.py` |
| Miscalibrated chances | Held-out reliability | Isotonic calibration of rain, moderate-or-heavy and heavy, per lead band; online recalibration per parish from airport reports | `calibrate.py`, `online.py` |
| Intervals for totals that do not cover | Held-out coverage | Conformal margins | `prob/conformal.py` |
| A model that drifts as seasons change | Online verification against airports | Corrections pulled back towards none unless errors repeat | `online.py` |
| A half-written forecast read by a user | Atomic publish | A new folder becomes `latest` only when complete | `ops/loop.py` |
| Flood risk taken as a warning | Explicit labelling | Flood levels are a rule, not verified against flood reports. CAP alerts have status "Exercise" | `flood.py`, `render/alerts.py` |
| A smooth long-range map taken for a picture of the day | Sharpness of the expected amount falls with lead | Scenarios: calibrated amounts arranged like the ECMWF ensemble's members, each as varied as weather | `scenarios/ecc.py` |
| A return period quoted far beyond what the record can support | The record's length | Beyond four times the record the wording is only "rarer than once in N years" | `tails/returns.py` |
| A hurricane track read as a line | The spread of the systems and their members | A circle sized to hold the centre two times in three, and the chance of the centre passing within 150 km of each place | `cyclone/cone.py`, `cyclone/approach.py` |

## Known limits that are not handled

* Truth is satellite rain at 8 km and airport reports. Neither sees every shower, and mountain rain is
  under-reported by the satellite.
* The flood rule and its thresholds are not verified against flood reports.
* Driving models' own biases in a new season are only corrected as fast as airport reports reveal them.
* Live "rain now" outside radar coverage comes from GOES infrared, which is cruder than the CMORPH used in training.
* Where a place's staged storms switch the record-storm hand-over off, a storm beyond the training record is
  forecast by the calibration alone, which shrinks it. The table still flags it and counts the raw systems that
  call for 50 mm. The verdict rests on four storms of one training period and can be wrong either way.
* Pluvium Cyclone forecasts the track only: no intensity, size, wind, surge or rain. It follows storms that already
  exist and are named in the National Hurricane Center's decks; it does not forecast formation.
