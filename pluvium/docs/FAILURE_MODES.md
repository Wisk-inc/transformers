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

## Known limits that are not handled

* Truth is satellite rain at 8 km and airport reports. Neither sees every shower, and mountain rain is
  under-reported by the satellite.
* The flood rule and its thresholds are not verified against flood reports.
* Driving models' own biases in a new season are only corrected as fast as airport reports reveal them.
* Live "rain now" outside radar coverage comes from GOES infrared, which is cruder than the CMORPH used in training.
