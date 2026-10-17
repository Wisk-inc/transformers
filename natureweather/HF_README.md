---
license: apache-2.0
language:
  - en
library_name: pytorch
tags:
  - weather
  - weather-forecasting
  - hurricane
  - tropical-cyclone
  - era5
  - weatherbench2
  - hurdat2
  - geodesic-attention
  - physics-informed
  - probabilistic-forecasting
  - ihelix
datasets:
  - weatherbench2/era5
  - hurdat2
pipeline_tag: other
---

# NatureV1 — a probabilistic weather and hurricane model built on geometric attention

> **Status: trained and scored — a research model, not an operational forecast.** The first full run is complete
> and every number in [section 4](#4-results) was measured on data the model never trained on, including an
> end-to-end test on the 2023 hurricane season (a strong El Niño year no part of the model ever saw). It beats
> physics-only steering and straight-line extrapolation on hurricane track at every lead, but it is several times
> less accurate than the National Hurricane Center and the leading global models, it misses rapid intensification,
> and a red-team pass (section 4.8) found failure modes you should know before using any output. **Do not use it
> for real decisions** — for those, use [the National Hurricane Center](https://www.nhc.noaa.gov).
>
> **Training took about one week** of wall-clock time on a single GPU in a hosted notebook (molab), including
> session restarts and re-staging; the rollout stage alone was 74 GPU-hours.

NatureV1 is an 89-million-parameter model of the atmosphere that forecasts global weather out to five days and,
for tropical cyclones, where a storm will go, how strong it will get, whether it will intensify rapidly, and
whether it will make landfall. Every number it produces comes with an uncertainty. It is built on
[iHELIX](https://pypi.org/project/ihelix/), an attention engine that works on the physical geometry of the planet
rather than on a flat grid, and trained by a single, resumable, self-verifying pipeline that lives in one notebook
cell. The whole stack — engine, model, training pipeline, verification — is open and written by one person.

- Code and training pipeline: `pip install naturev1[all]` — [naturev1 on PyPI](https://pypi.org/project/naturev1/)
- Engine: `pip install ihelix` — [ihelix on PyPI](https://pypi.org/project/ihelix/)
- Author: Nathan. License: Apache-2.0.

---

## 1. What it does

Given six frames of the atmosphere (the last 30 hours of ERA5 reanalysis: 89 channels — 11 surface fields and 6
variables on 13 pressure levels — on a 1.5° global grid), NatureV1 produces:

**Global weather, at 6, 12, 18, 24, 36, 48, 72, 96 and 120 hours**
- A **full-state rollout**: the whole atmosphere — all 86 prognostic channels, surface to 50 hPa — stepped forward
  6 hours at a time on its own forecast, with mass conservation, out to 72 hours, as an ensemble. **This is the
  weather output with measured skill** (section 4.2).
- 2 m temperature, mean sea-level pressure, 10 m wind and precipitation directly at each lead, as a mean and a 90%
  interval. These direct surface heads were not updated after the rollout stage and have no skill in this release;
  use the rollout.
- A weather-type category head exists but was never trained, so it is never shown.

**For a tropical cyclone, given its current centre**
- **Track as scenarios, not a line**: six weighted trajectories ("38% recurves offshore, 27% tracks the coast…"),
  each with a tilting uncertainty ellipse per lead, so the cone of uncertainty is drawn from the model, not
  from a lookup table.
- A **multi-model consensus track**: the learned track, a physics-only steering-flow track, and straight
  extrapolation, blended with per-lead weights fitted on held-out seasons — the way the National Hurricane Center's
  consensus beats every single model in it, done inside one model. Every member is reported alongside the blend.
- **Intensity**: maximum wind and minimum central pressure per lead, with intervals and the Saffir–Simpson category.
- **Rapid intensification**: the probability of a 25, 30 and 35 kt gain in 24 hours, plus the expected change.
  It ranks risk well but its probabilities are far too high (section 4.8).
- **Landfall**: the probability of landfall by each lead. (The landfall-timing head was never trained and is not
  shown.)
- **Core structure**: peak eyewall wind, radius of maximum wind, and 34/50/64 kt wind radii by quadrant.
- **The storm's environment**, read from the model's own atmosphere: deep-layer steering, 200–850 hPa shear,
  mid-level humidity, ocean temperature — the predictors operational statistical schemes (SHIPS) are built on.

**A local forecast** for any point on Earth, in physical units, from the gridded fields.

### What it refuses to do
This is as much a part of the model as the outputs above.
- An output whose head has **never received a training step is never shown**. It is replaced by
  `{"available": false, "reason": ...}`. The training record travels inside the checkpoint.
- A forecast run from an input path the model was never trained on (today, the satellite path) is marked
  **`trustworthy: false`** and the live satellite forecast is refused outright.
- `check_forecast` tests every scenario track, intensity, eyewall and RI value against **physical bounds**: no
  400 kt hurricane, no pressure outside 850–1100 hPa, no cone wider than an ocean, no probabilities that do not sum
  to one. It reports problems; it does not block the forecast, so a caller must act on what it returns. Section
  4.8 lists what it does not catch.
- The pipeline publishes only if the model beats persistence somewhere on held-out data and demonstrably changes
  its answer when the weather changes. That gate is weak (precipitation alone can pass it); this release was
  published after the full scorecards and red-team below were reviewed.

---

## 2. How it works

### 2.1 The engine: iHELIX, attention over physical space
Most weather transformers flatten the globe into tokens and let attention learn geography from scratch. iHELIX
keeps the geometry:

- **Geometry axes.** A domain is composed from spherical, periodic, linear and logarithmic axes. `Geometry.globe()`
  is the Earth's surface; `Geometry.atmosphere()` is a spherical shell whose vertical axis is log-pressure, so that
  1000→500 hPa is the same distance as 500→250 hPa. Every point carries a local east/north/up frame, so "20 km
  north" means the same thing everywhere.
- **FieldGrid: data, not architecture.** The points data lives on — ERA5's 240×121 grid, a 4,096-point Fibonacci
  mesh, a satellite scan's pixel footprints — with true area weights. The same model reads any of them.
- **GeodesicAttention.** Every attention head has a physical radius in kilometres with compact support: heads
  from 120 km (fronts, convection) to 1,600 km (synoptic scale). Attention is over great-circle distance.
- **IndexAttention.** A region tree with beam search lets a point reach distant points (teleconnections) without
  paying for the whole globe.
- **TemporalStrand.** A gated delta-rule recurrence carries state across the six input frames at linear cost —
  inherited from HELIX, a linear-context language model this engine grew out of.
- **Encode → process → decode.** Any source grid is read onto a shared 4,096-point mesh, processed by 17 blocks,
  and decoded onto any output grid.

### 2.2 The model: NatureV1
| | |
|---|---|
| Parameters | 89.45 M: 17 iHELIX blocks 84.67 M, input encoders 2.42 M, decoder 0.79 M, storm heads 1.23 M, time embedding 0.13 M, other output heads (field, weather type, noise, ENSO) 0.11 M, state head 0.09 M, norms + mesh seed ~3 k |
| Hidden size / heads | 512 / 8 heads (4 key-value heads), head dim 64, MLP 2048 |
| Latent mesh | 4,096 points, 48 neighbours, cluster 64 |
| Head radii | 120 km to 1,600 km |
| History | 6 frames × 6 h |
| Inputs | ERA5 analysis (89 channels + observation masks, padded to 96); calendar features (solar angle, season) given explicitly; optional satellite and environment encoders; per-member noise for ensembles |
| Outputs | gridded fields with log-variance; full-state residual with log-variance; track mixture (6 modes × 9 leads × 5 parameters); intensity; landfall; RI; eyewall; ENSO |
| Precision | bf16 mixed precision; gradient checkpointing |

**The storm path** pools the processed mesh around the storm centre with a 500 km Gaussian and, since 0.9, a
second 1,000 km pool (the ridge that steers a storm sits a thousand kilometres from the eye). It adds the storm's
own state (wind, pressure, 12 h motion, 12 h wind change — what a 1.5° reanalysis cannot show), the storm's
position (Coriolis and beta drift), and the **physical environment** computed from the state tensor. The track
head predicts the **residual from a steering-flow track**, which is a far easier target than the whole track:
beta drift, trough interactions, what steering alone gets wrong.

**Everything new starts at exactly zero effect.** The state head opens as persistence, the ensemble noise as
nothing, the environment inputs as nothing. Adding a capability can never make an existing checkpoint worse, and
uncertainties open at climatological spread rather than at a confident guess.

### 2.3 The physics inside the model
- **Channel roles.** Every input channel is prognostic (evolved), forcing (incoming solar radiation, recomputed
  each step from orbital geometry) or static (land mask, orography, observation masks, padding). A static field
  marked prognostic lets a model slowly move a coastline; this is what prevents it.
- **Rollout.** The whole atmosphere advances 6 h at a time on the model's own forecast, with surface-pressure mass
  conservation and a GraphCast-style curriculum from one step to twelve.
- **Ensembles.** Per-point noise is spread spatially by the geodesic blocks; a fair CRPS loss grows the spread only
  as far as the data justifies; spread/skill is reported.
- **Hurricane physics** (`naturev1.steering`): thickness-weighted 850–200 hPa steering over the 200–800 km ring
  around the centre, 200–850 hPa shear, 700 hPa specific humidity, SST versus 26.5 °C; advection with the correct
  longitude scaling and climatological beta drift; a steering track from today's analysis held steady, or from the
  model's own rolled-out atmosphere step by step. The steering track needs **no training at all**.

### 2.4 Training, in stages
The whole run took **about one week of wall-clock time** on one RTX PRO 6000 (96 GB) in a hosted notebook. The
GPU time was roughly three to four days; the rest was hosted-session overhead — sessions end after 12 hours or 90
idle minutes and the disk is wiped, so each restart re-pulled checkpoints and staged data from this repository
before training resumed. Every stage is resumable from a one-minute checkpoint, so no restart lost more than a
minute of training.

1. **Stage one — pretraining on reanalysis.** Every surface channel at every lead, Gaussian negative
   log-likelihood, area-weighted, on years of ERA5. This is where the backbone learns what the atmosphere does,
   from about 2×10¹¹ supervised values.
2. **Stage 1b — full-state rollout.** The residual state head is trained to roll the whole atmosphere forward,
   curriculum to 12 steps (72 h), 2-member CRPS ensemble.
3. **Stage two — hurricane heads on HURDAT2.** The backbone is **frozen**; only ~1.2 M parameters ever see a
   best-track label. The frozen backbone runs once per storm into a cached feature bank; the heads then train in
   seconds per epoch with early stopping on seasons never trained on. Afterwards the consensus weights are fitted
   on the held-out seasons.

### 2.5 Data
- **ERA5 via WeatherBench 2**, 1.5°, 240×121, six-hourly, 13 pressure levels. Split **by time** over 1959–2021
  (train to 2015, validation 2016–2019, test 2020–2021); this run staged the last **five training years,
  2011–2015 (7,300 global states, ~38 GB)**, and validated on 2016. A random split would leak tomorrow's answer
  into training.
- **HURDAT2** Atlantic best tracks, 1979 onward (the satellite era). Split **by season** (validation 2017, 2019,
  2021; test 2018, 2020). Labels are paired with the reanalysis **by timestamp**, not row index; off-synoptic
  landfall specials are dropped rather than snapped; scarce labels (radius of maximum wind exists on under 5% of
  points) arrive as NaN and are masked, never imputed.
- Normalisation per flat channel with log and asinh transforms for spiky fields (precipitation, vertical velocity)
  and observation masks for gappy ones (sea-surface temperature).

---

## 3. Verification and safety

The model is scored the way operational centres score: against the baselines a forecaster would actually use, on
data it never saw, in physical units.

- **WeatherBench-style scorecards**: area-weighted RMSE and anomaly correlation for t2m, mslp, u10, v10,
  precipitation and for Z500 / T850 rollouts, against **persistence** and **climatology** on identical batches,
  with a 1% margin before anything is called skill.
- **Storm scorecard** on held-out seasons: great-circle track error of the learned track, the physics-only
  steering track, the consensus (scored leave-one-season-out), a storm that stays put, and straight extrapolation;
  intensity MAE against persistence. Beating extrapolation and persistence is the bar; beating steering is what
  the learned part is for.
- **Hindcast**: a held-out hurricane (Dorian 2019) forecast from its real analysis, every member against what
  happened, with wind forecast versus actual.
- **Training watchdog**: raises on a non-finite loss, warns on divergence and on collapse; held-out validation
  every 500 steps with an explicit over-fitting flag.
- **Preflight**: grid alignment, normalisation spikes, grid weights and device agreement are checked before step
  one, and the run halts if anything fails.
- **Does it respond to the weather?** A probe confirms different inputs give different outputs.

Resilience is part of safety for a multi-day run: atomic checkpoint writes every minute with snapshots and a
second copy elsewhere; the optimizer, schedule, scaler and random states travel with the weights; damaged
checkpoints fall back to older ones and incompatible ones are quarantined rather than force-loaded; a supervised
background process restarts the run on crash, out-of-memory or stall; every file the run reads back is written
atomically; and on a hosted notebook everything expensive is mirrored to this repository so a wiped disk costs
minutes.

---

## 4. Results

All numbers below come from the published weights. Track errors are great-circle kilometres; "steering" is the
physics-only steering-flow track (no training), "straight" is extrapolation of the last 12 hours, "stay-put" is a
storm that does not move, and wind "persistence" is "it stays as strong as it is now". The raw outputs are in
[`results/`](./results) and the scripts that produced them in [`eval/`](./eval).

### 4.1 Training, in numbers
| Stage | What trained | Steps | Notes |
|---|---|---|---|
| One — reanalysis pretraining | backbone + surface heads | 7,275 (8 epochs) | ERA5 2011–2015, Gaussian NLL |
| 1b — full-state rollout | backbone + state head | 4,000 | curriculum to 12 steps (72 h), 2-member CRPS; **74.3 GPU-hours** (4,459 min) |
| Two — hurricane heads | storm heads only, backbone frozen | best epoch 3 of 28 | 584 storms / 16,219 samples, 1979+; feature banks ~2 h, heads minutes; early-stopped on held-out seasons |

**About one week end to end** (three to four GPU-days plus hosted-notebook restarts), one RTX PRO 6000 (96 GB). The full per-output training record is in `training_record.json`; outputs with zero
steps (weather type, landfall timing, RI onset, ENSO, satellite and environment encoders) are never shown.

### 4.2 Global weather (held-out data)
From the run's own WeatherBench-style scorecard, on identical batches for every baseline:
- **Full-state rollout: beats climatology by more than 1% on 84 of 84 field-lead pairs, and has skill against both
  persistence and climatology out to +72 h on msl, q700, t2m, t850, u10, v10 and z500.** The ensemble
  spread/skill ratio at +24 h is **1.08** (1.0 is perfectly calibrated).
- The **direct multi-lead surface heads do not have skill** (they beat persistence on 6 of 45 field-lead pairs, all
  precipitation). They were trained in stage one and not updated when the rollout retrained the backbone under
  them; use the full-state rollout for weather. The complete tables are in `logs/train.log`.

### 4.3 Hurricane track and intensity — held-out seasons 2017, 2019, 2021
59 Atlantic storms (including Harvey, Irma, Maria, Dorian, Lorenzo, Ida) that neither the backbone nor the storm
heads trained on. Consensus weights scored leave-one-season-out.

| Lead | n | **NatureV1** consensus | learned track | steering | straight | stay-put | wind error | persistence |
|---|---|---|---|---|---|---|---|---|
| 6 h | 1589 | **39** | 42 | 54 | 53 | 140 | 6.2 kt | 3.4 kt |
| 12 h | 1530 | **75** | 84 | 107 | 121 | 275 | 7.8 kt | 6.6 kt |
| 24 h | 1412 | **163** | 180 | 233 | 283 | 527 | **11.0 kt** | 11.8 kt |
| 48 h | 1187 | **421** | 457 | 572 | 628 | 977 | **16.1 kt** | 18.8 kt |
| 72 h | 997 | **739** | 844 | 978 | 1002 | 1369 | **19.5 kt** | 24.2 kt |
| 120 h | 677 | **1373** | 1576 | 1779 | 1828 | 2081 | **22.8 kt** | 31.6 kt |

Track beats steering and straight-line extrapolation at **every** lead; wind beats persistence from 24 h on.

### 4.4 End to end on an unseen El Niño season — the 2023 hurricanes
2023 was a strong El Niño year and is outside everything the model saw. ERA5 for 2023 was rebuilt at WeatherBench 2's
1.5° grid from ARCO-ERA5 with area-weighted conservative regridding — checked against the WeatherBench 2 store on
an overlapping date, where it agrees to rounding precision (0.03 Pa in sea-level pressure) — and every forecast
ran through the **full** model, raw weather in. All seven hurricanes, one forecast per day at 00Z (96 forecasts),
with the shipped consensus weights and no refitting.

| Lead | n | **NatureV1** | steering | straight | stay-put | wind error | persistence | 2017/19/21 for comparison |
|---|---|---|---|---|---|---|---|---|
| 6 h | 96 | **38** | 47 | 53 | 125 | 6.5 kt | 2.9 kt | 39 |
| 12 h | 95 | **74** | 99 | 128 | 248 | 7.5 kt | 5.8 kt | 75 |
| 24 h | 89 | **167** | 231 | 309 | 476 | **10.7 kt** | 11.9 kt | 163 |
| 48 h | 82 | **503** | 689 | 719 | 866 | **14.7 kt** | 18.7 kt | 418 |
| 72 h | 75 | **810** | 1146 | 1132 | 1211 | **15.8 kt** | 21.5 kt | 735 |
| 120 h | 61 | **1487** | 2028 | 1868 | 1807 | **16.1 kt** | 28.0 kt | 1374 |

| Storm (peak) | 24 h | 48 h | 72 h |
|---|---|---|---|
| Lee (145 kt) | **100** (steering 120) | **240** (378) | **325** (635) |
| Margot | 84 (167) | 350 (522) | 678 (926) |
| Tammy | 125 (223) | 441 (601) | 838 (1295) |
| Don | 169 (222) | 439 (577) | 858 (1031) |
| Idalia | 165 (245) | 534 (718) | 1049 (1373) |
| Franklin | 211 (262) | 680 (944) | 827 (1263) |
| Nigel | 292 (369) | 727 (924) | 1111 (1429) |

Out to 24 h the El Niño season scored the same as the held-out neutral/La Niña seasons; from 48 h the errors are
larger, but the margin over physics steering grew (29% better at 72 h, against 25%). It beat steering for every
storm at every one of these leads. One season, seven storms, 00Z only: encouraging, not conclusive.

### 4.5 Landfall, rapid intensification and structure (2017/19/21)
| Output | Result | Verdict |
|---|---|---|
| Landfall probability | Brier skill over the base rate +4% (6 h) to +29% (72 h); AUC 0.81–0.86. 2023: +30 to +42% from 72 h, AUC 0.81–0.95 | skilful |
| Rapid intensification (3 thresholds) | ranks risk well (AUC 0.82–0.84) but probabilities are too high (Brier worse than the base rate) | needs calibration before use |
| Peak wind (eyewall head) | MAE 6 / 11 / 19 kt at 6 / 24 / 72 h vs 22–26 kt for the climatological mean | beats climatology |
| 34/50/64 kt wind radii | MAE 32–37 n mi vs 47–49 for the mean | about 30% better |
| Radius of maximum wind | 28 vs 29 n mi (6 h), loses by 72 h | no skill |

### 4.6 Individual forecasts
**Maria 2017**, issued 18 Sep 00Z east of the Lesser Antilles, the full model from raw ERA5 (`results/maria_2017_end_to_end.txt`):
miss **14 / 26 / 15 / 65 / 118 / 343 km** at 6 h / 1 / 2 / 3 / 4 / 5 days, against 33 / 91 / 101 / 136 / 181 / 245 km
for steering. Wind forecast about 70 kt against an observed 145–150 kt.

**Dorian 2019**, issued 30 Aug 18Z, from the run's hindcast:

| Lead | 6 h | 24 h | 48 h | 72 h | 120 h |
|---|---|---|---|---|---|
| learned track (top scenario 26%) | 24 | 99 | 336 | 688 | 1034 |
| consensus | 12 | 106 | 355 | 669 | 935 |
| steering | 14 | 177 | 382 | 546 | 531 |
| straight | 17 | 161 | 397 | 651 | 1012 |

Good to two days; Dorian then stalled over the Bahamas, which steering handled better.

### 4.7 In context
For scale only: the National Hurricane Center's official Atlantic track error averaged about 66 km at 24 h and
168 km at 72 h over 2018–2022, and the leading global AI models (GraphCast, Pangu-Weather, FuXi) and ECMWF are in
the same league. NatureV1's 163 km and 739 km are **2.5–4.5× larger**. It is a 1.5° model trained on five years
of data with a few GPU-days of compute; those systems use 0.25°, decades of data and orders of magnitude more
compute. Intensity is the weakest output: like every 1.5° model it cannot see an eyewall, and it forecasts about
70 kt for storms that reach 145–160 kt.

### 4.8 Red team: where it breaks
A deliberate attempt to break the published model: physically impossible inputs, calibration against what
actually happened, removing one input at a time, logical consistency, a data-leak audit, and the hardest storm we
could find. Everything here is on held-out data unless noted; scripts in [`eval/`](./eval).

**Data leakage — none found.** Storm-head training seasons are 1979–2016; held-out 2017/2019/2021 and test
2018/2020/2022 never appear in training; the weather backbone trained on 2011–2015 only. 2023 is outside
everything.

**Calibration — is the uncertainty honest?** (2017/19/21; 2023 in brackets)

| Output | Claimed | Actually |
|---|---|---|
| Track, full 6-scenario mixture, 50% region | 50% | 47–53% (32–55%) — honest |
| Track, full mixture, 90% region | 90% | 87–92% (81–93%) — honest, slightly tight at 4–5 days in 2023 |
| Track, **top scenario's "95%" cone** (what a single cone shows) | 95% | **60–87%** — over-confident; the other scenarios carry the rest |
| `cone_radius_km_68` field | 68% | it is the 1-sigma ellipse, which holds **39%** in two dimensions — mislabelled |
| Wind, 90% interval | 90% | 85–93% — honest |
| Landfall by 72 h | 18% / 39% / 61% / 79% | 10% / 31% / 51% / 79% — too high below 70% |
| Rapid intensification (+30 kt/24 h) | 21% / 39% / 55% | **1% / 10% / 38%** — far too high; do not read these as probabilities |

**Intensity is biased low, and worse for stronger storms** (forecast minus actual, held-out):

| Current strength | 24 h | 72 h |
|---|---|---|
| Tropical storm (<64 kt) | −3 kt | −6 kt |
| Category 1–2 | −7 kt | −15 kt |
| Category 3–5 | −10 kt | −18 kt |

**What the hurricane heads actually rely on** (remove one input, held-out error change):

| Removed | Track 24 h | Track 72 h | Wind 24 h |
|---|---|---|---|
| Physics steering-track prior | **+240%** | +40% | 0% |
| All backbone features | +30% | +13% | +18% |
| Backbone features from a different storm | +29% | +12% | +43% |
| 1,000 km far pool | +10% | +3% | +5% |
| Physical environment (shear, steering, humidity, SST) | +14% | +4% | **+2%** |
| Current wind and pressure | 0% | 0% | **+58%** |
| Last 12 h of motion | +10% | +15% | −1% |

The track is the physics steering track plus a learned correction worth about 30% at 24 h. The intensity forecast
leans on the storm's current wind; the shear/humidity/SST inputs built to help intensity barely move it.

**Inputs the safety checks do not catch.**
- **Garbage in, plausible out.** Random noise in place of the backbone features produced a normal-looking forecast
  (22–45 kt, a sensible track) with **zero** check failures. There is no out-of-distribution detector.
- **A corrupted steering prior passes straight through.** Shifting the prior by 30° moved the forecast 30° with
  zero check failures.
- **The consensus track is not bounds-checked** — a centre near the pole gave consensus latitudes above 90°;
  only the scenario tracks were flagged.
- A missing (NaN) storm centre is flagged on the track, but intensity and RI are still produced as if valid.
- Caught correctly: a 400 kt current wind (forecast 370 kt → flagged), positions off the planet, cones wider than an
  ocean, and the untrained satellite path (`trustworthy: false`). The dateline and Southern Hemisphere beta drift
  are handled correctly.
- Bug: `build_forecast` without a training record crashes on outputs that carry no ENSO field.

**Logical consistency.**
- Landfall is "probability of landfall *by* this lead", which can never fall as the lead grows. It falls in **20%**
  of forecasts (by up to 12 points).
- A 50 kt or 64 kt wind radius exceeds the 34/50 kt radius in **21%** of forecasts, and the radius of maximum wind
  exceeds the 34 kt radius in **17%** — both physically impossible.
- Consistent: the three RI thresholds are always ordered, and the two intensity heads agree to 1.5 kt on average.

**The hardest case: Hurricane Otis, 2023** (East Pacific — outside the Atlantic the storm heads trained on; full
model from raw ERA5, 5 forecasts). Otis went from 50 kt to a 130 kt Category 5 in 24 hours and struck Acapulco.
From 24 Oct 00Z NatureV1 forecast **45 kt** — a slight weakening — and gave rapid intensification 44%, about what
it said on the quiet days before (34–38%). Track: 104 km at 24 h (steering 156 km), but at 72 h it kept Otis at the
same latitude while it moved 5° north. **It cannot warn of explosive intensification.**

**Reproducibility gap.** The full-state rollout scales its residuals by per-channel tendency statistics measured
from the training data; they are not shipped in this release, so reproducing the weather rollout exactly means
re-measuring them from ERA5 2011–2015 (64 evenly spaced training windows).

Still running at the time of writing: six more storms outside the Atlantic (West Pacific, Southern Hemisphere,
East Pacific) and a 10-day rollout to test stability beyond the 72 h trained horizon. They will be added here.

## 5. Limitations

- **Resolution.** 1.5° (~165 km) cannot resolve an eyewall. Intensity is expected to be the weakest output; the
  explicit environment inputs exist to compensate, and the scorecard will say how far they get.
- **Not operational.** It runs on reanalysis, which is available days after the fact. There is no real-time
  analysis feed, and the satellite input path exists but is untrained.
- **Verification is internal.** Three held-out seasons plus the 2023 season, scored by this project's own
  scripts; no independent party has verified it.
- **Intensity.** Biased low, by 10–18 kt for major hurricanes; misses explosive intensification entirely (Otis).
  Rapid-intensification probabilities are far too high; radius of maximum wind has no skill.
- **Uncertainty display.** The full scenario mixture is well calibrated, but a single scenario's cone is
  over-confident and the `cone_radius_km_68` field is a 39% ellipse. Draw cones from the mixture.
- **No input sanity check.** Corrupted inputs can produce plausible-looking forecasts that pass every check
  (section 4.8). Physically impossible structure (wind radii out of order, landfall probability falling with lead)
  is not yet caught.
- **Surface heads are stale.** The direct multi-lead surface fields were not retrained after the rollout stage;
  use the full-state rollout for weather.
- **Small data.** Five years of reanalysis at 1.5°, and the storm heads start memorising after about three epochs.
- **Atlantic only** for the storm heads, 1979 onward; other basins are out of distribution.
- **Training cost.** About one week wall-clock on one GPU in a hosted notebook; most of the overhead was session
  restarts. A machine that keeps its disk would cut it to three or four days.
- **Tornadoes, hail, flash floods** are far below this resolution and out of scope.

---

## 6. Using it

Install and train, or score, from one cell (marimo, Colab, Jupyter or a plain script):

```python
pip install "naturev1[all]"
```
The complete pipeline is the cell `colab_naturev1.py` shipped with the package; it installs what it needs, stages
the data, trains in stages, scores itself, and only publishes if it earned it. Every setting is at the top.

Load the released model (weights, config and the normalisation statistics it was trained with):

```python
from naturev1.hub import from_pretrained

model, config, normalizer = from_pretrained("Sigmandndnns/NatureV1-500")
print(model.trained_heads())      # steps each output has received; 0 = never trained, never shown
```

The weights are also provided as `model.safetensors`. To reproduce the results in section 4, see [`eval/`](./eval):
`elnino2023.py` rebuilds 1.5° ERA5 for any date from ARCO-ERA5 and runs the full model on a season's storms.

`build_forecast(outputs, config.lead_times_hours, issued, storm_center=(lat, lon), trained=model.trained_heads())`
turns a forward pass into the forecast bundle: track scenarios with cones, consensus with members, landfall,
intensity, rapid intensification, eyewall, and a point forecast, in physical units with 90% intervals, with
every untrained section marked unavailable.

### Files in this repository
```
model.safetensors                 the released weights (89.45 M parameters), safetensors format
model.safetensors.pt              the same weights as a PyTorch state dict (what from_pretrained loads)
config.json, normalizer.json      model configuration and the training normalisation statistics
training_record.json              steps per output, parameter count, stage-two state
results/                          every scorecard and test in section 4, as produced
eval/                             the scripts that produced them
checkpoints/stage1|rollout|stage2 full training checkpoints (with optimizer state) for resuming or fine-tuning
checkpoints/storm_bank_*.pt       frozen-backbone features for the storm heads
staging/                          the normalised ERA5 staging, so a new session needs no re-download
logs/train.log                    the complete training log, including every scorecard
```

---

## 7. Why it is built this way

Three convictions run through the whole stack.

**Geometry is not something a weather model should have to learn.** Distances, areas, the sphere, the vertical:
these are known. iHELIX gives attention physical radii and true area weights so the parameters go on the
atmosphere, not on rediscovering the globe.

**Scarce labels must be spent on few parameters.** There are 55,230 Atlantic best-track points in history and no
engineering makes more. The backbone learns from 10¹¹ free reanalysis values; the hurricane heads, about a million
parameters, learn from the labels on top of a frozen backbone, with the physics handed to them explicitly.

**A forecast that cannot say "I don't know" is dangerous.** Every output carries an uncertainty, every untrained
head is hidden, the physical bounds are checked, the model is scored against persistence before anyone is
allowed to believe it, and it has been red-teamed so its failures are written down (section 4.8). The light bulb was not made in one shot; this one is instrumented so that when it fails,
it fails visibly.

---

*NatureV1, iHELIX and the training pipeline are by Nathan, Apache-2.0. This card describes the first full
training run of version 0.10 of the pipeline (about one week of training, completed October 2026). ERA5 data: Copernicus Climate Change Service; HURDAT2: NOAA
National Hurricane Center.*
