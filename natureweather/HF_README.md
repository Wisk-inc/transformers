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

> **Status: training in progress.** This repository holds the live checkpoints of the first full training run,
> mirrored here every ten minutes while it trains. No skill is claimed yet. The scorecards that decide whether
> this model is any good are produced by the training pipeline itself and will be added to this card when the
> run finishes. Until then, treat every number it produces as unverified. **Do not use it for real decisions.**

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
- 2 m temperature, mean sea-level pressure, 10 m wind, precipitation, each as a mean and a 90% interval, at every
  grid point.
- A weather-type category (clear to thunderstorm to snow) at every grid point.
- A **full-state rollout**: the whole atmosphere — all 86 prognostic channels, surface to 50 hPa — stepped forward
  6 hours at a time on its own forecast, with mass conservation, out to 72 hours, as an ensemble.

**For a tropical cyclone, given its current centre**
- **Track as scenarios, not a line**: six weighted trajectories ("38% recurves offshore, 27% tracks the coast…"),
  each with a tilting uncertainty ellipse per lead, so the cone of uncertainty is drawn from the model, not
  from a lookup table.
- A **multi-model consensus track**: the learned track, a physics-only steering-flow track, and straight
  extrapolation, blended with per-lead weights fitted on held-out seasons — the way the National Hurricane Center's
  consensus beats every single model in it, done inside one model. Every member is reported alongside the blend.
- **Intensity**: maximum wind and minimum central pressure per lead, with intervals and the Saffir–Simpson category.
- **Rapid intensification**: the probability of a 25, 30 and 35 kt gain in 24 hours, plus the expected change.
- **Landfall**: the probability of landfall by each lead and the most likely window.
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
- Every number is checked against **physical bounds** before it is shown or saved: no 400 kt hurricane, no
  pressure outside 850–1100 hPa, no cone wider than an ocean, no probabilities that do not sum to one.
- It is **published only if it beats persistence** on held-out data by a margin, and only if it demonstrably
  changes its answer when the weather changes.

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
- **ERA5 via WeatherBench 2**, 1.5°, 240×121, six-hourly, 13 pressure levels, 1959–2021. Split **by time**: train
  to 2015, validation 2016–2019, test 2020–2021. A random split would leak tomorrow's answer into training.
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

## 4. Results so far

**None for skill.** The first full run is in progress. What has been verified end to end is the *mechanism*: the
whole pipeline — staging, three training stages, scorecards and the hindcast — runs on real ERA5 and HURDAT2 data
at reduced scale; 236 tests cover the physics (a storm in a uniform westerly moves east by exactly u·t, shear equals
the level difference, a consensus recovers the member that is exactly right), the label pairing, the resilience
paths and the honesty rules.

One result stands independently of training, because it uses no trained parameters: on Hurricane Dorian, issued
2019‑08‑30 18Z from the 1.5° analysis, the **physics-only steering track** had errors of 177 km at 24 h, 382 km at
48 h, 546 km at 72 h and 531 km at 120 h. Straight extrapolation scored 161, 397, 651 and 1012 km: it wins at
24 h, and steering wins from 48 h onward. For scale only, NHC's official Atlantic track error averaged 35.7 n mi
(66 km) at 24 h and 90.5 n mi (168 km) at 72 h over 2018–2022. That is an average over many storms from an operational
centre, set against one storm from a 1.5° analysis, so it is not a like-for-like comparison. The steering member is
the floor the learned model starts from, not the ceiling it is aiming at.

When the run finishes, this section will carry the scorecards verbatim.

---

## 5. Limitations

- **Resolution.** 1.5° (~165 km) cannot resolve an eyewall. Intensity is expected to be the weakest output; the
  explicit environment inputs exist to compensate, and the scorecard will say how far they get.
- **Not operational.** It runs on reanalysis, which is available days after the fact. There is no real-time
  analysis feed, and the satellite input path exists but is untrained.
- **Verification is internal.** Three validation seasons, scored by the pipeline's own scorecards; no independent
  party has verified it. Calibration of the landfall and RI probabilities has not yet been measured.
- **Atlantic only** for the storm heads, 1979 onward.
- **Tornadoes, hail, flash floods** are far below this resolution and out of scope.

---

## 6. Using it

Install and train, or score, from one cell (marimo, Colab, Jupyter or a plain script):

```python
pip install "naturev1[all]"
```
The complete pipeline is the cell `colab_naturev1.py` shipped with the package; it installs what it needs, stages
the data, trains in stages, scores itself, and only publishes if it earned it. Every setting is at the top.

Load a checkpoint from this repository:

```python
from naturev1 import NatureV1, NatureConfig, build_forecast
from huggingface_hub import hf_hub_download
import torch
from ihelix import fibonacci_sphere

path = hf_hub_download("Sigmandndnns/NatureV1-500", "checkpoints/stage1/latest.pt")
payload = torch.load(path, map_location="cpu", weights_only=False)
config = NatureConfig.from_dict(payload["config"])
model = NatureV1(config, fibonacci_sphere(config.latent_points, num_neighbours=config.latent_neighbours,
                                          cluster_size=config.latent_cluster))
model.load_state_dict(payload["model"])
print(model.trained_heads())      # steps each output has received; 0 = never trained, never shown
```

`build_forecast(outputs, config.lead_times_hours, issued, storm_center=(lat, lon), trained=model.trained_heads())`
turns a forward pass into the forecast bundle: track scenarios with cones, consensus with members, landfall,
intensity, rapid intensification, eyewall, and a point forecast, in physical units with 90% intervals, with
every untrained section marked unavailable.

### Files in this repository
```
checkpoints/stage1/latest.pt      stage one, pushed every 10 minutes while training
checkpoints/rollout/latest.pt     stage 1b, once it starts
checkpoints/stage2/latest.pt      hurricane heads, once they start
checkpoints/storm_bank_*.pt       cached frozen-backbone features for the storm heads
staging/era5.npy, era5_val.npy    the normalised ERA5 staging (mirror, so a new session needs no re-download)
staging/stats.json, batch.json    normalisation statistics and the measured batch size
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
head is hidden, every number is bounds-checked, and the model is scored against persistence before anyone is
allowed to believe it. The light bulb was not made in one shot; this one is instrumented so that when it fails,
it fails visibly.

---

*NatureV1, iHELIX and the training pipeline are by Nathan, Apache-2.0. This card describes version 0.10 of the
pipeline; it will be updated with the scorecards when the first full training run completes.*
