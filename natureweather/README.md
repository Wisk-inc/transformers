# NatureV1

A probabilistic weather and hurricane-track model that treats satellite imagery as **geolocated samples on
a sphere**, not a rectangle of pixels.

```bash
pip install "naturev1[all]"
```

## Why not a ViT or CNN over the image

Three things break when you crop a GOES scene, divide by 255 and hand it to a vision backbone:

1. **A GOES pixel is not a fixed size.** ABI scans in fixed *angular* steps from geostationary orbit.
   Measured off a real scene, the limb pixel covers **64× the ground area** of the nadir pixel. A
   convolution weights them equally and silently over-counts the stretched edge of the disk.
2. **An array index is not a place.** Nothing in `image[400, 1200]` says 24.7°N 81.3°W.
3. **Normalizing to 0–1 destroys the physics.** Channel 13 is a brightness temperature: 180 K is an
   overshooting top, 300 K is warm ocean. Per-image rescaling erases exactly what was measured.

NatureV1 solves the fixed-grid projection for every pixel's latitude/longitude (validated against the
file's own metadata — it recovers 2.07 km at nadir where nominal C13 resolution is 2 km), computes each
pixel's true ground footprint and uses it as a quadrature weight, and keeps brightness temperature in
kelvin. Satellite and analysis data then fuse without either being resampled onto the other's grid.

## Every output is a distribution

Track comes out as weighted **scenarios**, each a full trajectory with a tilting uncertainty ellipse.
Collapsing them to one line scores 116 under the mixture likelihood where keeping both branches scores
8.2 — because the average of "recurves offshore" and "hits the coast" is a track through somewhere the
storm was never going. Fields carry per-point variance, landfall a probability per lead time, intensity a
Saffir-Simpson category with an interval.

## It trains twice, because the labels are 4 orders of magnitude too few

An 88M-parameter model fitted to the Atlantic best-track archive is not a weather model. The archive is
55,230 points — 1,167 landfalls, 1,839 rapid intensifications, 2,587 with a radius of maximum wind. That
is **0.0006 supervised values per parameter**, and what you get back is an expensive lookup table for
storms that already happened.

So the backbone never sees it.

**Stage one is self-supervised on ERA5 reanalysis**, where the label is free: given the atmosphere now,
predict it at +6 through +120 hours. The WeatherBench 2 six-hourly store is 92,040 timesteps from 1959 to
2021 — 1.6×10¹⁰ supervised values, **180 per parameter**, and nobody annotates anything.

**Stage two fine-tunes the storm heads with the backbone frozen.** 0.96M parameters of 89M — 1.1% — ever
see a best track. Those 24,585 paired storm points can fit 0.96M parameters; they cannot fit 89M.

```python
from naturev1 import ERA5Window, StormWindow, pair_tracks_with_reanalysis

pretrain = ERA5Window(era5, indices=splits["train"], lead_steps=offsets)   # free labels
starts, targets, _ = pair_tracks_with_reanalysis(tracks, store_times, offsets)
finetune = StormWindow(ERA5Window(era5, indices=starts), targets)          # real outcomes
model.freeze_backbone(True)
```

Longitude rotation is an **exact** augmentation here, not an approximation: on an equiangular grid,
rotating the globe by a whole number of cells is a roll of the array, and this architecture is exactly
equivariant to it. 240 valid atmospheres, free, nothing resampled.

## Four things the obvious implementation gets wrong

Each of these ran clean on shapes, ranges and a falling loss while the data was wrong, so the regression
tests assert on physics and geometry instead:

1. **Dimension order.** WeatherBench 2 stores `(time, longitude, latitude)`; the coordinate mesh is
   latitude-major. Read untransposed you get the right point count, sane values, and every sample bound
   to the wrong place on Earth. Correlation of temperature against its assigned latitude: ~0 → **−0.873**.
2. **Cell weights.** `cos(latitude)` gives the pole row 6.1e-17 — zero in float32 — deleting the poles
   from every area-weighted sum. True spherical cell area makes a pole cell 306× lighter, not 10¹⁶×.
3. **Precipitation.** 15% of the grid is exactly zero and the max sits 30σ out, so a z-score trains the
   model to predict zero everywhere. `log1p(x/0.1mm)` brings it to 4.5σ and inverts exactly.
4. **Missing is not average.** Sea-surface temperature is absent over 27.9% of the grid; filling it with
   the ocean mean asserts warm water over Kansas. Gappy variables get an observation channel.

## Heads start at climatology, not at zero

A linear head emits about zero, so asked for central pressure in hPa it opens 1000 off — a squared error
of a million. Measured on a real fine-tuning step, the intensity term was 119,191 of a 60,565 total, with
a gradient norm of 207,018. Anchoring each head at Atlantic climatology and starting each log-variance at
the observed spread (rather than claiming ±1 kt about peak wind) brings the same step to **total 58,
intensity 4, eyewall 5, gradient norm 13** — with no architecture change. The RI classifier opens at the
measured 4.11% base rate, not at even odds.

## Built for a runtime that dies

Checkpoints go down on a wall-clock interval, written to a temp file and renamed into place, so a killed
cell leaves the previous checkpoint intact. Resume restores optimizer moments, LR schedule, scaler and RNG
state — not just weights.

## 0.8: the whole atmosphere, and an audit

The model now predicts and evolves every prognostic channel — 13 pressure levels included — as a residual
from persistence, with solar forcing computed from orbital geometry, mass conserved, a pressure- and
area-weighted loss, and fair-CRPS ensembles. Storm heads pool around the storm and are given its current
intensity and motion. Stage two runs the frozen backbone once per storm and trains the heads on the
cached features, which turned days per epoch into seconds.

A full read of the code found and fixed, among others: a GPU memory leak on every training step with
DataLoader workers; hurricane lead targets at the wrong valid time on 10.9% of samples (92% of them on
landfalling stretches); a rollout loop that never rolled out; a publish gate that could never open; and
untrained heads printed as forecasts. Each has a regression test in `tests/test_fixes_080.py`.

## 0.12: the satellite path rebuilt, every basin, 8x B300

Built from what the trained model and its red team showed: intensity biased low, rapid intensification
missed, southern-hemisphere tracks worse than physics, poleward turns missed, a storm centre required.

- `naturev1.ibtracs` — every basin since 1980 (IBTrACS), wind normalised to one-minute, ~230,000 labelled fixes.
- `naturev1.satstage` — GridSat-B1 1980-2021 (4-6 TB) staged into quarter-degree global frames (mean, coldest
  pixel, spread, coverage, satellite, view angle) and 320 px storm loops; hand-decoded, quality-controlled, resumable.
- `naturev1.satnet` — **SatNet-Storm** (intensity, centre fix from a 150 km-wrong guess, radii, intensity change,
  ordinal RI, motion; hemisphere-mirrored) and the **satellite retrieval** (stale state + satellite loop + cloud
  motion → full state, starting exactly at the stale state).
- `naturev1.stormtrain` — stage B: heads + the backbone's last blocks on every basin, retrieval in the loop,
  RI 2.0, poleward frame; banks with two rollout track members; consensus, calibration, intensity bias
  correction; scored on test seasons against the model it started from.
- `naturev1.tracker` — vortex tracking through rollouts, storm detection.
- `naturev1.engine` — DDP on N GPUs, bf16, a clock-driven schedule, every checkpoint pushed to the Hub in order.
- `b300_naturev1.py` — the runbook: preflight, staging, stage A (3 h, both satellite networks side by side),
  stage B, and a publish step that only replaces the release if the new model beat it.

Then more ways to be right, each weighted by the held-out seasons (weights may be zero):

- `naturev1.thermo` — potential intensity (Bister-Emanuel), ventilation, entropy deficit, genesis potential,
  cyclone phase space and SHIPS-like predictors from the model's own atmosphere, fed to every storm head.
- `naturev1.windfield` — asymmetric Holland wind field, NHC-style 34/50/64 kt probabilities and arrival times
  (points or a 0.25 degree map), inland decay, gusts, R-CLIPER rain.
- `naturev1.statistical` — SHIPS-style regression, LGEM-style logistic growth, climatology-persistence, an
  intensity consensus with per-lead weights and biases, track bias correction, a predicted-error cone, an RI consensus.
- `naturev1.tracker` — multi-field centre fix, intensity and phase along the rollout, rollout-ensemble and
  analog track members; the satellite centre fix iterates to convergence.
- Training and forecast time: intensity loss rebalanced by class, an average of the weights (EMA), and a
  test-time rotation ensemble used only if it beats a single pass on held-out seasons.

Tested end to end on CPU, including two-process data parallelism; not yet run on B300 hardware.

## 0.13: hurricane physics that beats extrapolation -- measured, not promised

Technical report: [Hurricane tracks without a neural network](https://corxlabs-papers.static.hf.space/naturev1-hurricane-physics-report.html) (CorX Labs report 05, also as a PDF).

`naturev1.cyclone` -- 39 modules, each one idea, no neural network -- forecasts a storm's track and intensity from
one analysis and the best track so far, and detects storms before they form. Every constant was fitted on
1980-2014 best tracks or the 2016/2017/2019 seasons. On the held-out seasons (2018, 2020, 2021, 2022; 4,447
forecasts, 407 storms, every basin; storm-bootstrap 90% intervals):

| track error (km) | 6 h | 12 h | 24 h | 48 h | 72 h | 120 h |
|---|---|---|---|---|---|---|
| strongest extrapolation | 33 | 74 | 179 | 442 | 751 | 1,417 |
| CLIPER (rebuilt) | 32 | 70 | 163 | 378 | 589 | 914 |
| **physics consensus** | **29** | **59** | **127** | **301** | **515** | **884** |
| GraphCast (2018, 2020 only) | 49 | 54 | 67 | 102 | 162 | 296 |

Intensity (SHIFOR + SHIPS-style + logistic growth, blended per lead): mean absolute error 4/6/11/14/17/20/21/22 kt
at 6-120 h against persistence's 4/8/14/19/23/28/31/32 -- +14% to +31%, every interval clear of zero.

It beats extrapolation and CLIPER at every lead, and loses to every modern global model (GraphCast, Pangu, HRES,
Aurora, FGN/WeatherNext 2, FuXi, GenCast) past 6 hours; combined with one of them, lead by lead, it is never worse
than that model and better at 6-12 hours. Genesis: 68% of formations flagged a median 24 hours ahead (52% false
alarms), Brier skill +0.22.

```python
from naturev1.cyclone import physics_forecast       # track, members, cone, regime, Dorian-type stall warning,
fc = physics_forecast(case, environment)            # shear profile, intensity along the track
```

Also in 0.13: `naturev1.distill` (multi-teacher distillation with spatial / vertical / temporal alignment of every
teacher into NatureV1's space), `naturev1.assimilation` (bias correction, quality control, local optimal
interpolation for live inputs), `naturev1.highres` (decoding the latent mesh at 0.25 degrees), and
`molab_naturev1.py`, the runbook for one 96 GB GPU. Distillation and the 0.25-degree decoder are tested on CPU,
not yet trained.

## 0.13: the swarm sandbox, the satellite path and Caribbean impacts

`naturev1.swarm` (idea: Nathan) runs alongside the physics. It is a particle filter over possible futures. Each
world is a 4D worldline (longitude, latitude, steering depth, time) with its own physics, integrated in micro-segments
and re-weighted minute by minute by NHC's satellite, recon and radar centre fixes. Its wind and rain are then painted
onto every parish, province and district of the Caribbean: 309 units, with terrain.

```python
from naturev1.swarm.geography import CARIBBEAN, RegionMap
from naturev1.swarm.landfall import format_landfall, landfall_summary
from naturev1.swarm.live import forecast_storm
region = RegionMap.build(CARIBBEAN)                                   # once; cached
earlier = forecast_storm("al132025", when="2025-10-26T12", archive=True, particles=100_000, region=region)
run = forecast_storm("al132025", when="2025-10-27T00", archive=True,  # Melissa as it was known then
                     particles=100_000, region=region,
                     previous={"time": "2025-10-26T12", "track": earlier["result"]["track"]})   # 0.15: its error feeds back
print(run["impact"].table())                                          # parish-by-parish probabilities
print(format_landfall(landfall_summary(region, run["result"], run["case"], run["impact"], "Jamaica",
                                       analysis_time="2025-10-27T00")))  # when, where, how strong; rain if hit
print(run["result"]["peak_probability"])                              # P(peak within 5 days >= 64/96/113/137 kt)
print(run["result"]["intensity_if_ri"], run["result"]["ri_probability"])   # the rapid-intensification branch
```

On unseen seasons, its stall probabilities are skilful (Brier +0.08 over climatology) and its loop probabilities are
reliable. Its cone is not better than the fixed one, and its own mean track is worse than the consensus, which stays
the track. On Melissa the consensus missed the northward turn. The swarm kept western Jamaica at risk, but at 13-20%
for hurricane-force wind where the storm brought it to most of the island, because a 160 kt Category 5 was forecast
at 111-125 kt. `molab_swarm_cell.py` is the one-cell molab runbook for live storms.

## 0.13.1: the turns, and rapid intensification

The track now uses a dynamical model's **forecast** winds, the way NHC's Beta and Advection Model does. They come
from GFS live and from ECMWF HRES in the hindcasts. The model's own storm is removed at every forecast hour. The
consensus is then blended with the model's own track, one weight per lead. That lets the track see a trough two
days out, which the analysis cannot. Held-out seasons 2018 and 2020-2022, km:

| lead (h) | 6 | 12 | 24 | 36 | 48 | 72 | 96 | 120 |
|---|---|---|---|---|---|---|---|---|
| 0.13.0 consensus | 29 | 59 | 127 | 207 | 301 | 512 | 711 | 874 |
| HRES's own track | 49 | 60 | 79 | 100 | 123 | 187 | 251 | 326 |
| **0.13.1 consensus** | **26** | **45** | **72** | **96** | **120** | **184** | **249** | **326** |

On storms that turned by more than 45 degrees, the 72 h error fell from 601 to 209 km. The swarm, steered by the
same forecast, now gives skilful recurvature probabilities: Brier skill +0.26, up from +0.05. Its loop
probabilities got worse (−0.25), so treat them as unreliable.

For intensity, a rapid-intensification probability (Brier skill +0.10, AUC 0.81) switches the intensity consensus
to weights fitted on such storms. That lowers the error by 1-3% overall and by 5-10% when P(RI) ≥ 30%, at 6-36 h.

On Melissa from 27 October, the track now curves north into Jamaica: 55 / 79 / 246 km errors at 36 / 48 / 72 h,
against 187 / 392 / 892 km before. Every parish gets a 24-47% chance of hurricane-force wind, up from 13-20%. The
intensity is still far too low (118 kt at 24 h against 155 kt). The live track leans on GFS, which has
historically erred somewhat more than HRES. Details are in section 4.14 of the model card.

## 0.14: several models at once, a cone sized per forecast, a swarm on several models

The consensus is now blended with the **equal-weight mean of several models' tracks**, not one model's. Live they
are GFS, ECMWF IFS and AIFS (open data), GraphCast-GFS and the GEFS mean, each read by byte range from its public
bucket (`naturev1.swarm.models`); any model not yet published is skipped. On the held-out seasons, with the
archived models WeatherBench 2 holds (HRES, Pangu, GraphCast, FuXi, Aurora, FGN) and no weight fitted to them:

| lead (h) | 6 | 12 | 24 | 36 | 48 | 72 | 96 | 120 |
|---|---|---|---|---|---|---|---|---|
| 0.13.1, one model | 26 | 45 | 72 | 96 | 120 | 184 | 249 | 326 |
| **0.14, the models' mean** | **25** | **42** | **64** | **81** | **96** | **142** | **200** | **263** |

Also: a cone sized per forecast from member spread, model disagreement, speed, latitude and intensity (held-out
rank correlation with the error 0.17-0.38, the fixed cone 0); each swarm world follows one model's forecast
winds. Three ideas measured and rejected on the held-out seasons: anchoring the model track to the observed
position, shrinking the model's weight when it disagrees with the physics, and a quieter swarm (its spread matched
the error on the tuning seasons, but its scenario probabilities became over-confident on the held-out ones).

Intensity gains the models' own deepening (their central-pressure-deficit change along their tracks) as a member
and as a rapid-intensification predictor: held-out error down 4.8% at 24 h, 6.3% at 36 h and 7.2% at 48 h, every
lead significant; P(RI) Brier skill +0.12, AUC 0.84; and calibrated P(≥ 64 / 96 / 113 / 137 kt) at every lead
(Brier skill +0.65 / +0.48 / +0.36 / +0.18 at 24 h).

On Melissa from 27 October, with GFS, GraphCast-GFS and the GEFS mean: 56 / 46 / 69 / 194 / 274 km at 24-96 h
(0.13.1: 91 / 55 / 79 / 246 / 369). The intensity is still far too low (114 kt at 24 h against 155). Each world of
the swarm now follows one model; the parish probabilities are about where 0.13.1 had them. Details and every
caveat are in section 4.15 of the model card.

## 0.15: the ocean's depth, landfall parish by parish, and the benchmark table

The intensity now knows how deep the warm water is: the depth of the 26 C isotherm and the heat potential from
the World Ocean Atlas climatology (packaged), deepened or shoaled by this year's sea-surface anomaly, size the
cold wake and enter the SHIPS and rapid-intensification predictors with the storm's observed size. Landfall is
answered the way a parish needs it (`naturev1.swarm.landfall`): when the centre comes ashore with its 10-90%
range and dates, which parish first, the intensity then and at peak, and each parish's peak wind, gust, rain and
arrival times. The wind field is sized to the observed 34-kt radius, the worlds' intensities drawn from the
forecast's own calibrated error distribution. `naturev1.swarm.observations` fits a vortex to barometer readings
(a centre fix with uncertainty and a central pressure) and blends point winds into the map -- operators for feeds
not yet read here. The track benchmark against every archived model is in section 4.16 of the model card.
**Measured on the held-out seasons (2018, 2020-2022), none of it tuned there.** Intensity: 0.15's tables lower the
error 0.9-1.3% at 6-24 h over 0.14 (16-33% over persistence, 5-16% over SHIFOR at every lead), and lift the
rapid-intensification Brier skill from 0.122 to 0.154 (AUC 0.840 → 0.864). The under-forecast of the wind is conditional,
not a bias: −25 kt at 24 h on the storms that intensify rapidly, +14 kt on those that weaken fast, and the forecast
peak sits 30 kt under the true peak of the storms that reach 113 kt. Two products answer that: the rapid-intensification
path issued beside the consensus with its probability (60% better than the consensus at 24 h on the storms that did go
rapid, 2-3 times worse on those that did not, hence never blended), and a peak-intensity probability -- P(peak within
5 days ≥ 64 / 96 / 113 / 137 kt) -- with held-out Brier skill 0.32 / 0.31 / 0.28 / 0.28 against 0.26 / 0.22 / 0.15 / 0.16
for the best per-lead category probability. Track: the previous cycle's error fed back into the new blend is worth
+1.1-1.2% at 12-24 h (significant) and nothing past 48 h. Rain: R-CLIPER along the observed track of 24 US landfalls
against gauge-corrected radar delivered half the observed rain; a ×1.95 multiplier set on 8 tuning storms raises the
held-out threat score for 100 mm from 0.20 to 0.29 and for 250 mm from 0.12 to 0.29, at the cost of a +13 mm bias in
the light-rain cells. Parish rain is now reported *if the parish is hit* and as its wettest cell, not only as the
swarm mean that every missing world dilutes. Full tables in the model card, section 4.16.

<!-- PENDING readme 0.15 melissa -->

## Not trained

No weights here have seen real data. The architecture, losses, ingest and training loop are complete and
exercised end to end on real GOES-19 imagery, but **nothing here should inform a decision about a real
storm.** The National Hurricane Center is the authoritative source for tropical cyclone forecasts.

Made by Nathan.
