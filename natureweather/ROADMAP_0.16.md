# naturev1 0.16: the engineering roadmap, closed out

The six deficits of 0.15, what the archive said causes each, the experiment that decided each fix, and where it ended.
Every item was held to the same rule as everything before it: fitted on the tuning seasons (2016, 2017, 2019), scored
once on the held-out seasons (2018, 2020-2022), shipped only if it helped there, and reported either way. "Held-out"
below always means those seasons. The matrix the user supplied is kept, with the mechanics corrected where the archive
said otherwise -- a roadmap built on the wrong cause fixes the wrong thing.

| # | deficit | what the data say causes it | fix | state |
|---|---|---|---|---|
| 1 | explosive peak gap: forecast peak 25-37 kt under the true peak of category-5 storms | not a loss-function artefact (the intensity is a simplex-weighted consensus, no network); it is regression to the mean under rapid-intensification uncertainty -- any weight that moves the mean toward the RI path costs the 94% of cases that do not explode | close it in the **distribution**, not the mean: calibrated peak quantiles (50/75/90/95%) and E[peak \| RI]; then the operational guidance as members | **shipped** (`peak_quantiles`, `peak_if_ri`): held-out q90 sits +4 kt above the true peak of the >=137 kt storms where the consensus max sits -37 kt; coverage 0.53/0.77/0.91/0.95; E[peak \| RI] +0.6 kt bias on the storms that went rapid. The a-deck blend (row 7) takes 26-35% off the intensity error at 24-120 h but moves the RI-case bias at 24 h only from -23 to -21 kt (the official forecast: -19): the gap is a property of every mean forecast, which is why the distribution products stay the answer |
| 2 | track: 2-7% behind FGN past 12 h | an equal-weight mean is diluted by weaker members; the live set has no FGN at all | dynamic per-storm model weighting from the last cycles' errors (tested: ~0); NNLS member weights (+0.3-1.1%, not shipped); the NHC a-deck's consensus aids as live members | **shipped** (row 7): the plain mean of our blend and HCCA, chosen on the tuning seasons, lowers the held-out Atlantic track error 11-13% at 12-120 h and beats the official forecast by 12-22% at 6-72 h |
| 3 | light-rain bloat from the rain multiplier | the tuning storms are wetter than the held-out ones (obs/forecast 1.9 vs 1.5), so a pooled multiplier runs high in the periphery; it is a per-storm scale error, largest in the cores of slow movers, not a radial-tail problem | profile reshaped (tail x1.5, front-right asymmetry 0.3, multiplier 1.19); the model's own rain as a member | **shipped**: reshape (held-out CSI 250 mm 0.295 -> 0.348, strong storms' area >=250 mm 3.2x -> 1.7x) and the GFS accumulated precipitation travelling with every world as half of its rain (radar check on US landfalls: Ida 23 vs 45 mm MAE, Nicholas 17 vs 55, Fred 21 vs 36, Ian level). Rejected on the tuning storms: every nonlinear heavy-favouring transform, a 115-kt saturation, a translation-speed factor |
| 4 | terrain and moisture: rain under the wind, not where two days of easterlies fed the Blue Mountains | a parametric profile has no moisture flux and no outer bands; the orographic term is a slope x wind product without moisture; the multiplier was calibrated on flat ground | reshaped profile; the model-QPF member, which carries the model's moisture and its own orography | **partly**: east of Melissa's track right since the reshape (Portland 470 vs 536 mm, St Catherine 624 vs 668, Kingston 447 vs 567); the QPF member is the moisture term; an integrated-vapour-transport orographic scheme needs a calibration set that does not exist yet |
| 5 | stalls over water held too strong | the continuous depth-aware cooling lost on the tuning seasons (15.3 -> 15.4-15.6 kt); a stall rule (speed < 3 kt for > 12 h, D26 < 50 m, -5 to -15 kt) was tested rather than assumed | penalty fitted on stall cases, with the deep-water sign check | **rejected**: the sign of the stall residual flips between the tuning and the held-out seasons, and the fixed rule is 45-122% worse on the stall cases themselves; Melissa stalled over 120 m of 26 C water and intensified. Nothing shipped; the wake cooling already in the SHIPS-style members stands |
| 6 | landfall timing 3-4 h early | hypothesis was a missing coastal slowdown; measured instead as along-track bias by speed, lead and distance to land | along/cross decomposition, speed-dependent correction, arrival-time check | **measured, no correction**: no systematic along-track bias (+1/+4/+5/+2/-1 km at 6-48 h); the arrival-time error is 3.6 h at 24 h overall and 7.3 h for storms moving under 3 m/s -- so the error is reported as a window (`ARRIVAL_TIME_ERROR`, in the landfall report) and nothing is nudged |
| 7 | the operational guidance was not in the room | the NHC a-deck carries every aid's track and intensity minutes after each cycle; nothing read it | `naturev1.swarm.adeck` reads the public a-deck (live and archive); `naturev1.cyclone.aids` blends the aids in with tables fitted on the tuning seasons | **shipped**: intensity -- simplex weights over our consensus, IVCN, HWRF, HMON, SHIPS, LGEM, GFS and HCCA (leave-one-season-out gains 13-28% at 24-96 h in every tuning season; held-out 26-35% at 24-120 h, 4-6% ahead of the official forecast at 24-72 h); track -- the mean with HCCA (row 2). HAFS-A/B stand in for HWRF/HMON and TVCN for HCCA when those are absent (lineage stand-ins, reported as such) |
| 8 | the NHC's rapid-intensification guidance was not in the room | the SHIPS text carries the SHIPS-RII, logistic, Bayesian, DTOPS and consensus probabilities live; the e-deck carries them after the fact | read both; verify against ours on the archive | **shown beside ours, not the headline**: the archive's e-decks carry no RI rows before 2020, so nothing could be chosen on the tuning seasons; on 2020-2022 (453 cases, 24 events) ours has Brier skill 0.13, the SHIPS-RII consensus 0.07, DTOPS 0.17 and the equal-weight mean of ours and DTOPS 0.20 -- that mean is reported in the record and the cell as `ri_probability_combined`, with this caveat attached |

## Shipped in 0.16

* **The NHC guidance as members** -- `naturev1.swarm.adeck` (a-deck, e-deck, SHIPS text), `naturev1.cyclone.aids`, `coefficients.ADECK_TRACK` / `ADECK_INTENSITY`; `forecast_storm(guidance=True)` by default; `track_ours` / `intensity_ours` keep our own forecast beside the blend; the official forecast and the RI guidance in `run["guidance"]` and the cycle record.
* **Rain** -- `RAIN_SCALE = {multiplier 1.19, tail 1.5, asymmetry 0.3 front-right}`; orographic cap 2; the GFS QPF member (`models.fetch_qpf`, `qpf_weight` 0.5).
* **Peak quantiles and E[peak | RI]** -- `coefficients.PEAK_QUANTILES`, `PEAK_IF_RI`; in the landfall report ("plan for the 90% figure").
* **Arrival-time window** -- `coefficients.ARRIVAL_TIME_ERROR`; in the landfall report.
* **Operations layer** -- `naturev1.swarm.bulletin` (every cycle saved as JSON + parish CSV; the previous cycle found again and fed back automatically), `naturev1.swarm.infrared` (GOES band-13 structure diagnostics with trend), `forecast_storm(previous="auto")`, the molab cell with its watch loop.
* **Audit fixes** -- missing physics context reported, swarm rescale keeps the longitude frame across the dateline, 24/48 h indices follow the caller's leads, silent fallbacks leave `intensity_note`, NHC-only ids, previous analysis position as the first feedback knot.
* **The swarm sized to the flow-dependent cone** (0.15.0, verified on 396 held-out forecasts).

## Tried and rejected (kept here so nobody tries them again without new evidence)

Stall penalty (row 5); translation-speed rain factor and 115-kt saturation; nonlinear rain transforms; dynamic per-storm
model weighting; NNLS physics-member weights (+0.3-1.1%, inside the noise); the RI-path mixture into the mean; the
continuous depth-aware wake cooling; CHIRPS as rain truth (too coarse at the parish scale); the fitted simplex over
the track aids (within 0.2% of the plain mean on the tuning seasons, so the mean ships).

## Real-time data still to bring in (each needs a reader and a verification before it changes a number)

* NESDIS ocean heat content (26 C depth, TCHP) on CoastWatch ERDDAP, daily -- replaces the climatology under the storm.
* ECMWF open-data TC-track BUFR for IFS-ENS, AIFS-single and AIFS-ENS (51 + 51 member tracks and intensities).
* HAFS-A/B tracks (3-hourly to 126 h, with radii) on NOMADS -- the a-deck already carries their interpolated runs.
* CIMSS ADT / SATCON / AiDT / D-MINT text feeds (30-minute satellite intensity, radii) and STAR SAR winds -- fixes with intensity for the particle filter.
* GLM lightning and GOES band-13 structure as RI predictors (an archive of storm-centred imagery is needed to fit them).
* NHC wind-speed probabilities and arrival-time grids, STOFS-2D water levels and the Jamaica SLOSH MOM layer for a surge pillar.

## The benchmark we publish: the answer key

`benchmark_heldout_atlantic_2018_2020_2022.csv.gz` on the model repository: the 849 held-out Atlantic cases (storm,
time, position, intensity) with the best-track truth at every lead and every reference's forecast or error on it --
NatureV1 0.14/0.16 tracks and 0.15/0.16 intensities, the NHC official forecast and its aids (HCCA, TVCN, IVCN, HWRF,
HMON, SHIPS, LGEM, GFS, GEFS mean), the archived models' track errors (HRES, Pangu, GraphCast, FuXi, Aurora, FGN),
CLIPER and extrapolation. `naturev1.cyclone.verify.compare` scores any new forecast against it on the same storms,
with the storm-block bootstrap, so that a claim about a hurricane model can be checked rather than believed.
