# naturev1 0.16: the engineering roadmap

The six deficits of 0.15, what the archive says causes each, the experiment that decides each fix, and its state.
Every item is held to the same rule as everything before it: fitted on the tuning seasons (2016, 2017, 2019), scored
once on the held-out seasons (2018, 2020-2022), shipped only if it helps there, and reported either way. "Held-out"
below always means those seasons. The matrix the user supplied is kept, with the mechanics corrected where the
archive says otherwise -- a roadmap built on the wrong cause fixes the wrong thing.

| # | deficit | what the data say causes it | fix | state |
|---|---|---|---|---|
| 1 | explosive peak gap: forecast peak 25-37 kt under the true peak of category-5 storms | not a loss-function artefact (the intensity is a simplex-weighted consensus, no network); it is regression to the mean under rapid-intensification uncertainty -- any weight that moves the mean toward the RI path costs the 94% of cases that do not explode | close it in the **distribution**, not the mean: calibrated peak quantiles (50/75/90/95%) and E[peak \| RI] | **shipped** (`peak_quantiles`, `peak_if_ri`): held-out q90 sits +4 kt above the true peak of the >=137 kt storms where the consensus max sits -37 kt; coverage 0.53/0.77/0.91/0.95; E[peak \| RI] +0.6 kt bias on the storms that went rapid |
| 2 | track: 2-7% behind FGN past 12 h | an equal-weight mean is diluted by weaker members; the live set has no FGN at all | dynamic per-storm model weighting from the last cycles' errors; dilution test; spread-dependent physics share; the NHC a-deck's operational aids (HAFS, HWRF, GDMN, TVCN/HCCA) as live members, verified on the archive | **in progress** (track-weighting and a-deck agents) |
| 3 | light-rain bloat from the rain multiplier | the tuning storms are wetter than the held-out ones (obs/forecast 1.9 vs 1.5), so a pooled multiplier runs high in the periphery; it is a per-storm scale error, largest in the cores of slow movers, not a radial-tail problem | profile reshaped (tail x1.5, front-right asymmetry 0.3, multiplier 1.19); nonlinear heavy-favouring transforms tested | **partly**: reshape shipped (held-out CSI 250 mm 0.295 -> 0.348, strong storms' area >=250 mm 3.2x -> 1.7x); every nonlinear transform overfit the tuning storms and was rejected; next: condition the multiplier on something observed at forecast time (model QPF, precipitable water, speed) |
| 4 | terrain and moisture: rain under the wind, not where two days of easterlies fed the Blue Mountains | a parametric profile has no moisture flux and no outer bands; the orographic term is a slope x wind product without moisture; the multiplier was calibrated on flat ground | reshaped profile (east of Melissa's track now right: Portland 470 vs 536 mm, St Catherine 624 vs 668, Kingston 447 vs 567); model-QPF member and integrated-vapour-flux orography under study | **partly**: east fixed, far west still high (Westmoreland 630 vs 309); QPF-member and rain-science agents reporting |
| 5 | stalls over water held too strong | the continuous depth-aware cooling lost on the tuning seasons (15.3 -> 15.4-15.6 kt); a stall-specific rule (speed < 3 kt for > 12 h, D26 < 50 m, -5 to -15 kt) is being tested rather than assumed | penalty fitted on stall cases, with the deep-water sign check (Melissa stalled over 120 m of 26 C water and intensified) | **in progress** (ocean-stall agent) |
| 6 | landfall timing 3-4 h early | hypothesis: missing coastal slowdown; to be measured as along-track bias by speed, lead and distance to land before anything is corrected | along/cross decomposition, speed-dependent along-track correction, arrival-time check | **in progress** (timing agent) |

## Shipped in 0.16 so far

* **Rain shape** -- `RAIN_SCALE = {multiplier 1.19, tail 1.5, asymmetry 0.3 front-right}`; orographic cap 2.
* **Peak quantiles and E[peak | RI]** -- `coefficients.PEAK_QUANTILES`, `PEAK_IF_RI`; in the landfall report ("plan for the 90% figure").
* **Operations layer** -- `naturev1.swarm.bulletin` (every cycle saved as JSON + parish CSV; the previous cycle found again and fed back automatically), `naturev1.swarm.infrared` (GOES band-13 structure diagnostics with trend), `forecast_storm(previous="auto")`.
* **Audit fixes** -- missing physics context reported, swarm rescale keeps the longitude frame across the dateline, 24/48 h indices follow the caller's leads, silent fallbacks leave `intensity_note`, NHC-only ids, previous analysis position as the first feedback knot.
* **The swarm sized to the flow-dependent cone** (0.15.0, verified on 396 held-out forecasts).

## Under evaluation (agents on the archive)

1. Timing: along-track bias by translation speed, lead, latitude, intensity and distance to land; speed-dependent correction; arrival-time error.
2. Track weighting: recent-error weighting per storm, FGN dilution test, lead-dependent weights, spread-dependent physics share; which findings transfer to the live set (GFS, IFS, AIFS, GraphCast-GFS, GEFS).
3. NHC a-deck: homogeneous verification of OFCL, TVCN, HCCA, HAFS, HWRF, HMON, CTCX, UKMET, CMC, GEFS mean, DSHP, LGEM, IVCN against our cases; equal-weight blends; the official forecast's own speed bias.
4. Ocean stall: the stall penalty rule on stall cases over shallow and deep water.
5. Model QPF: GFS accumulated precipitation as a rain member, against radar.
6. Rain science: PHRaM/IPET asymmetry coefficients, intensity saturation, orographic schemes with moisture, open verification data.

## Real-time data to bring in (from the data survey; each needs a reader and a verification before it changes a number)

* NESDIS ocean heat content (26 C depth, TCHP) on CoastWatch ERDDAP, daily -- replaces the climatology under the storm.
* The public a-deck and e-deck: every operational track and intensity aid and the operational RI probabilities, minutes after each cycle.
* ECMWF open-data TC-track BUFR for IFS-ENS, AIFS-single and AIFS-ENS (51 + 51 member tracks and intensities).
* HAFS-A/B tracks (3-hourly to 126 h, with radii) on NOMADS.
* CIMSS ADT / SATCON / AiDT / D-MINT text feeds (30-minute satellite intensity, radii) and STAR SAR winds -- fixes with intensity for the particle filter.
* GLM lightning and GOES band-13 structure as RI predictors (an archive of storm-centred imagery is needed to fit them).
* NHC wind-speed probabilities and arrival-time grids, STOFS-2D water levels and the Jamaica SLOSH MOM layer for a surge pillar.

## The benchmark we publish

The held-out case list and every reference's error on it (track: extrapolation, CLIPER, HRES, Pangu, GraphCast, FuXi,
Aurora, FGN, and the a-deck aids once matched; intensity: persistence, SHIFOR, SHIPS, LGEM, the a-deck aids; rain: the
MRMS storm set; probabilities: RI, peak, categories, cone coverage) will be released as a frozen table with the code
that scores a new model against it, so that a claim about a hurricane model can be checked on the same storms.
