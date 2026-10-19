# NatureV1 0.16 -- the results, and the record of what was built

CorxLabs, Westmoreland, Jamaica. October 2026. This is the complete account of the hurricane engine as it stands at
0.16.0: what the architecture is, how every number was earned, what was tried and rejected, what shipped, and where it
is published. Every figure here comes from the held-out seasons unless it says otherwise; nothing was tuned on them.

## 1. What the system is

NatureV1 is a weather model with a hurricane engine around it. Three layers:

1. **iHELIX**, the engine: attention over physical space on the sphere (conservative regrid, spherical convolution,
   binning), the kernels the model is built from (`ihelix`, on PyPI).
2. **NatureV1-500**, the model: the global atmosphere at 1.5 degrees with storm heads (track, intensity, rapid
   intensification, eyewall, landfall), every output a distribution, trained in stages (stage 1 the atmosphere,
   stage 2 the storm heads on the storm bank, stage 3 the satellite path: SatNet-Storm 2.0 and the global satellite
   retrieval on GridSat/HURSAT/GOES band 13), published at `CorxLabs/NatureV1-500` with its checkpoints
   (`checkpoints/stage1`, `stage2`, `rollout`, the storm banks), config and normaliser.
3. **The hurricane physics engine** (`naturev1.cyclone`, `naturev1.swarm`; versions 0.13 to 0.16), which is what
   this document is about. It is deterministic physics plus fitted tables, no network in the loop, and it is the part
   verified against every archived reference:
   * **Track.** A consensus of physics members (extrapolation, CLIPER rebuilt, beta-and-advection steering from the
     analysis at several depths, an analog member, the forecast-flow member that follows the dynamical model's own
     steering), simplex weights per lead; blended with the mean track of the live models (GFS, ECMWF IFS and AIFS,
     GraphCast-GFS, the GEFS mean, each followed at native resolution); the previous cycle's error fed back; and
     since 0.16 the NHC's operational aids blended in (section 4).
   * **Cone and swarm.** A flow-dependent cone (Goerss-style regression of the blend's error on member spread, model
     disagreement, speed, latitude, intensity) and a particle-filter swarm of up to a million worlds, each world
     steered by one model's flow, filtered on every satellite, recon and radar centre fix, sized to the cone,
     re-weighted by fixes that arrive after the analysis. It gives scenario probabilities (stall, loop, recurve,
     steady), strike probabilities and the parish impacts.
   * **Intensity.** SHIFOR, decay-SHIFOR, a SHIPS-style statistical-dynamical member, LGEM, a trend member and the
     models' own deepening, in a simplex consensus with a rapid-intensification regime; a 17-feature logistic P(RI);
     the RI path as a branch beside the mean; ocean heat content and 26 C isotherm depth from the World Ocean Atlas
     under the storm; peak probabilities P(peak >= 64/96/113/137 kt), calibrated peak quantiles and E[peak | RI];
     category probabilities per lead; since 0.16 the NHC intensity aids blended in.
   * **Impacts.** A parametric asymmetric wind field sized to the observed 34-kt radius, land decay, gusts, arrival
     times; rain from R-CLIPER reshaped against gauge-corrected radar with an orographic term, and since 0.16 the
     GFS's own precipitation travelling with every world; painted onto every parish, province and district of the
     Caribbean with terrain; the landfall report (when, where first, how strong, each parish's numbers, the
     arrival-time window).
   * **Operations.** One call from an NHC storm id (`naturev1.swarm.live.forecast_storm`), live or replayed as it was
     known; every cycle saved as JSON and a parish CSV; the previous cycle found again automatically; GOES band-13
     read quantitatively; the molab cell that runs all of it and keeps watching NHC.

**The protocol that every number obeys.** Fitted on the tuning seasons 2016, 2017 and 2019. Scored once on the
held-out seasons 2018, 2020, 2021 and 2022 (the Atlantic where the NHC guidance exists; all basins for the physics).
Significance by a storm-block bootstrap (a star marks a whole 90% interval above zero). Shipped only if it helped
held-out, reported either way. The coefficients module records the provenance of every table.

## 2. Where it stands against the references (held-out)

**Track, km, Atlantic 2018 / 2020-2022, homogeneous (749 / 703 / 626 / 554 / 490 / 374 / 281 / 206 cases at
6 / 12 / 24 / 36 / 48 / 72 / 96 / 120 h, 79 storms).**

| | 6 h | 12 h | 24 h | 36 h | 48 h | 72 h | 96 h | 120 h |
|---|---|---|---|---|---|---|---|---|
| NatureV1 0.14 blend (physics + live models) | 22.1 | 40.1 | 62.2 | 82.3 | 95.5 | 168.7 | 260.5 | 365.1 |
| NatureV1 0.16, mean with HCCA | 21.5 | 35.7 | 54.1 | 72.5 | 89.4 | 149.6 | 227.7 | 318.7 |
| **NatureV1 0.16 shipped (earned split + outlier gate)** | **22** | **36** | **53** | **72** | **89** | **149** | **225** | **317** |
| HCCA (HFIP corrected consensus) | 29.9 | 40.7 | 60.3 | 80.7 | 104.6 | 170.5 | 253.8 | 363.9 |
| TVCN (track consensus) | 29.8 | 41.8 | 62.7 | 84.5 | 109.5 | 170.0 | 248.8 | 346.5 |
| NHC official forecast (OFCL) | 27.6 | 42.4 | 64.1 | 84.9 | 109.2 | 169.4 | 241.6 | 334.4 |

The 0.16 blend is 11-13% under 0.14 at 12-120 h (every lead significant) and 12-22% under the official forecast at
6-72 h, 5-6% at 96-120 h. The aids' 6 h values are interpolations between their analysis and their 12 h point, so
read the aids from 12 h on.

**Track against the archived models, all basins (from 4.16 of the model card; each reference on its own cases).**
NatureV1 beats HRES (+18% at 24 h, +24% at 72 h), Pangu (+9%, +8%), GraphCast (+4%, +11%), Aurora (+6%, +16%) at
every lead to day 5, is level with FuXi past 24 h, and trails FGN (WeatherNext 2) by 5-7% from 24 h on its 787
cases. Against the single dynamical models in the a-deck (GFS, HWRF, HMON, COAMPS-TC, the GEFS mean) it is better at
every lead by 6-39%.

**Intensity, kt mean absolute error, Atlantic held-out, homogeneous with the official forecast.**

| | 6 h | 12 h | 24 h | 36 h | 48 h | 72 h | 96 h | 120 h |
|---|---|---|---|---|---|---|---|---|
| persistence / SHIFOR (0.15's own references) | -- | -- | 0.15 is 16-33% better than persistence and 5-16% better than SHIFOR at every lead | | | | | |
| NatureV1 0.15 consensus | 3.4 | 6.0 | 9.6 | 12.2 | 13.9 | 16.2 | 18.4 | 18.8 |
| **NatureV1 0.16 (0.15 + the aids)** | **3.3** | **5.4** | **7.1** | **8.5** | **9.5** | **10.5** | **12.5** | **12.8** |
| IVCN (intensity consensus) | 4.3 | 5.5 | 7.2 | 8.5 | 9.4 | 10.7 | 12.7 | 13.8 |
| HCCA | 4.3 | 5.5 | 7.2 | 8.7 | 9.6 | 11.1 | 12.5 | 13.0 |
| HWRF (HWFI) | 4.6 | 6.2 | 8.0 | 9.4 | 10.6 | 13.3 | 15.3 | 16.3 |
| SHIPS with decay (DSHP) | 4.5 | 6.1 | 8.4 | 10.2 | 11.6 | 13.4 | 16.3 | 18.3 |
| LGEM | 4.5 | 6.4 | 9.0 | 11.1 | 12.2 | 13.3 | 15.8 | 17.5 |
| NHC official forecast | 3.6 | 5.0 | 7.4 | 8.8 | 10.1 | 11.2 | 12.4 | 13.4 |

0.16 is 26-35% under 0.15 from 24 h on (every tuning season agreed in leave-one-season-out, 13-28% at 24-96 h) and
4-6% under the official forecast at 24-72 h, level at 96-120 h. The official forecast is itself written from this
guidance three hours later: read it as a product comparison.

**Rapid intensification, P(+30 kt in 24 h).** Our logistic (17 predictors: the SHIPS-style set, the models' deepening,
ocean heat content and 26 C depth, the gale radius): held-out Brier skill 0.154, AUC 0.864 (0.14: 0.122, 0.840). On the
2020-2022 cases where the NHC e-deck carries RI rows (there are none in the archive before 2020): ours 0.125, the
SHIPS-RII consensus 0.068, the deterministic SHIPS-RII 0.020, the logistic -0.070, the Bayesian 0.074, DTOPS 0.171;
the equal-weight mean of ours and DTOPS 0.199. That mean is reported beside ours, not as the headline, because it
could not be chosen on a tuning season.

**Peak intensity.** P(peak within 5 days >= 64 / 96 / 113 / 137 kt): held-out Brier skill 0.32 / 0.31 / 0.28 / 0.28
against 0.26 / 0.22 / 0.15 / 0.16 for the best per-lead category probability. Calibrated peak quantiles: coverage
0.53 / 0.77 / 0.91 / 0.95 at the 50 / 75 / 90 / 95% levels; the 90% quantile sits +4 kt above the true peak of the
>= 137 kt storms where the consensus maximum sits -37 kt. E[peak | RI]: +0.6 kt bias, 13 kt MAE on the storms that went
rapid. The bias that remains is conditional: -21 kt at 24 h on the storms that intensify rapidly (0.15: -23; the
official forecast: -19), +14 kt on those that weaken fast.

**Cone and swarm.** The flow-dependent cone replaces the climatological one; the swarm's raw worlds were 1.4-3.2 times
too wide (the truth inside the nominal 67% radius 81-99% of the time); sized to the cone, 396 held-out forecasts put
the truth inside the 50 / 67 / 90% radii 50-63 / 64-75 / 85-90% of the time.

**Rain.** R-CLIPER along the observed track of 24 US landfalls against NOAA MRMS gauge-corrected radar delivered half
the observed rain. The reshaped profile (multiplier 1.19, radial tail x1.5, front-right asymmetry 0.3) lifts the
held-out threat score for 100 mm to 0.296 and for 250 mm from 0.12 to 0.348, and brings the strong storms' area
>= 250 mm from 3.2x observed to 1.7x. The GFS's accumulated precipitation moved with each world, at half weight:
Ida 23 vs 45 mm mean error, Nicholas 17 vs 55, Fred 21 vs 36, Ian level.

**Timing.** No systematic along-track bias (+1 / +4 / +5 / +2 / -1 km at 6-48 h). Arrival-time error 3.6 h at 24 h,
7.3 h for storms under 3 m/s: reported as a window in the landfall report, nothing corrected. The previous cycle's
error fed back into the track: +1.1-1.2% at 12-24 h, nothing past 48 h.

## 3. The 0.16 work, item by item

The release started from six named deficits (the roadmap matrix) and ended with eight rows, each with a state.

| deficit | what the archive said | what shipped / what was rejected |
|---|---|---|
| explosive peak gap (forecast peak 25-37 kt under category-5 peaks) | regression to the mean under RI uncertainty, not a loss artefact; any weight toward the RI path costs the 94% that do not explode | **shipped**: calibrated peak quantiles, E[peak \| RI]; the aids take 26-35% off the intensity error but move the RI-case bias only -23 -> -21 kt |
| track 2-7% behind FGN past 12 h | an equal-weight mean diluted by weaker members; no FGN live | **shipped**: the NHC aids (HCCA) as a member, -11-13%; **rejected**: dynamic per-storm model weighting (~0), NNLS member weights (+0.3-1.1%, noise) |
| light-rain bloat | a per-storm scale error from wetter tuning storms, not a radial-tail problem | **shipped**: the reshaped profile and the GFS rain member; **rejected**: every nonlinear transform, a 115-kt saturation, a speed factor |
| terrain and moisture | no moisture flux in a parametric profile | **partly**: east of Melissa right since the reshape; the QPF member is the moisture term; IVT orography needs a calibration set |
| stalls over water held too strong | the sign of the stall residual flips between seasons | **rejected**: the fixed rule is 45-122% worse on the stall cases; Melissa stalled over 120 m of 26 C water and intensified |
| landfall timing 3-4 h early | no along-track bias; the error is a spread, largest for slow movers | **measured, reported as a window**, nothing nudged |
| the operational guidance was not in the room | the a-deck carries every aid minutes after the cycle | **shipped**: `naturev1.swarm.adeck` + `naturev1.cyclone.aids`, two fitted tables, stand-ins by lineage (HAFS for HWRF/HMON, TVCN for HCCA) |
| the NHC RI guidance was not in the room | the SHIPS text carries five schemes live | **shown beside ours** with the equal-weight mean and the no-tuning-sample caveat |

**The averaging penalty**, raised on the Melissa 26 Oct 12 UTC cycle (ours 15 km at 24 h, HCCA 70, the mean 42), taken
to the archive:

| held-out cases | n | ours alone | HCCA | TVCN | 50/50 mean | ours closer than HCCA | ours the outlier against TVCN |
|---|---|---|---|---|---|---|---|
| all with HCCA | 641 | 62 km | 60 | 63 | 54 | -- | -- |
| the 20% that disagree most (>= 92 km apart at 24 h) | 159 | 133 | 93 | 95 | 94 | 31% | 97% |
| the 10% that disagree most (>= 128 km apart) | 80 | 220 | 101 | 106 | 138 | 16% | 100% |

When the two disagree strongly, our standalone is the one that is wrong two times in three. Rejected on the tuning
seasons: ours 2/3, a gate toward ours, the median and the trimmed mean of ours/HCCA/TVCN. Shipped: (1) **weights earned
from the previous cycle** -- the previous forecasts of ours and of HCCA verified against the position just analysed,
`sigmoid(0.5 log(e_HCCA / e_ours))` within 25-75% at 6-24 h; (2) **the outlier gate** -- beyond 1.5 cone radii of
disagreement up to 10% of the pair's weight moves away from whichever member is further from TVCN, from 12 h on.
Together on held-out: 1.4 / 2.5 / 1.1 / 0.7 / 1.0 / 1.9 / 2.3% lower error at 12 / 24 / 36 / 48 / 72 / 96 / 120 h, every
lead significant. The disagreement is reported per lead and both tracks are in the record.

**Also built in 0.16:** the GOES band-13 diagnostics with trend (`infrared.py`); the cycle record and history
(`bulletin.py`); `forecast_storm(previous="auto", guidance=True)`; the molab cell rewritten with the watch loop; audit
fixes (dateline-safe swarm rescale, caller's leads for the 24/48 h indices, silent fallbacks leave a note, NHC-only
ids, the previous analysis position as the first feedback knot); the benchmark table.

## 4. Melissa (AL13 2025), replayed with what was known at the time

| forecast from | before landfall | P(landfall, Jamaica) | centre ashore, median (10-90%) | consensus landfall | intensity then (10-90%) | RI branch (its probability) | P(peak >= cat 4 / cat 5) | track error 24 / 48 / 72 h: 0.16 (own; official) |
|---|---|---|---|---|---|---|---|---|
| 26 Oct 00 UTC | 65 h | 72% | 28 Oct 09 UTC (27 Oct 21 -> 28 Oct 21) | 28 Oct 06 UTC, Clarendon | 96 kt (64-132) | 124 kt, peak 130 (23%) | 62% / 28% | 44 / 121 / 195 km (42 / 158 / 277; 34 / 123 / 124) |
| 26 Oct 12 UTC | 53 h | 83% | 28 Oct 12 UTC (06 -> 21) | 28 Oct 11 UTC, St Elizabeth | 101 kt (71-136) | 151 kt, peak 152 (19%) | 100% / 31% | 42 / 72 / 147 km (15 / 53 / 161; 40 / 54 / 73) |
| 27 Oct 00 UTC | 41 h | 83% | 28 Oct 15 UTC (09 -> 21) | 28 Oct 14 UTC, St Elizabeth, 18.06N 77.72W | 98 kt (78-121) | 149 kt, peak 160 (8%) | 100% / 23% | 54 / 71 / 212 km (38 / 69 / 215; 79 / 79 / 164) |

Actual: landfall near New Hope, St Elizabeth (18.09N 77.97W) about 17 UTC on 28 October at 160 kt; peak 165 kt. Right:
the landfall point and probability, the parish ranking (St Elizabeth 77%, Manchester 69%, Westmoreland 60%, Clarendon
54%; the east 10-15%), the far-west rain (Westmoreland 413 mm if hit against the 309 mm gauge where 0.15 said 689;
Manchester 584 / 865 against 817; St Catherine 468 / 666 against 668). Wrong, said plainly: the intensity -- the aids
were all low on this storm and pulled the 48 h intensity to 85-100 kt against 105-165; the explosive case stayed a tail
event (the 95% peak quantile 133-140 kt against 165), as it did for the official forecast (140 kt at 24 h against 155);
parish timing about four hours early.

## 5. Published

* **PyPI**: `naturev1` 0.16.0 (wheel and sdist; install verified from the index). `pip install "naturev1[swarm,hub]"`.
* **Hugging Face** `CorxLabs/NatureV1-500`: the model card (section 4.17 carries the 0.16 tables), `melissa_landfall.png`,
  `melissa_jamaica.png`, `benchmark_heldout_atlantic_2018_2020_2022.csv.gz` (849 cases, 120,472 rows: the truth and
  every reference's forecast or error at every lead; `naturev1.cyclone.verify.compare` scores a new model on it).
* **Git**, branch `claude/linear-context-architecture-a35c6h` of `Wisk-inc/transformers`, `natureweather/`: commits
  `ed86e0e` (rain shape, operations layer, infrared, audit), `5eae22b` (peak quantiles, arrival window, QPF member,
  cell, roadmap), `b3d5a09` (the guidance layer), `4a5917b` (Melissa), `4ac6e53` (earned weights), `45e92bd` (the
  outlier gate). Tests: 403 passed, 1 skipped, plus the 0.16 additions (`tests/test_adeck.py`, `test_peak.py`,
  `test_operational.py`).
* **The cell**: `molab_swarm_cell.py`, one marimo cell, live or replay.

## 6. Limits that remain, and the next steps

* The explosive cases are tail events to every mean forecast here, ours and the NHC's; the answer stays in the
  distribution (quantiles, the RI branch) until a predictor that separates them exists. The 0.12 satellite path is the
  candidate: GOES structure and lightning as RI predictors need an archive of storm-centred imagery to fit.
* The guidance tables were fitted on Atlantic storms; an eastern Pacific storm gets the same weights, unverified there.
  HWRF and HMON appear in about half the 2026 cycles; the blend renormalises and the record says what it used.
* The cone is sized for the track before the aids; with the aids the error is 11-13% smaller, so coverage runs a
  little above nominal. A refit on the blended track is straightforward.
* The RI blend with the NHC schemes needs a tuning sample: build 2023-2025 as a new tuning set.
* A split swarm when ours and HCCA disagree beyond the cone (half the worlds on each track) would show both scenarios
  in the threat map instead of a midpoint neither predicts; it needs a swarm re-verification on the 396 cases.
* Real-time data still to bring in: NESDIS ocean heat content, ECMWF ENS tracks (IFS-ENS, AIFS-ENS), HAFS-A/B tracks,
  CIMSS ADT/SATCON/AiDT, NHC wind-speed probabilities, STOFS and SLOSH for surge.

## 7. NatureV2-500

`CorxLabs/NatureV2-500` starts from the NatureV1-500 stage-1 checkpoint (`checkpoints/stage1/latest.pt`, with its
`config.json` and `normalizer.json`) and carries this code base. The plan: fine-tune the satellite encoder (the
GOES/GridSat band-13 path of 0.12, SatNet-Storm 2.0 and the global retrieval) on the storm-centred imagery, add the
new features the 0.16 verification asked for (infrared structure and its trend, lightning, the ocean heat feed) as
inputs to the storm heads and to the RI probability, and verify them on the same held-out seasons with the same
benchmark table before any of it changes a number in the operational path.
