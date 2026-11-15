# A learning consensus for tropical cyclone tracks from six public forecast systems

**Pluvium technical paper 4.** Nathan, November 2026.

## Summary

We build tropical cyclone track forecasts from six freely available global systems: two physical ensembles
(ECMWF ENS, NOAA GEFS), two physical high-resolution runs (ECMWF HRES, NOAA GFS) and two machine-learning models
(ECMWF AIFS, NOAA GraphCast). Every storm is tracked in every member from sea-level pressure alone. The tracks are
combined with weights that are learned through the season from forecasts that have already verified.

On an untouched test season (East Pacific 2025, 167 forecasts of 18 storms) the consensus was better than every
single system at nearly every lead from 12 to 120 hours, including each one as tracked by the US National Hurricane
Center (NHC). The exceptions: it was level with ECMWF AIFS to 36 hours, and behind the NHC's GraphCast track at 120
hours. It was level with the NHC's simple consensus aid. It was behind the NHC official forecast at every lead, by 2 to 25 nautical
miles, and behind Google DeepMind's cyclone model. On a development season (North Atlantic 2025) it was ahead of the
official forecast at every lead, which is exactly why a development season cannot be the evidence.

The uncertainty products held up on the test season. A circle meant to hold the centre two times in three held it
70% of the time. The chance of the centre passing within 150 km of a place scored a Brier skill of 0.47 against the
base rate, where the raw cloud of ensemble members scored 0.42 and a yes/no read off the track scored below zero.

## 1. Why another consensus

A consensus of good track forecasts beats its members. That has been known for decades, and the NHC's own aids
(TVCN, HCCA) are built on it. Three things have changed.

1. Machine-learning models now forecast tracks as well as the best physical models, and err differently.
2. Their output, and that of the physical ensembles, is public within hours.
3. Models are upgraded between seasons, so weights fixed from past seasons age quickly.

The question here is narrow. With only public fields, a tracker of our own, and no forecaster, how close can a
consensus that learns as it goes come to the official forecast?

## 2. Method

### 2.1 Tracking

For each run we read mean sea-level pressure every 12 hours to 120 hours over the basin (5 to 60 N, 105 to 5 W for
the Atlantic). The tracker starts from the storm's position and motion as known in real time (the `CARQ` record of
the NHC's decks). At each step it looks near the position expected from the previous motion for the nearest closed
low, measures its depth against a ring around it, and refines the centre below the grid spacing with a parabola. A
member that loses the storm stops there. An ensemble's track is the mean of its members' positions on the sphere,
taken only where at least 40% of the members still carry the storm.

The tracker was checked against the NHC's own tracks of the same model runs, on the forecasts both have, in 2025.

| Mean error, nautical miles, ours / NHC's | 24 h | 48 h | 72 h | 120 h |
|---|---|---|---|---|
| ECMWF HRES, Atlantic | 31 / 30 | 63 / 61 | 98 / 97 | 157 / 157 |
| ECMWF HRES, East Pacific | 34 / 33 | 52 / 52 | 71 / 71 | 119 / 119 |
| NOAA GFS, Atlantic | 44 / 45 | 80 / 79 | 118 / 117 | 294 / 297 |
| NOAA GFS, East Pacific | 38 / 36 | 60 / 56 | 89 / 78 | 140 / 138 |
| ECMWF ENS mean, Atlantic | 28 / 28 | 56 / 57 | 92 / 91 | 154 / 161 |
| ECMWF ENS mean, East Pacific | 32 / 31 | 53 / 49 | 73 / 68 | 121 / 124 |
| GraphCast, Atlantic | 32 / 25 | 45 / 38 | 76 / 55 | 170 / 128 |
| GraphCast, East Pacific | 30 / 31 | 46 / 36 | 67 / 53 | 117 / 79 |

For ECMWF HRES the two trackers agree to within 3 nautical miles everywhere. For GFS and the ensemble means ours is
level in the Atlantic and a few miles worse in the East Pacific. For GraphCast our tracks are 25% to 35% worse at
long leads than the `GRPH` aid in the decks. We read NOAA's GFS-initialised GraphCast; `GRPH` may be another
configuration, and we could not settle which. Either way this member is weaker here than it could be.

### 2.2 Weights that learn

Let `e_s` be the error vector of system `s` for a verified forecast. The weights that minimise the expected squared
error of the weighted mean are proportional to the inverse error covariance applied to a vector of ones. We keep the
sums of products that define that covariance up to date after every verified forecast, with four safeguards.

- Each forecast's errors are divided by the systems' own mean error on it (with a floor at half the typical error for
  the lead), so one wild case does not decide the weights.
- A system with little record is given the typical variance until its record outweighs eight forecasts' worth. A new
  system therefore starts as an average member.
- Covariances are shrunk by 30%, and weights are pulled a quarter of the way to equal.
- Weights are constrained to be non-negative.

Evidence fades with a half-life of one year. Two sets of weights are kept, for leads to 48 hours and beyond.

Everything is causal. A forecast is learned from only after its valid time, and against the position known in real
time, never the best track written after the season.

### 2.3 In-storm memory

Within a storm, each system's weight is multiplied by a factor that falls with its error on that storm so far,
relative to the other systems, with a half-life of 36 hours.

### 2.4 Early guidance

A run is not complete when an advisory is written. As in the NHC's interpolated aids, the previous run is read 12
hours further along and slid so that it starts at the observed position. Our runs end at 120 hours, so the last 12
hours are filled by carrying the track on at its final motion.

### 2.5 Uncertainty

- **Cone.** The radius that should hold the centre two times in three is the two-thirds quantile of verified
  consensus errors at each lead. It is scaled by today's spread between the systems, raised to a power learned from
  the same record. Past errors are each divided by their own spread factor before the quantile is taken, so the cone
  stays calibrated when spread varies.
- **Cloud.** Every system's track is given the pooled perturbations of the ensemble members about their own means, so
  a single run is a cloud and not a line. The clouds are mixed with the consensus weights, centred on the consensus,
  and scaled to the errors verified so far.
- **Passage chance.** The weight of the cloud's tracks that pass within 150 km of a place in five days.
- **Scenarios.** The cloud's tracks are grouped by k-medoids. A split is reported only when the silhouette is at
  least 0.45 and each group holds at least 10% of the weight.

## 3. Experiment

| Season | Role | Forecasts | Storms |
|---|---|---|---|
| North Atlantic 2024 | development | 81 | 17 |
| East Pacific 2024 | development | 46 | 14 |
| North Atlantic 2025 | development | 118 | 13 |
| East Pacific 2025 | test | 167 | 18 |

Five learners were run through all seasons in time order: the weights above, the same with in-storm memory,
exponentiated-gradient weights with and without memory, and the weights above with a bias correction learned online.
The learner was chosen by its error relative to an equal-weight mean on the development seasons, with a margin of 2%
that complexity must pay, and ties within 0.5% going to the simpler learner. The weights with in-storm memory were
chosen (ratio 0.92). The East Pacific 2025 season was then scored once.

Truth is the NHC best track. Cases follow the NHC's verification rules. Every comparison is on the forecasts both
sides made, with a 95% interval from a bootstrap over storms.

## 4. Results

### 4.1 Test season: East Pacific 2025

Early guidance, mean error in nautical miles on the forecasts all four made.

| | 12 h | 24 h | 36 h | 48 h | 60 h | 72 h | 96 h | 120 h |
|---|---|---|---|---|---|---|---|---|
| Pluvium Cyclone | 20 | 30 | 37 | 46 | 57 | 69 | 90 | 125 |
| NHC official forecast | 18 | 27 | 34 | 40 | 46 | 52 | 73 | 100 |
| NHC consensus TVCN | 19 | 27 | 35 | 43 | 53 | 64 | 92 | 127 |
| NHC corrected consensus HCCA | 19 | 30 | 34 | 43 | 51 | 58 | 78 | 107 |

- Against the official forecast: higher error at all eight leads, with 95% confidence at four (12, 24, 96 and
  120 hours).
- Against TVCN: lower error at two leads, higher with confidence at one. Level.
- Against Google DeepMind's model (late guidance): 18 against 15 at 12 hours, 38 against 35 at 48, 95 against 72 at
  120. Behind at every lead, with confidence at two.
- Against every single model's own track from the NHC decks (late guidance): lower error at every lead, except
  against GraphCast at 120 hours (84 against 79 on 17 forecasts). With confidence at all eight leads against ECMWF
  HRES and the ECMWF ensemble mean, at five against NOAA GFS and the GEFS mean, at four against the HAFS-A hurricane
  model.
- Against ECMWF AIFS alone, with our tracker: level to 36 hours, then ahead (42 against 44 at 48 hours, 93 against 106
  at 120).
- Against an equal-weight mean of the same six tracks: 98 against 104 at 120 hours, 85 against 91 at 96, level to
  72. The learning is worth about 3%.

### 4.2 Development season: North Atlantic 2025

The same table for the Atlantic reads 20, 30, 39, 49, 64, 79, 94, 125 for Pluvium Cyclone against 21, 34, 45, 56,
70, 85, 106, 170 for the official forecast: lower at every lead, with confidence at 24 and 36 hours. The learner was
chosen with this season in view. It is reported for completeness and is not evidence of beating the official
forecast.

### 4.3 What each ingredient was worth

Error relative to the equal-weight mean, by season (below 1 is better).

| Learner | Atlantic 2024 | East Pacific 2024 | Atlantic 2025 | East Pacific 2025 |
|---|---|---|---|---|
| minimum-variance weights | 0.963 | 0.988 | 0.927 | 0.960 |
| gradient weights | 0.975 | 0.992 | 0.906 | 0.980 |
| minimum-variance weights + storm memory (chosen) | 0.960 | 0.992 | 0.882 | 0.967 |
| gradient weights + storm memory | 0.973 | 0.996 | 0.867 | 0.987 |
| minimum-variance weights + online bias | 0.941 | 1.110 | 0.959 | 0.948 |

Learned weights help by up to 13%, more in the Atlantic than in the East Pacific. Online bias correction is the best
learner in two seasons and the worst in the other two. We read that as bias being mostly a property of a season's
storms and steering, not of the system.

At the end of 2025 ECMWF's AIFS held about a third of the weight, and no system held less than 0.10.

### 4.4 Uncertainty products on the test season

| | 12 h | 24 h | 36 h | 48 h | 60 h | 72 h | 96 h | 120 h |
|---|---|---|---|---|---|---|---|---|
| Centre inside the circle (target 67%) | 70% | 68% | 69% | 69% | 72% | 68% | 72% | 66% |
| The same for a circle that ignores spread | 64% | 65% | 68% | 70% | 78% | 77% | 78% | 69% |
| Mean radius, nautical miles | 21 | 30 | 39 | 46 | 59 | 67 | 95 | 127 |

The circle that follows the systems' spread stays within five points of the target at every lead. The static circle
drifts to 78%. In the East Pacific the spread between systems correlated with the error at 0.4 to 0.8 from 12 to 96
hours. In the Atlantic the correlation was weak at most leads beyond 24 hours and the two circles behaved alike.

Passage chances over 1,945 place forecasts, 101 of which verified: Brier skill 0.465 against the base rate for the
cloud described above, 0.485 for the same cloud before scaling to verified errors, 0.418 for the raw member cloud,
and -0.149 for a yes/no read off the consensus track. Scaling the cloud had helped on both Atlantic seasons and cost
0.02 here. High chances were too high on this season: forecasts of 75% to 100% verified 73% of the time.

The cloud split into scenarios in 5 of 151 forecasts. That is too few to say anything about them.

### 4.5 A storm: Melissa

Hurricane Melissa crossed western Jamaica on 28 October 2025 as a category 5 hurricane. The chance given for the
centre passing within 150 km of Kingston in the next five days was 42% five days ahead, 67% four days ahead and 90%
three days ahead. Track errors for the forecast three days ahead were 36, 45 and 44 nautical miles at 48, 72 and 96
hours. Earlier forecasts gave 13% to 34% for five-day windows in which the centre did not arrive, because the storm
moved more slowly than every system expected.

## 5. Discussion

**What the result is.** A consensus of six public systems, with a tracker that uses only sea-level pressure, is
clearly better than any one of them and as good as a simple consensus of the models the NHC uses. It is not as good
as the NHC's forecasters or as its corrected consensus.

**Why it falls short of the official forecast on the test season.** Three reasons are visible. Early guidance here
comes from runs 12 hours old, where the NHC's aids use 6-hour-old runs when they exist. The official forecast and
HCCA draw on models we do not read, among them regional hurricane models and, in 2025, Google DeepMind's model. And
the East Pacific official forecast is very good: 40 nautical miles at 48 hours.

**The best single addition.** Google DeepMind's track, added as a seventh member with no record, earned a weight of
about 0.3 within the season. With it the consensus matched DeepMind's own track to 60 hours on the test season and
was still behind beyond that.

**What we would not claim.** That the consensus beats the official forecast. One season says yes and one says no,
and the one that says yes took part in the design.

## 6. Limits

- Track only. No intensity, size, wind, surge or rain.
- Two basins, two seasons, 00 and 12 UTC runs.
- Fields at a quarter of a degree. Weak or broad lows are sometimes lost.
- Our GraphCast tracks are worse than the NHC's GraphCast aid (section 2.1).
- The settings were chosen on three seasons and tested on one.

## 7. Reproducing

```bash
pip install pluvium
python -m pluvium.benchmarks.cyclone_bench --decks decks --fields pressure --out cyclone_out
```

`pluvium.cyclone.atcf.fetch_decks` downloads the decks and `pluvium.cyclone.fields.stage_fields` the pressure
fields. The full tables are in [CYCLONE_BENCHMARK.md](../CYCLONE_BENCHMARK.md).

## References

- Goerss, J. S. (2000). Tropical cyclone track forecasts using an ensemble of dynamical models. *Monthly Weather
  Review*, 128, 1187-1193.
- Goerss, J. S. (2007). Prediction of consensus tropical cyclone track forecast error. *Monthly Weather Review*, 135,
  1985-1993.
- Simon, A., Penny, A. B., DeMaria, M., Franklin, J. L., Pasch, R. J., Rappaport, E. N., and Zelinsky, D. A. (2018).
  A description of the real-time HFIP Corrected Consensus Approach (HCCA) for tropical cyclone track and intensity
  guidance. *Weather and Forecasting*, 33, 37-57.
- Lang, S., and co-authors (2024). AIFS: ECMWF's data-driven forecasting system. arXiv:2406.01465.
- Lam, R., and co-authors (2023). Learning skillful medium-range global weather forecasting. *Science*, 382,
  1416-1421.
