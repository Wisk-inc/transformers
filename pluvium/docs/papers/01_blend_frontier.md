# Tail-aware pooling with online weights: a multi-model rain forecast, second generation

Pluvium technical paper 1. Nathan, November 2026.

## Summary

Pluvium Blend turns the forecasts of six operational systems (ECMWF ENS, HRES and AIFS; NOAA GEFS, GFS and GraphCast) into calibrated chances of daily rain for days 1 to 14. A benchmark over a full year found it weakest at 50 mm, in continental winters, and in the structure of its long-range maps. This paper describes the changes aimed at the first two of those and reports what they did on the same test year.

The components are now trained on every heavy-rain row, with weights that keep the chances unbiased, and the pool is fitted on the same rows. The components and the pool are also re-weighed online after every verified day. In 7 places, Brier skill at 50 mm improved on the first Blend in 13 of 28 place and lead-group cells, and the Blend leads every other system, raw or post-processed, in 15 of them (the first led in 14). Against single post-processed systems at 50 mm, 79 of 161 comparisons are better with 95% confidence and 0 are worse. At 1 mm it leads in 24 of 28 cells (the first: 26).

The gains are small. Mean Brier skill at 50 mm on days 1-3 went from +0.1901 to +0.2133, and at 1 mm from +0.3588 to +0.3591. Part of the gain at 50 mm comes from a hand-over for record storms, which is the subject of paper 2; without it the changes described here score +0.1996. On days 11-14 the new Blend leads in fewer places than the first did (at 1 mm in 4 of 7, the first in 6).

The rain test year (April 2025 to March 2026) is the year on which the weak spots were found. Every design choice was made inside the training year and written down before the test year was scored, but a second look at a year is weaker evidence than a fresh year.

## 1. What the first benchmark found

The first Blend post-processed each system with a small network, fitted one more network over all six, and pooled the seven with weights per band of thresholds and lead. Scored over a year in six regions and Jamaica ([FULL_YEAR_BENCHMARK.md](../FULL_YEAR_BENCHMARK.md)), it led every other system on most scores at most leads. It was weakest at 50 mm, where it led in 14 of 28 cells, and in the Great Plains winter, where post-processed ECMWF ENS alone was better.

We traced the weakness at 50 mm to two causes. A uniform sample of 400,000 training rows holds only a few thousand days with 50 mm, so the rarest thresholds were learned from very little. And a pool fitted once on last year's weather keeps last year's opinion of each system all through the next year.

## 2. Method

### 2.1 Components that see every heavy day

A row is one place on one day at one lead. Rows are drawn unevenly on purpose. Every row where the observed rain, or any system's forecast, reaches 25 mm is kept (up to half the budget), and the rest are thinned to fill the budget. A row kept with probability `pi` carries the weight `1 / pi`, so the weighted loss

    L = sum_i  (1 / pi_i) * loss_i

is an unbiased estimate of the loss over all rows (Horvitz and Thompson, 1952). The chances stay calibrated and the tail is learned from several times more events. Rows that were forecast heavy and stayed dry are kept as well, because the networks have to learn false alarms at 50 mm as much as hits.

### 2.2 The pool

Each component `c` gives log-odds `z_ck` for threshold `k`. The pool is a weighted logarithmic pool with an offset (Ranjan and Gneiting, 2010):

    logit p_k = W * ( sum_c w_c * present_c * z_ck ) / ( sum_c w_c * present_c ) + b_k + e_k * lead
    w_c = softplus(a_c + d_c * lead),    W = sum_c w_c

The weights `w` are shared within three bands of thresholds (below 5 mm, 5 to 25 mm, 35 mm and above) and change with lead. A missing component drops out of the average and the rest carry on. The pool is fitted by cross-entropy on out-of-fold rows, so that no component is judged on days it was trained on, with the row weights of section 2.1.

### 2.3 Online weights

The experts are the seven components and the pool. Each lead group (days 1-3, 4-7, 8-10, 11-14) and band of thresholds keeps its own weights. After every verified day, with `L_e` expert `e`'s mean Brier score on that day's rows, `L_mix` the mixture's and `S` a running mean of `L_mix`:

    r_e = clip( (L_mix - L_e) / S, -2, 2 )
    log w_e  <-  log w_e + eta * r_e - (eta * r_e)^2 / 2
    w  <-  (1 - alpha) * w / sum(w) + alpha * prior

The second-order term is that of Bernstein online aggregation (Wintenberger, 2017), which Berrisch and Ziel (2021) applied to probabilistic forecasts. Dividing by `S` lets one learning rate serve 1 mm and 50 mm. The last line is a fixed share (Herbster and Warmuth, 1998): a little weight returns to the prior at every step, so an expert that was written off can come back. The prior gives half the weight to the pool. The settings are `eta = 1` and `alpha = 0.02`.

The weights do not start cold. Before the test they are run through the training period's out-of-fold forecasts in time order. In the benchmark a day's rows are then learned from 48 hours after the day ends. For the truth used here that delay is much too short; the record is published about four and a half months late.

### 2.4 Record storms

A forecast beyond anything in the training period is handed back, in part, to what the raw systems say, in the places where that improved the forecast of storms withheld from training. That is the subject of [paper 2](02_record_storms.md).

### 2.5 How the design was chosen

Every choice was made inside the training year, by cross-validation or by forecasting its later months from its earlier ones, in four regions. Most of what we tried was dropped. The numbers are in [WHAT_DID_NOT_WORK.md](../WHAT_DID_NOT_WORK.md).

| Tried | Outcome |
|---|---|
| Weights for each of the 16 thresholds, tied by a smoothness penalty | No gain over three bands |
| Weights that follow the season | Worse: one year cannot tell a season from that year's weather |
| Weights that follow the regime (how wet and how split the components are) | Better at 50 mm, worse at 1 mm |
| A learning rate of 2 in place of 1 | No better |
| A separate forecast of the expected amount | Gains under 1% in RMSE |

The design was then written down with its settings, and the test year was scored.

## 3. Test

There are 7 places: Jamaica at 8 km (with Pluvium's network), Caribbean (tropical islands and coasts), Maritime Continent (tropical, deep convection), Alps (mid-latitude mountains), US Pacific Northwest (mid-latitude coast and mountains), US Great Plains (mid-latitude storms), Southern Andes and Chile (southern mid-latitudes). Training used runs from April 2024 to March 2025. The test is April 2025 to March 2026, with everything refitted every quarter on all data before it. Runs every second day are scored for days 1 to 3 and every fourth day to day 14. The truth is the CMORPH daily satellite estimate (Joyce et al., 2004) on land. Skill is against the training year's climatology by place and month. Confidence intervals come from a block bootstrap over runs. In the scoreboard a system counts as beaten only if the Blend is ahead of it both raw and post-processed; Pluvium's own variants are not counted as rivals.

## 4. Results

### 4.1 Scoreboard

In how many of the 7 places the new Blend beat every other system. In brackets: the first Blend.

| Score | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Brier skill, 1 mm | 7 of 7 (was 7) | 7 of 7 (was 7) | 6 of 7 (was 6) | 4 of 7 (was 6) |
| Brier skill, 10 mm | 7 of 7 (was 7) | 7 of 7 (was 7) | 7 of 7 (was 7) | 4 of 7 (was 3) |
| Brier skill, 25 mm | 6 of 7 (was 6) | 6 of 7 (was 6) | 7 of 7 (was 7) | 4 of 7 (was 4) |
| Brier skill, 50 mm | 6 of 7 (was 3) | 4 of 7 (was 3) | 3 of 7 (was 3) | 2 of 7 (was 5) |
| CSI, 1 mm | 7 of 7 (was 7) | 6 of 7 (was 6) | 6 of 7 (was 6) | 2 of 7 (was 2) |
| CSI, 10 mm | 6 of 7 (was 6) | 6 of 7 (was 6) | 5 of 7 (was 7) | 2 of 7 (was 3) |
| ROC area, 1 mm | 7 of 7 (was 7) | 7 of 7 (was 7) | 6 of 7 (was 6) | 4 of 7 (was 4) |
| CRPS | 7 of 7 (was 6) | 7 of 7 (was 7) | 7 of 7 (was 7) | 4 of 7 (was 3) |
| RMSE | 6 of 7 (was 4) | 5 of 7 (was 6) | 6 of 7 (was 6) | 4 of 7 (was 4) |
| ACC | 7 of 7 (was 6) | 5 of 7 (was 5) | 3 of 7 (was 3) | 3 of 7 (was 2) |

### 4.2 Heavy rain

Brier skill score for 50 mm or more: first Blend -> new Blend (best other system). Bold where the new Blend is ahead of every other system. A trailing + where it improved on the first.

| Place | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Jamaica at 8 km (with Pluvium's network) | +0.439 -> **+0.515** (+0.511) + | +0.233 -> **+0.247** (+0.242) + | +0.066 -> +0.065 (+0.117) | +0.004 -> -0.003 (+0.355) |
| Caribbean (tropical islands and coasts) | +0.079 -> **+0.087** (+0.083) + | +0.056 -> +0.056 (+0.057) | +0.029 -> **+0.029** (+0.028) | +0.019 -> **+0.018** (+0.011) |
| Maritime Continent (tropical, deep convection) | +0.081 -> **+0.082** (+0.076) + | +0.050 -> **+0.050** (+0.046) + | +0.031 -> **+0.031** (+0.029) | +0.021 -> **+0.021** (+0.018) |
| Alps (mid-latitude mountains) | +0.087 -> +0.091 (+0.097) + | +0.027 -> +0.032 (+0.038) + | +0.020 -> +0.020 (+0.021) | +0.004 -> +0.001 (+0.002) |
| US Pacific Northwest (mid-latitude coast and mountains) | +0.288 -> **+0.307** (+0.288) + | +0.179 -> **+0.168** (+0.160) | +0.099 -> +0.097 (+0.114) | +0.016 -> +0.008 (+0.009) |
| US Great Plains (mid-latitude storms) | +0.105 -> **+0.129** (+0.114) + | +0.032 -> +0.029 (+0.035) | +0.017 -> +0.015 (+0.017) | -0.002 -> +0.003 (+0.004) + |
| Southern Andes and Chile (southern mid-latitudes) | +0.250 -> **+0.283** (+0.228) + | +0.115 -> **+0.133** (+0.107) + | +0.034 -> **+0.032** (+0.029) | +0.005 -> +0.006 (+0.006) + |

In the places scored after the design was final (Jamaica and the US Great Plains), skill at 50 mm improved on the first Blend in 4 of 8 cells and the new Blend leads every other system in 3 (the first: 0).

### 4.3 What each part added

The mean over the places of each score, for the pool alone and with each further part.

Brier skill, 1 mm (mean over 7 places).

| Design | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Pool alone | +0.3583 | +0.2491 | +0.1231 | +0.0396 |
| Pool + online weights | +0.3591 | +0.2494 | +0.1226 | +0.0390 |
| Pool + hand-over, always on | +0.3583 | +0.2491 | +0.1230 | +0.0397 |
| Pool + online weights + hand-over, always on (as first declared) | +0.3591 | +0.2494 | +0.1226 | +0.0390 |
| Pool + online weights + hand-over where the staged storms say so (Pluvium Blend) | +0.3591 | +0.2494 | +0.1226 | +0.0390 |

Brier skill, 50 mm (mean over 7 places).

| Design | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Pool alone | +0.1921 | +0.0990 | +0.0416 | +0.0091 |
| Pool + online weights | +0.1996 | +0.1021 | +0.0411 | +0.0076 |
| Pool + hand-over, always on | +0.2008 | +0.0958 | +0.0398 | +0.0091 |
| Pool + online weights + hand-over, always on (as first declared) | +0.2089 | +0.0997 | +0.0394 | +0.0077 |
| Pool + online weights + hand-over where the staged storms say so (Pluvium Blend) | +0.2133 | +0.1022 | +0.0412 | +0.0077 |

CRPS (mean over 7 places).

| Design | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Pool alone | 1.8593 | 2.1063 | 2.3292 | 2.4778 |
| Pool + online weights | 1.8548 | 2.1055 | 2.3306 | 2.4780 |
| Pool + hand-over, always on | 1.8387 | 2.1030 | 2.3297 | 2.4776 |
| Pool + online weights + hand-over, always on (as first declared) | 1.8359 | 2.1022 | 2.3312 | 2.4778 |
| Pool + online weights + hand-over where the staged storms say so (Pluvium Blend) | 1.8352 | 2.1020 | 2.3307 | 2.4778 |

### 4.4 Where the weight went

The online weights for light rain on days 1-3 in the US Great Plains, at the end of each quarter of the test year. The prior gives the pool 0.50 and each component 0.07.

| After | ECMWF ENS | NOAA GEFS | ECMWF HRES | ECMWF AIFS | NOAA GraphCast | NOAA GFS | one network | the pool |
|---|---|---|---|---|---|---|---|---|
| quarter 1 | 0.01 | 0.01 | 0.01 | 0.04 | 0.02 | 0.01 | 0.14 | 0.77 |
| quarter 2 | 0.01 | 0.01 | 0.01 | 0.04 | 0.02 | 0.01 | 0.16 | 0.73 |
| quarter 3 | 0.04 | 0.03 | 0.02 | 0.02 | 0.03 | 0.03 | 0.12 | 0.72 |
| quarter 4 | 0.05 | 0.01 | 0.02 | 0.04 | 0.01 | 0.01 | 0.10 | 0.77 |

### 4.5 Other scores

CRPS leads in 25 of 28 cells (the first: 23), RMSE in 21 of 28 (20) and the anomaly correlation in 18 of 28 (16). At 1 mm on days 1-7, 81 of 84 comparisons with single post-processed systems are better with 95% confidence and 0 are worse.

### 4.6 Winter against a second truth

CMORPH is weak on frozen precipitation, so a winter loss could be the forecast's or the truth's. The US places were scored again for December to February against MRMS, the US radar and gauge analysis (Zhang et al., 2016), on the same rows. Everything was trained on CMORPH, so every system carries the same handicap against MRMS, and the question is whether the order between the systems changes.

US Pacific Northwest, DJF: 90 days. Each cell: Brier skill against CMORPH / against MRMS, each against its own base rate. Bold is the best in the column for that truth.

Rain of 1 mm or more.

| System | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Pluvium Blend | +0.407 / +0.569 | **+0.314** / +0.418 | +0.160 / **+0.194** | +0.115 / +0.076 |
| ECMWF ENS, post-processed | **+0.407** / +0.523 | +0.289 / +0.372 | **+0.171** / +0.193 | **+0.124** / **+0.088** |
| NOAA GEFS, post-processed | +0.383 / +0.508 | +0.245 / +0.314 | +0.117 / +0.121 | +0.094 / +0.044 |
| ECMWF HRES, post-processed | +0.391 / +0.547 | +0.287 / +0.341 | +0.118 / +0.124 | +0.086 / +0.024 |
| ECMWF AIFS, post-processed | +0.392 / **+0.572** | +0.306 / +0.393 | +0.128 / +0.155 | +0.103 / +0.063 |
| NOAA GraphCast, post-processed | +0.345 / +0.533 | +0.248 / +0.317 | +0.116 / +0.133 | - |
| NOAA GFS, post-processed | +0.369 / +0.511 | +0.242 / +0.290 | +0.065 / +0.052 | +0.058 / +0.007 |
| One network over the six | +0.403 / +0.569 | +0.308 / **+0.429** | +0.143 / +0.194 | +0.105 / +0.079 |
| Share of rows with the rain: CMORPH / MRMS | 20.4% / 27.1% | 20.7% / 27.6% | 20.7% / 27.1% | 20.7% / 27.6% |
| Rows where the two truths agree | 84% | 84% | 84% | 84% |

Rain of 10 mm or more.

| System | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Pluvium Blend | +0.424 / +0.520 | **+0.284** / **+0.349** | +0.136 / **+0.186** | +0.071 / +0.062 |
| ECMWF ENS, post-processed | +0.421 / +0.504 | +0.263 / +0.325 | **+0.138** / +0.173 | **+0.078** / **+0.082** |
| NOAA GEFS, post-processed | +0.395 / +0.463 | +0.205 / +0.262 | +0.079 / +0.120 | +0.072 / +0.057 |
| ECMWF HRES, post-processed | +0.406 / +0.518 | +0.252 / +0.309 | +0.088 / +0.112 | +0.063 / +0.045 |
| ECMWF AIFS, post-processed | +0.417 / **+0.526** | +0.277 / +0.336 | +0.108 / +0.165 | +0.060 / +0.044 |
| NOAA GraphCast, post-processed | +0.330 / +0.517 | +0.203 / +0.285 | +0.069 / +0.123 | - |
| NOAA GFS, post-processed | +0.391 / +0.469 | +0.188 / +0.234 | +0.029 / +0.066 | +0.033 / +0.015 |
| One network over the six | **+0.425** / +0.512 | +0.283 / +0.345 | +0.123 / +0.182 | +0.035 / +0.035 |
| Share of rows with the rain: CMORPH / MRMS | 7.5% / 8.8% | 7.7% / 9.2% | 7.8% / 9.2% | 7.7% / 9.2% |
| Rows where the two truths agree | 93% | 93% | 93% | 93% |

US Great Plains, DJF: 90 days. Each cell: Brier skill against CMORPH / against MRMS, each against its own base rate. Bold is the best in the column for that truth.

Rain of 1 mm or more.

| System | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Pluvium Blend | +0.291 / +0.493 | +0.121 / +0.254 | +0.090 / +0.123 | **-0.072** / -0.017 |
| ECMWF ENS, post-processed | **+0.311** / +0.436 | +0.097 / +0.253 | +0.035 / +0.103 | -0.077 / **-0.015** |
| NOAA GEFS, post-processed | +0.266 / +0.440 | +0.098 / +0.202 | +0.048 / +0.093 | -0.083 / -0.044 |
| ECMWF HRES, post-processed | +0.290 / +0.459 | +0.063 / +0.189 | -0.056 / -0.011 | -0.108 / -0.052 |
| ECMWF AIFS, post-processed | +0.260 / **+0.511** | +0.048 / +0.182 | +0.071 / +0.070 | -0.103 / -0.046 |
| NOAA GraphCast, post-processed | +0.231 / +0.461 | **+0.128** / +0.209 | +0.032 / +0.042 | - |
| NOAA GFS, post-processed | +0.250 / +0.431 | +0.102 / +0.181 | +0.004 / +0.050 | -0.114 / -0.057 |
| One network over the six | +0.284 / +0.504 | +0.109 / **+0.266** | **+0.098** / **+0.137** | -0.095 / -0.025 |
| Share of rows with the rain: CMORPH / MRMS | 5.3% / 7.5% | 5.2% / 7.5% | 6.2% / 8.6% | 5.2% / 7.5% |
| Rows where the two truths agree | 94% | 94% | 93% | 94% |

Rain of 10 mm or more.

| System | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Pluvium Blend | +0.490 / +0.536 | +0.228 / +0.280 | +0.065 / +0.052 | **-0.111** / **-0.067** |
| ECMWF ENS, post-processed | +0.483 / +0.496 | +0.170 / +0.241 | -0.010 / +0.006 | -0.112 / -0.068 |
| NOAA GEFS, post-processed | +0.417 / +0.459 | +0.248 / +0.252 | -0.008 / -0.004 | -0.139 / -0.095 |
| ECMWF HRES, post-processed | +0.452 / +0.468 | +0.132 / +0.181 | -0.100 / -0.060 | -0.167 / -0.105 |
| ECMWF AIFS, post-processed | +0.431 / +0.492 | +0.076 / +0.145 | +0.060 / +0.038 | -0.146 / -0.094 |
| NOAA GraphCast, post-processed | +0.442 / +0.515 | **+0.270** / **+0.314** | +0.047 / +0.035 | - |
| NOAA GFS, post-processed | +0.411 / +0.453 | +0.174 / +0.205 | -0.018 / +0.005 | -0.161 / -0.104 |
| One network over the six | **+0.502** / **+0.555** | +0.244 / +0.306 | **+0.100** / **+0.077** | -0.135 / -0.080 |
| Share of rows with the rain: CMORPH / MRMS | 1.3% / 1.8% | 1.3% / 1.8% | 1.7% / 2.3% | 1.3% / 1.8% |
| Rows where the two truths agree | 99% | 99% | 99% | 99% |

## 5. Limits

- The test year is the one on which the weak spots were found (see the summary).
- The gains are small in absolute terms, and at long leads every system is close to the climatology.
- The online weights were tested with rain known after two days. The record arrives about 140 days late. The quarterly refits also used runs more recent than a live system could train on.
- There is one truth, a satellite estimate, which underestimates orographic and frozen precipitation and the heaviest rain.
- The scores are for daily totals on a quarter-degree grid (8 km for Jamaica). Nothing here speaks to hourly rain.
- The anomaly correlation of the expected amount at long range is still behind raw models in mid-latitudes.
- The full list is in [FRONTIER_BENCHMARK.md](../FRONTIER_BENCHMARK.md#what-is-still-weak).

## 6. Reproducing

```bash
pip install pluvium
python -m pluvium.benchmarks.frontier_bench --archive ARCHIVE --truth TRUTH --out OUT
python scripts/doc_frontier.py OUT/results FIRST EXPERIMENTS docs/FRONTIER_BENCHMARK.md
```

`pluvium.sources.archive.stage_archive` downloads the runs and `pluvium.sources.cmorph_daily.stage_daily` the truth. The delay of the truth is the `latency` argument of `pluvium.frontier.evaluate.evaluate`, in hours. To train the same design for a country: `pluvium blend train` ([TRAINING.md](../TRAINING.md)).

## References

- Berrisch, J., and Ziel, F. (2021). CRPS learning. arXiv:2102.00968.
- Herbster, M., and Warmuth, M. K. (1998). Tracking the best expert. *Machine Learning*, 32, 151-178. doi:10.1023/A:1007424614876
- Horvitz, D. G., and Thompson, D. J. (1952). A generalization of sampling without replacement from a finite universe. *Journal of the American Statistical Association*, 47, 663-685. doi:10.1080/01621459.1952.10483446
- Joyce, R. J., Janowiak, J. E., Arkin, P. A., and Xie, P. (2004). CMORPH: a method that produces global precipitation estimates from passive microwave and infrared data at high spatial and temporal resolution. *Journal of Hydrometeorology*, 5, 487-503. doi:10.1175/1525-7541(2004)005<0487:CAMTPG>2.0.CO;2
- Ranjan, R., and Gneiting, T. (2010). Combining probability forecasts. *Journal of the Royal Statistical Society B*, 72, 71-91. doi:10.1111/j.1467-9868.2009.00726.x
- Wintenberger, O. (2017). Optimal learning with Bernstein online aggregation. *Machine Learning*, 106, 119-141. doi:10.1007/s10994-016-5592-6
- Zhang, J., Howard, K., Langston, C., Kaney, B., and co-authors (2016). Multi-Radar Multi-Sensor (MRMS) quantitative precipitation estimation: initial operating capabilities. *Bulletin of the American Meteorological Society*, 97, 621-638. doi:10.1175/BAMS-D-14-00174.1
