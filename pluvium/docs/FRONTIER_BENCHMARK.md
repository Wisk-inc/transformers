# Pluvium Blend on the frontier: the second full-year benchmark

The first full-year benchmark ([FULL_YEAR_BENCHMARK.md](FULL_YEAR_BENCHMARK.md)) found three weak spots in Pluvium Blend: the heavy tail (50 mm), winter in mid-latitude continental climates, and structure at long range. This page scores a redesign aimed at them, on the same data, the same test year and the same rules.

## Read this first

The test year is the same as before (April 2025 to March 2026), and the weak spots were found on it. This is a second look at a year already seen. Every design choice was made inside the training year (April 2024 to March 2025), by cross-validation or by forecasting its later months from its earlier ones, and the design was written down before the test year was scored. The parts are shown separately below, including where they cost something.

One thing was changed after scoring began. The design as written down handed every record storm to the raw systems. The first places scored showed that this helps in some places and does harm in others. Each place now decides from its own training year: the hand-over is used only where it improved the forecast of storms withheld from training. A first remedy, an audit on out-of-fold forecasts, was tried and dropped. Its verdict went against the test year in 3 of 7 places, and its sign changes with which calibrated forecast it is computed on. The rule now used was fixed when 5 of the 7 places had been scored (the Caribbean, the Maritime Continent, the Alps, the US Pacific Northwest and the Southern Andes). Only Jamaica and the US Great Plains were scored after it. The design as written down, the final one and the design with no hand-over are scored everywhere and shown side by side.

One assumption of the test was too kind. The online weights were given each day's rain two days after it fell. The satellite record is published about four and a half months late. The truth is CMORPH, a satellite estimate, which is weak on frozen precipitation and on the heaviest rain.

## What changed

1. Each post-processing network is trained on all rows with observed or forecast rain of 25 mm or more. The other rows are thinned, with weights that undo the thinning.
2. The pool is fitted on the same kind of rows.
3. The components and the pool are re-weighed online after every verified day, with a 48-hour delay for the truth to arrive and one learning rate, chosen on the training year.
4. A forecast outside anything in the training period is handed back, in part, to what the raw systems say.
5. Each place decides about that hand-over. The four wettest days of the training year are withheld from training and then forecast, and the hand-over is used in a place only if it lowered the Brier score at 50 mm on those storms.
6. Scenarios arrange the calibrated amounts in space like the ECMWF ensemble's members, so each scenario is as varied as real weather.

## The short version

- At 50 mm, Brier skill improved on the first Blend in 13 of 28 place and lead-group cells, and in 7 of 7 on days 1-3. Mean skill over the places on days 1-3 went from +0.1901 to +0.2133. The new Blend leads every other system at 50 mm in 15 of 28 cells (the first led in 14).
- In the places scored after the design was final (Jamaica and the US Great Plains), 50 mm skill improved on the first Blend in 4 of 8 cells, and the new Blend leads every other system in 3 (the first: 0).
- Against each post-processed system at 50 mm, 79 of 161 comparisons are better with 95% confidence and 0 are worse.
- At 1 mm, which is rain or no rain, the new Blend leads in 24 of 28 cells (the first: 26), and improved on the first in 13. On days 1-7, 81 of 84 comparisons with single post-processed systems are better with 95% confidence and 0 are worse.
- The CRPS leads in 25 of 28 cells (was 23), the RMSE in 21 of 28 (was 20) and the anomaly correlation in 18 of 28 (was 16).
- Days 11-14 got slightly worse. There the new Blend leads every other system at 1 mm in 4 of 7 places (the first: 6) and at 50 mm in 2 (the first: 5). Mean Brier skill at 1 mm on those days went from +0.0394 to +0.0390. The pool alone scores +0.0396: the online weights for that lead group learn from few forecasts that verify late, and cost a little.
- Each place's staged storms kept the record-storm hand-over in Jamaica, the Caribbean, the Maritime Continent, the US Great Plains and the Southern Andes and switched it off in the Alps and the US Pacific Northwest. That verdict had the sign of what the hand-over then did in the test year in 7 of 7 places (2 of the 2 scored after the rule was final). The out-of-fold audit tried first would have been right in 4 of 7.
- On days 11-14 one coupled scenario has 0.88 of the weather's spatial variability (raw ECMWF ENS: 0.84), and the CRPS of rain averaged over two-degree boxes is 2.26 mm against 2.93 for uncoupled draws and 2.57 for the raw ensemble (means over 7 places).
- The weak spots that remain are listed [near the end](#what-is-still-weak).

## Scoreboard

In how many of the 7 places the new Blend beat every other system, raw and post-processed, on each score. In brackets: the first Blend's count. Pluvium's own variants are not counted as rivals.

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

## Weak spot 1: the heavy tail

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

Skill in the tail against the raw consensus of the six systems (threshold-weighted CRPS above 25 mm; 1 is perfect, 0 no better than the raw consensus).

| Place | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Jamaica | -0.003 | -0.032 | -0.019 | -0.222 |
| Caribbean | +0.064 | +0.053 | +0.053 | +0.047 |
| Maritime Continent | +0.065 | +0.059 | +0.050 | +0.046 |
| Alps | +0.125 | +0.081 | +0.118 | +0.065 |
| US Pacific Northwest | +0.206 | +0.141 | +0.161 | +0.108 |
| US Great Plains | +0.030 | +0.048 | +0.060 | +0.067 |
| Southern Andes | +0.352 | +0.233 | +0.192 | +0.166 |

What a warning for 50 mm meant on days 1-3 (issued: share of all place-days; came: how often 50 mm or more then fell; caught: share of all 50 mm cases covered).

| Place | 50 mm days | Watch (10%+): issued / came / caught | Warning (30%+) | Extreme (60%+) |
|---|---|---|---|---|
| Jamaica | 0.83% | 1.18% / 44% / 63% | 0.57% / 81% / 55% | 0.40% / 98% / 47% |
| Caribbean | 0.77% | 1.24% / 19% / 30% | 0.20% / 39% / 10% | 0.04% / 69% / 3% |
| Maritime Continent | 3.07% | 5.55% / 18% / 33% | 0.57% / 42% / 8% | 0.07% / 77% / 2% |
| Alps | 0.32% | 0.58% / 18% / 33% | 0.10% / 31% / 10% | 0.00% / 18% / 0% |
| US Pacific Northwest | 0.27% | 0.62% / 31% / 71% | 0.23% / 53% / 46% | 0.06% / 67% / 15% |
| US Great Plains | 0.49% | 0.93% / 19% / 37% | 0.19% / 39% / 16% | 0.02% / 89% / 4% |
| Southern Andes | 0.51% | 1.23% / 30% / 72% | 0.46% / 49% / 44% | 0.13% / 66% / 17% |

### The hand-over: staged, checked, and scored three ways

Storms withheld from training (training year only). The wettest days were taken out of training, then forecast. The calibrated forecast shrinks them. The hand-over gives some of that back where the raw systems are the better guide, and does harm where they are not. Ordinary days are left alone. A place uses the hand-over only where the Brier score in the fifth column fell (bold).

| Place | Withheld days | Observed mean, mm | Calibrated amount, mm | Brier at 50 mm: calibrated -> with hand-over | Raw systems alone | Ordinary days handed over |
|---|---|---|---|---|---|---|
| Jamaica | 2024-07-03, 2024-11-26, 2024-11-05, 2024-09-23 | 48.1 | 15.5 | 0.3419 -> **0.3202** | 0.3197 | 0.00% |
| Caribbean | 2024-11-15, 2024-06-23, 2024-10-19, 2024-06-14 | 11.8 | 7.3 | 0.0468 -> **0.0460** | 0.0516 | 0.01% |
| Maritime Continent | 2025-03-19, 2024-11-26, 2025-01-26, 2025-01-10 | 21.6 | 14.0 | 0.0965 -> **0.0962** | 0.1129 | 0.08% |
| Alps | 2024-10-08, 2024-09-08, 2024-10-03, 2024-04-01 | 20.3 | 10.4 | 0.0770 -> 0.0781 | 0.0845 | 0.07% |
| US Pacific Northwest | 2024-11-13, 2024-11-22, 2025-01-31, 2024-05-04 | 18.9 | 7.3 | 0.0835 -> 0.0835 | 0.0953 | 0.01% |
| US Great Plains | 2024-11-18, 2024-11-03, 2025-04-05, 2024-04-28 | 20.7 | 11.9 | 0.1083 -> **0.1036** | 0.1017 | 0.02% |
| Southern Andes | 2024-08-01, 2025-03-29, 2024-09-01, 2024-04-02 | 15.1 | 4.3 | 0.0827 -> **0.0782** | 0.0666 | 0.14% |

Two rules from training data, and what happened. The staged storms are the rule that decides. The yearly check was tried first as the switch: the hand-over is applied to all the training period's out-of-fold forecasts and the Brier score at 50 mm compared. It touches one row in a thousand, so its effect on a year's score is a few tenths of a percent, and the sign of so small a number tells us nothing. The last column is the test year: Brier skill at 50 mm on days 1-3 with the hand-over always on, minus without it. A verdict is bold where it had the sign of the test year.

| Place | Staged storms: Brier at 50 mm on the withheld storms | Their verdict | Yearly check: Brier at 50 mm over the training period | It would have said | Test year: what the hand-over did |
|---|---|---|---|---|---|
| Jamaica | -6.4% | **keep** | -0.20% | **keep** (at 4 of 4 refits: keep) | +0.0566 |
| Caribbean | -1.8% | **keep** | +0.41% | switch off (at 0 of 4 refits: keep) | +0.0048 |
| Maritime Continent | -0.4% | **keep** | +0.00% | switch off (at 2 of 4 refits: keep) | +0.0015 |
| Alps | +1.5% | **switch off** | +0.20% | **switch off** (at 0 of 4 refits: keep) | -0.0110 |
| US Pacific Northwest | +0.0% | **switch off** | +0.21% | **switch off** (at 0 of 4 refits: keep) | -0.0191 |
| US Great Plains | -4.3% | **keep** | -0.17% | **keep** (at 4 of 4 refits: keep) | +0.0152 |
| Southern Andes | -5.4% | **keep** | +0.07% | switch off (at 0 of 4 refits: keep) | +0.0175 |

The staged storms had the sign of the test year in 7 of 7 places, the yearly check in 4 of 7. The staged rule was the first one written down, was replaced by the yearly check, and was restored when 5 of the 7 places had been scored, so its count is partly hindsight.

Brier skill at 50 mm in the test year, days 1-3, three ways. Bold is the best of the three.

| Place | No hand-over | Always on (as first declared) | As the staged storms decided (Pluvium Blend) | Their verdict |
|---|---|---|---|---|
| Jamaica | +0.4580 | **+0.5147** | **+0.5147** | keep |
| Caribbean | +0.0819 | **+0.0867** | **+0.0867** | keep |
| Maritime Continent | +0.0804 | **+0.0818** | **+0.0818** | keep |
| Alps | **+0.0908** | +0.0798 | **+0.0908** | switch off |
| US Pacific Northwest | **+0.3073** | +0.2882 | **+0.3073** | switch off |
| US Great Plains | +0.1135 | **+0.1287** | **+0.1287** | keep |
| Southern Andes | +0.2652 | **+0.2827** | **+0.2827** | keep |
| **Mean** | +0.1996 | +0.2089 | +0.2133 | |

Brier skill at 50 mm in the test year, days 4-7, three ways. Bold is the best of the three.

| Place | No hand-over | Always on (as first declared) | As the staged storms decided (Pluvium Blend) | Their verdict |
|---|---|---|---|---|
| Jamaica | +0.2427 | **+0.2469** | **+0.2469** | keep |
| Caribbean | **+0.0568** | +0.0559 | +0.0559 | keep |
| Maritime Continent | +0.0492 | **+0.0503** | **+0.0503** | keep |
| Alps | **+0.0325** | +0.0248 | **+0.0325** | switch off |
| US Pacific Northwest | **+0.1682** | +0.1589 | **+0.1682** | switch off |
| US Great Plains | **+0.0329** | +0.0287 | +0.0287 | keep |
| Southern Andes | +0.1323 | **+0.1326** | **+0.1326** | keep |
| **Mean** | +0.1021 | +0.0997 | +0.1022 | |

## Weak spot 2: winter in mid-latitude climates

Brier skill score for 1 mm or more in each place's winter: first Blend -> new Blend (best other system and which). Bold where the new Blend is ahead.

| Place (its winter) | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Alps (DJF) | +0.390 -> +0.388 (+0.389, ECMWF ENS) | +0.253 -> **+0.255** (+0.239, ECMWF AIFS) | +0.060 -> +0.069 (+0.090, ECMWF AIFS) | -0.001 -> +0.005 (+0.007, NOAA GEFS) |
| US Pacific Northwest (DJF) | +0.345 -> +0.347 (+0.347, ECMWF ENS) | +0.244 -> **+0.245** (+0.236, ECMWF AIFS) | +0.078 -> +0.073 (+0.086, ECMWF ENS) | +0.034 -> +0.027 (+0.037, ECMWF ENS) |
| US Great Plains (DJF) | +0.330 -> +0.337 (+0.356, ECMWF ENS) | +0.183 -> +0.181 (+0.187, NOAA GraphCast) | +0.130 -> **+0.129** (+0.110, ECMWF AIFS) | -0.005 -> **+0.001** (-0.004, ECMWF ENS) |
| Southern Andes (JJA) | +0.410 -> **+0.406** (+0.401, ECMWF ENS) | +0.296 -> **+0.295** (+0.284, ECMWF ENS) | +0.145 -> **+0.144** (+0.136, ECMWF ENS) | +0.048 -> +0.047 (+0.050, ECMWF ENS) |

### Winter against a second truth

CMORPH is weak on frozen precipitation, so a winter loss could be the forecast's or the truth's. The two US places were scored again for December to February against MRMS, the US radar and gauge analysis, on the same rows. Everything was trained on CMORPH, so every system carries the same handicap against MRMS, and the question is whether the order between the systems changes.

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


## Weak spot 3: structure at long range

The map of expected amounts is smooth at long range, and should be: nobody knows where the rain band will lie in twelve days. Its sharpness in the scoreboard's sense has not changed and will not. At that range a user is better served by a set of scenarios that each look like weather, and those are scored here.

Each cell: raw ECMWF ENS / Pluvium coupled / Pluvium uncoupled.

Sharpness of one scenario (1 = as varied as the weather).

| Place | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Jamaica | **1.18** / 1.42 / 1.66 | **0.91** / 1.18 / 1.49 | 0.42 / 0.61 / **0.84** | 0.61 / 0.69 / **1.00** |
| Caribbean | 0.82 / 0.93 / **0.93** | 0.85 / 0.90 / **0.91** | 0.97 / 0.98 / **1.00** | 0.92 / 0.95 / **0.97** |
| Maritime Continent | 0.49 / 0.82 / **0.84** | 0.50 / 0.85 / **0.87** | 0.48 / 0.84 / **0.87** | 0.51 / 0.90 / **0.94** |
| Alps | 0.98 / 0.98 / **0.99** | **1.00** / 1.03 / 1.08 | 0.95 / **1.00** / 1.11 | 0.92 / **0.96** / 1.08 |
| US Pacific Northwest | **1.03** / 1.10 / 1.11 | 0.97 / 0.96 / **1.00** | 0.84 / 0.79 / **0.84** | 0.96 / 0.95 / **1.04** |
| US Great Plains | 0.84 / 0.97 / **0.97** | 0.90 / **0.99** / 1.02 | **1.07** / 1.13 / 1.21 | 0.94 / **1.01** / 1.10 |
| Southern Andes | 1.12 / 0.93 / **0.94** | 1.37 / **1.02** / 1.04 | **1.12** / 0.76 / 0.78 | **1.06** / 0.72 / 0.75 |

CRPS place by place, mm (lower is better).

| Place | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Jamaica | 2.62 / **1.71** / 1.71 | 2.50 / **1.79** / 1.79 | 2.46 / **1.78** / 1.78 | 2.85 / **2.40** / 2.40 |
| Caribbean | 3.18 / **2.08** / 2.08 | 3.14 / **2.15** / 2.15 | 3.16 / **2.18** / 2.18 | 3.11 / **2.30** / 2.30 |
| Maritime Continent | 6.12 / **5.23** / 5.23 | 6.05 / **5.23** / 5.23 | 5.99 / **5.33** / 5.33 | 6.06 / **5.47** / 5.47 |
| Alps | 1.65 / **1.07** / 1.07 | 1.75 / **1.42** / 1.42 | 2.07 / **1.87** / 1.87 | 2.33 / **2.10** / 2.10 |
| US Pacific Northwest | 1.46 / **0.96** / 0.96 | 1.13 / **0.91** / 0.91 | 1.68 / **1.38** / 1.38 | 1.88 / **1.71** / 1.71 |
| US Great Plains | 1.55 / **1.38** / 1.38 | 1.73 / **1.62** / 1.62 | 1.30 / **1.24** / 1.24 | 1.76 / **1.71** / 1.71 |
| Southern Andes | 1.90 / **0.85** / 0.85 | 1.60 / **0.94** / 0.94 | 2.04 / **1.71** / 1.71 | 1.96 / **1.59** / 1.59 |

CRPS of the mean rain over two-degree boxes, mm (lower is better).

| Place | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Jamaica | 2.26 / **1.21** / 1.37 | 2.06 / **1.17** / 1.55 | 1.98 / **1.40** / 1.87 | 2.77 / **2.37** / 3.27 |
| Caribbean | 2.43 / **1.52** / 1.71 | 2.40 / **1.59** / 1.81 | 2.53 / **1.73** / 2.02 | 2.54 / **1.89** / 2.25 |
| Maritime Continent | 4.56 / **3.88** / 4.43 | 4.67 / **4.00** / 4.64 | 4.76 / **4.24** / 4.96 | 4.84 / **4.44** / 5.36 |
| Alps | 1.26 / **0.83** / 1.01 | 1.48 / **1.24** / 1.60 | 1.89 / **1.79** / 2.43 | 2.14 / **1.97** / 2.80 |
| US Pacific Northwest | 1.28 / **0.83** / 0.98 | 1.12 / **0.90** / 1.13 | 1.69 / **1.41** / 1.85 | 1.92 / **1.79** / 2.25 |
| US Great Plains | 1.37 / **1.18** / 1.38 | 1.52 / **1.41** / 1.77 | 1.25 / **1.19** / 1.67 | 1.63 / **1.59** / 2.26 |
| Southern Andes | 1.94 / **0.83** / 0.96 | 1.69 / **1.03** / 1.28 | 2.16 / **1.90** / 2.37 | 2.14 / **1.80** / 2.30 |

Energy score per root place, mm (lower is better).

| Place | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Jamaica | 3.79 / **2.89** / 2.97 | 3.72 / **3.10** / 3.15 | 3.81 / **3.34** / 3.56 | 4.14 / **3.92** / 4.49 |
| Caribbean | 6.33 / **5.42** / 5.43 | 6.29 / 5.49 / **5.48** | 6.04 / **5.32** / 5.35 | 6.16 / **5.63** / 5.67 |
| Maritime Continent | 11.49 / 10.27 / **10.23** | 11.47 / 10.33 / **10.29** | 11.18 / 10.24 / **10.23** | 11.22 / **10.33** / 10.33 |
| Alps | 3.02 / **2.26** / 2.29 | 3.16 / **2.78** / 2.90 | 3.68 / **3.52** / 3.81 | 4.05 / **3.84** / 4.28 |
| US Pacific Northwest | 2.87 / **2.06** / 2.07 | 2.26 / **1.97** / 2.05 | 3.35 / **3.02** / 3.21 | 3.51 / **3.36** / 3.51 |
| US Great Plains | 3.24 / **2.98** / 3.00 | 3.35 / **3.20** / 3.35 | 2.65 / **2.60** / 2.94 | 3.60 / **3.58** / 4.01 |
| Southern Andes | 4.28 / **2.36** / 2.39 | 3.66 / **2.51** / 2.61 | 4.75 / **4.28** / 4.40 | 4.74 / **4.15** / 4.40 |

Variogram score for places within 150 km (lower is better).

| Place | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Jamaica | 1.623 / **1.235** / 1.283 | 1.636 / **1.309** / 1.412 | 1.876 / **1.759** / 1.811 | **1.956** / 2.118 / 2.193 |
| Caribbean | 2.049 / **1.570** / 1.599 | 2.048 / **1.630** / 1.681 | 2.065 / **1.720** / 1.830 | 2.149 / **1.896** / 2.034 |
| Maritime Continent | 3.462 / **3.086** / 3.196 | 3.500 / **3.167** / 3.374 | 3.458 / **3.224** / 3.547 | 3.683 / **3.550** / 4.002 |
| Alps | 0.978 / **0.640** / 0.692 | 1.126 / **0.928** / 1.064 | 1.388 / **1.232** / 1.477 | 1.632 / **1.440** / 1.767 |
| US Pacific Northwest | 0.722 / **0.554** / 0.586 | 0.647 / **0.544** / 0.599 | 1.058 / **0.935** / 1.034 | 1.195 / **1.155** / 1.249 |
| US Great Plains | 0.801 / **0.744** / 0.817 | 0.949 / **0.890** / 1.089 | 0.835 / **0.816** / 1.079 | 1.298 / **1.288** / 1.612 |
| Southern Andes | 1.112 / **0.534** / 0.554 | 1.028 / **0.587** / 0.661 | 1.307 / **1.022** / 1.106 | 1.379 / **1.096** / 1.200 |

## What each part added

Brier skill, 1 mm (mean over 7 places).

| Design | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Pool alone | +0.3583 | +0.2491 | +0.1231 | +0.0396 |
| Pool + online weights | +0.3591 | +0.2494 | +0.1226 | +0.0390 |
| Pool + hand-over, always on | +0.3583 | +0.2491 | +0.1230 | +0.0397 |
| Pool + online weights + hand-over, always on (as first declared) | +0.3591 | +0.2494 | +0.1226 | +0.0390 |
| Pool + online weights + hand-over where the staged storms say so (Pluvium Blend) | +0.3591 | +0.2494 | +0.1226 | +0.0390 |

Brier skill, 25 mm (mean over 7 places).

| Design | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Pool alone | +0.2634 | +0.1582 | +0.0710 | +0.0075 |
| Pool + online weights | +0.2647 | +0.1586 | +0.0701 | +0.0093 |
| Pool + hand-over, always on | +0.2633 | +0.1577 | +0.0707 | +0.0076 |
| Pool + online weights + hand-over, always on (as first declared) | +0.2647 | +0.1581 | +0.0697 | +0.0094 |
| Pool + online weights + hand-over where the staged storms say so (Pluvium Blend) | +0.2653 | +0.1582 | +0.0700 | +0.0094 |

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

RMSE (mean over 7 places).

| Design | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Pool alone | 7.1442 | 8.0660 | 8.7846 | 9.2051 |
| Pool + online weights | 7.1114 | 8.0628 | 8.7912 | 9.2069 |
| Pool + hand-over, always on | 6.9663 | 8.0316 | 8.7995 | 9.2036 |
| Pool + online weights + hand-over, always on (as first declared) | 6.9542 | 8.0282 | 8.8062 | 9.2052 |
| Pool + online weights + hand-over where the staged storms say so (Pluvium Blend) | 6.9431 | 8.0232 | 8.7919 | 9.2051 |

## Hurricane Melissa over Jamaica

Observed over Jamaica on 2025-10-28 (CMORPH): 154 mm on average, up to 447 mm; 100% of places had 50 mm or more and 96% had 100 mm or more.

| Days ahead | First Blend: chance of 50 mm, expected mm | New Blend: chance of 50 mm, of 100 mm, expected mm | Places under an extreme warning | Raw consensus: chance of 50 mm, mean mm |
|---|---|---|---|---|
| 2 | 66%, 63 | 92%, 75%, 195 | 97% | 94%, 173 |
| 4 | 42%, 50 | 47%, 21%, 78 | 35% | 47%, 77 |
| 8 | 9%, 16 | 10%, 1%, 17 | 0% | 18%, 28 |
| 12 | 2%, 5 | 1%, 0%, 5 | 0% | 23%, 41 |

How rare that was: a generalised Pareto fit to Jamaica's CMORPH record from 2020-01 to 2025-04 (5.1 years, 450 places, all before the test year) puts the daily amount reached once a year at 79 mm on average over the places, and once in five years at 123 mm. On 2025-10-28, 100% of places had more than their once-a-year amount, 78% more than their once-in-five-years amount, and 35% an amount rarer than once in 20 years, which is as far as a record of that length can say. The median place had a day expected about once in 13 years. These are rough figures from five years of satellite estimates.

## What is still weak

- At 50 mm the Blend is behind some other system in 13 of 28 place and lead-group cells, 4 of them on days 1-7. The largest gaps: Jamaica days 11-14 (0.357 behind ECMWF HRES raw); Jamaica days 8-10 (0.052 behind ECMWF HRES); US Pacific Northwest days 8-10 (0.016 behind ECMWF ENS); US Great Plains days 4-7 (0.007 behind ECMWF ENS raw). Beyond a week the skill of every system at 50 mm is close to zero and one storm can decide a cell.
- On days 11-14 the anomaly correlation of the Blend's expected amount is behind another system in 4 of 7 places: Alps +0.05 against +0.26 (ECMWF ENS raw); US Great Plains +0.17 against +0.29 (ECMWF ENS raw); US Pacific Northwest +0.22 against +0.28 (ECMWF AIFS raw); Jamaica +0.15 against +0.17 (ECMWF AIFS raw). A calibrated expected amount is smooth by construction, and a raw model's field keeps its pattern even when the pattern is in the wrong place. A correction that traded RMSE for this score was tested and not adopted ([WHAT_DID_NOT_WORK.md](WHAT_DID_NOT_WORK.md)).
- Days 11-14 carry little skill for anyone. Brier skill at 1 mm on those days: Jamaica -0.006, Caribbean +0.054, Maritime Continent +0.113, Alps +0.032, US Pacific Northwest +0.002, US Great Plains +0.033, Southern Andes +0.045. Zero is the climatology.
- In the Alps and the US Pacific Northwest the staged storms were not forecast better with the hand-over, so it is off there, and a storm beyond the record is forecast by the calibration alone, which shrinks it. The table still flags such a forecast.
- The verdict on the hand-over rests on four storms per place. It had the sign of what the hand-over then did in the test year in 7 of 7 places. That is partly hindsight: the rule was fixed after 5 of the 7 places had been scored. It can be overruled (`pluvium blend train --handover always` or `never`).
- The online weights were scored with each day's rain known two days later. The satellite record is published about four and a half months late, so a live system learns that much later.
- The quarterly refits used every run before each quarter. A live system can train only on runs whose rain has been published, which are about five months old. That gap was not tested.
- The truth is one satellite product. CMORPH underestimates frozen and orographic precipitation and the heaviest rain. Winter was scored against radar and gauges for the two US places only.
- The test year is the one on which the weak spots were found. A fresh year is the test that counts, and it does not exist yet.

## Every place

Each cell: the new Blend (best other system and which). Bold where the Blend is ahead.

### Jamaica at 8 km (with Pluvium's network)

450 places.

| Score | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Brier skill, 1 mm | **+0.191** (+0.179, NOAA GraphCast) | **+0.137** (+0.118, ECMWF AIFS) | +0.074 (+0.086, NOAA GraphCast) | -0.006 (-0.005, ECMWF AIFS) |
| Brier skill, 10 mm | **+0.200** (+0.198, ECMWF AIFS) | **+0.161** (+0.146, ECMWF ENS) | **+0.077** (+0.043, ECMWF AIFS) | -0.019 (+0.008, ECMWF HRES) |
| Brier skill, 25 mm | +0.315 (+0.336, ECMWF AIFS) | **+0.219** (+0.219, ECMWF ENS) | **+0.099** (+0.070, ECMWF HRES) | -0.003 (+0.080, ECMWF HRES) |
| Brier skill, 50 mm | **+0.515** (+0.511, NOAA GraphCast) | **+0.247** (+0.242, ECMWF ENS raw) | +0.065 (+0.117, ECMWF HRES) | -0.003 (+0.355, ECMWF HRES raw) |
| CSI, 1 mm | **0.385** (0.378, NOAA GraphCast) | 0.345 (0.349, NOAA GraphCast) | 0.294 (0.318, NOAA GraphCast) | 0.253 (0.264, ECMWF AIFS) |
| CSI, 10 mm | 0.248 (0.257, ECMWF AIFS raw) | 0.225 (0.225, ECMWF ENS) | 0.137 (0.144, NOAA GraphCast) | 0.033 (0.115, ECMWF HRES) |
| ROC area, 1 mm | **0.802** (0.792, NOAA GraphCast) | **0.754** (0.747, NOAA GraphCast) | 0.688 (0.706, NOAA GraphCast) | 0.618 (0.627, ECMWF AIFS) |
| CRPS | **1.546** (1.620, NOAA GraphCast) | **2.001** (2.081, ECMWF ENS) | **2.392** (2.438, ECMWF HRES) | 2.450 (2.375, ECMWF HRES) |
| RMSE | **7.111** (7.266, ECMWF ENS raw) | 10.118 (10.071, ECMWF ENS raw) | 12.261 (11.976, ECMWF HRES) | 12.539 (9.475, ECMWF HRES raw) |
| ACC | **+0.404** (+0.403, ECMWF AIFS) | +0.353 (+0.362, ECMWF AIFS) | +0.252 (+0.264, ECMWF AIFS) | +0.153 (+0.173, ECMWF AIFS raw) |

Online weights for light rain on days 1-3, at the end of each quarter of the test year:

| After | ECMWF ENS | NOAA GEFS | ECMWF HRES | ECMWF AIFS | NOAA GraphCast | NOAA GFS | one network | Pluvium's network | Pluvium Blend | the pool |
|---|---|---|---|---|---|---|---|---|---|---|
| quarter 1 | 0.01 | 0.01 | 0.01 | 0.02 | 0.04 | 0.01 | 0.02 | 0.00 | 0.01 | 0.86 |
| quarter 2 | 0.01 | 0.01 | 0.01 | 0.03 | 0.03 | 0.01 | 0.05 | 0.01 | 0.09 | 0.73 |
| quarter 3 | 0.04 | 0.03 | 0.03 | 0.05 | 0.00 | 0.03 | 0.03 | 0.04 | 0.06 | 0.69 |
| quarter 4 | 0.01 | 0.01 | 0.01 | 0.03 | 0.08 | 0.01 | 0.21 | 0.01 | 0.15 | 0.47 |

### Caribbean (tropical islands and coasts)

1,500 places.

| Score | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Brier skill, 1 mm | **+0.229** (+0.219, ECMWF AIFS) | **+0.168** (+0.157, ECMWF AIFS) | **+0.102** (+0.089, ECMWF AIFS) | **+0.054** (+0.043, ECMWF ENS) |
| Brier skill, 10 mm | **+0.160** (+0.156, ECMWF AIFS) | **+0.111** (+0.108, ECMWF AIFS) | **+0.065** (+0.060, ECMWF AIFS) | **+0.037** (+0.027, ECMWF ENS) |
| Brier skill, 25 mm | **+0.114** (+0.113, ECMWF AIFS) | **+0.079** (+0.077, ECMWF AIFS) | **+0.043** (+0.040, ECMWF AIFS) | **+0.027** (+0.018, ECMWF ENS) |
| Brier skill, 50 mm | **+0.087** (+0.083, ECMWF AIFS) | +0.056 (+0.057, ECMWF AIFS) | **+0.029** (+0.028, ECMWF AIFS) | **+0.018** (+0.011, ECMWF ENS) |
| CSI, 1 mm | **0.491** (0.485, ECMWF AIFS) | **0.454** (0.447, ECMWF AIFS) | **0.414** (0.410, ECMWF AIFS) | **0.389** (0.388, ECMWF ENS) |
| CSI, 10 mm | **0.267** (0.265, ECMWF AIFS) | **0.233** (0.231, ECMWF AIFS) | **0.196** (0.191, ECMWF AIFS) | **0.174** (0.167, ECMWF AIFS) |
| ROC area, 1 mm | **0.853** (0.848, ECMWF AIFS) | **0.822** (0.816, ECMWF AIFS) | **0.785** (0.776, ECMWF AIFS) | **0.754** (0.751, ECMWF ENS) |
| CRPS | **2.033** (2.044, ECMWF AIFS) | **2.166** (2.176, ECMWF AIFS) | **2.287** (2.301, ECMWF AIFS) | **2.357** (2.380, ECMWF ENS) |
| RMSE | 7.964 (7.943, ECMWF AIFS) | 8.311 (8.306, ECMWF AIFS) | **8.650** (8.677, ECMWF AIFS) | **8.787** (8.858, ECMWF ENS) |
| ACC | **+0.453** (+0.447, ECMWF AIFS) | **+0.386** (+0.382, ECMWF AIFS) | **+0.303** (+0.292, ECMWF AIFS) | **+0.251** (+0.224, ECMWF ENS) |

Online weights for light rain on days 1-3, at the end of each quarter of the test year:

| After | ECMWF ENS | NOAA GEFS | ECMWF HRES | ECMWF AIFS | NOAA GraphCast | NOAA GFS | one network | the pool |
|---|---|---|---|---|---|---|---|---|
| quarter 1 | 0.02 | 0.01 | 0.02 | 0.05 | 0.05 | 0.01 | 0.10 | 0.74 |
| quarter 2 | 0.03 | 0.02 | 0.02 | 0.05 | 0.03 | 0.02 | 0.10 | 0.73 |
| quarter 3 | 0.05 | 0.02 | 0.03 | 0.05 | 0.02 | 0.02 | 0.10 | 0.71 |
| quarter 4 | 0.03 | 0.01 | 0.02 | 0.06 | 0.04 | 0.01 | 0.11 | 0.73 |

### Maritime Continent (tropical, deep convection)

1,500 places.

| Score | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Brier skill, 1 mm | **+0.306** (+0.295, ECMWF AIFS) | **+0.244** (+0.227, ECMWF AIFS) | **+0.169** (+0.156, ECMWF AIFS) | **+0.113** (+0.110, ECMWF AIFS) |
| Brier skill, 10 mm | **+0.213** (+0.208, ECMWF AIFS) | **+0.152** (+0.141, ECMWF AIFS) | **+0.101** (+0.092, ECMWF AIFS) | **+0.068** (+0.060, ECMWF AIFS) |
| Brier skill, 25 mm | **+0.135** (+0.132, ECMWF AIFS) | **+0.089** (+0.083, ECMWF AIFS) | **+0.056** (+0.052, ECMWF AIFS) | **+0.039** (+0.032, ECMWF AIFS) |
| Brier skill, 50 mm | **+0.082** (+0.076, ECMWF AIFS) | **+0.050** (+0.046, ECMWF AIFS) | **+0.031** (+0.029, ECMWF AIFS) | **+0.021** (+0.018, ECMWF AIFS) |
| CSI, 1 mm | **0.645** (0.641, ECMWF AIFS) | **0.623** (0.618, ECMWF AIFS) | **0.596** (0.591, ECMWF AIFS) | 0.572 (0.572, ECMWF AIFS) |
| CSI, 10 mm | **0.369** (0.364, ECMWF AIFS) | **0.328** (0.319, ECMWF AIFS) | **0.290** (0.283, ECMWF AIFS raw) | **0.260** (0.257, ECMWF AIFS) |
| ROC area, 1 mm | **0.824** (0.818, ECMWF AIFS) | **0.787** (0.777, ECMWF AIFS) | **0.739** (0.731, ECMWF AIFS) | **0.696** (0.694, ECMWF AIFS) |
| CRPS | **4.994** (5.037, ECMWF AIFS) | **5.296** (5.353, ECMWF AIFS) | **5.493** (5.531, ECMWF AIFS) | **5.705** (5.738, ECMWF AIFS) |
| RMSE | **14.815** (14.925, ECMWF AIFS raw) | **15.549** (15.672, ECMWF AIFS) | **15.907** (15.975, ECMWF AIFS) | **16.320** (16.383, ECMWF AIFS) |
| ACC | **+0.454** (+0.449, ECMWF AIFS) | **+0.383** (+0.371, ECMWF AIFS) | **+0.304** (+0.293, ECMWF AIFS) | **+0.266** (+0.248, ECMWF AIFS raw) |

Online weights for light rain on days 1-3, at the end of each quarter of the test year:

| After | ECMWF ENS | NOAA GEFS | ECMWF HRES | ECMWF AIFS | NOAA GraphCast | NOAA GFS | one network | the pool |
|---|---|---|---|---|---|---|---|---|
| quarter 1 | 0.02 | 0.01 | 0.01 | 0.04 | 0.03 | 0.01 | 0.12 | 0.76 |
| quarter 2 | 0.02 | 0.01 | 0.01 | 0.02 | 0.06 | 0.01 | 0.12 | 0.74 |
| quarter 3 | 0.03 | 0.01 | 0.02 | 0.06 | 0.01 | 0.01 | 0.14 | 0.71 |
| quarter 4 | 0.02 | 0.01 | 0.01 | 0.07 | 0.02 | 0.01 | 0.13 | 0.73 |

### Alps (mid-latitude mountains)

907 places.

| Score | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Brier skill, 1 mm | **+0.505** (+0.492, ECMWF ENS) | **+0.341** (+0.315, ECMWF ENS) | **+0.141** (+0.118, ECMWF ENS) | **+0.032** (+0.029, ECMWF ENS) |
| Brier skill, 10 mm | **+0.426** (+0.411, ECMWF ENS) | **+0.250** (+0.230, ECMWF AIFS) | **+0.089** (+0.075, ECMWF AIFS) | **+0.018** (+0.015, ECMWF ENS) |
| Brier skill, 25 mm | **+0.258** (+0.253, ECMWF AIFS) | +0.114 (+0.117, ECMWF AIFS) | **+0.043** (+0.041, ECMWF AIFS) | **+0.003** (+0.001, ECMWF ENS) |
| Brier skill, 50 mm | +0.091 (+0.097, ECMWF AIFS) | +0.032 (+0.038, ECMWF AIFS) | +0.020 (+0.021, NOAA GEFS) | +0.001 (+0.002, ECMWF ENS) |
| CSI, 1 mm | **0.576** (0.569, ECMWF ENS) | **0.467** (0.450, ECMWF ENS) | **0.339** (0.325, ECMWF ENS) | 0.252 (0.256, ECMWF ENS) |
| CSI, 10 mm | **0.439** (0.418, ECMWF ENS) | **0.315** (0.301, ECMWF AIFS) | **0.179** (0.171, ECMWF AIFS) | 0.072 (0.086, ECMWF AIFS raw) |
| ROC area, 1 mm | **0.925** (0.922, ECMWF ENS) | **0.872** (0.861, ECMWF ENS) | **0.761** (0.745, ECMWF ENS) | **0.668** (0.666, ECMWF ENS) |
| CRPS | **1.222** (1.248, ECMWF ENS) | **1.524** (1.560, ECMWF AIFS) | **1.748** (1.779, ECMWF AIFS) | **1.971** (1.976, ECMWF ENS) |
| RMSE | **4.877** (4.948, ECMWF ENS) | **5.731** (5.784, ECMWF AIFS) | **6.041** (6.097, ECMWF AIFS) | **6.753** (6.761, ECMWF ENS) |
| ACC | **+0.699** (+0.683, ECMWF ENS) | **+0.537** (+0.511, ECMWF AIFS) | +0.282 (+0.346, ECMWF AIFS raw) | +0.048 (+0.261, ECMWF ENS raw) |

Online weights for light rain on days 1-3, at the end of each quarter of the test year:

| After | ECMWF ENS | NOAA GEFS | ECMWF HRES | ECMWF AIFS | NOAA GraphCast | NOAA GFS | one network | the pool |
|---|---|---|---|---|---|---|---|---|
| quarter 1 | 0.02 | 0.03 | 0.01 | 0.01 | 0.02 | 0.02 | 0.13 | 0.76 |
| quarter 2 | 0.03 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.20 | 0.70 |
| quarter 3 | 0.07 | 0.02 | 0.03 | 0.02 | 0.01 | 0.02 | 0.14 | 0.69 |
| quarter 4 | 0.05 | 0.01 | 0.03 | 0.04 | 0.01 | 0.02 | 0.10 | 0.76 |

### US Pacific Northwest (mid-latitude coast and mountains)

1,378 places.

| Score | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Brier skill, 1 mm | **+0.379** (+0.365, ECMWF ENS) | **+0.253** (+0.234, ECMWF ENS) | **+0.093** (+0.077, ECMWF AIFS) | +0.002 (+0.003, ECMWF AIFS) |
| Brier skill, 10 mm | **+0.351** (+0.341, ECMWF ENS) | **+0.215** (+0.200, ECMWF ENS) | **+0.095** (+0.085, ECMWF ENS) | -0.015 (-0.005, ECMWF AIFS) |
| Brier skill, 25 mm | **+0.334** (+0.327, ECMWF ENS) | **+0.205** (+0.189, ECMWF ENS) | **+0.120** (+0.110, ECMWF ENS) | -0.021 (-0.013, NOAA GEFS) |
| Brier skill, 50 mm | **+0.307** (+0.288, ECMWF ENS) | **+0.168** (+0.160, ECMWF ENS) | +0.097 (+0.114, ECMWF ENS) | +0.008 (+0.009, ECMWF HRES) |
| CSI, 1 mm | **0.473** (0.469, ECMWF ENS) | **0.397** (0.384, ECMWF ENS) | **0.276** (0.271, ECMWF ENS) | 0.205 (0.215, ECMWF AIFS) |
| CSI, 10 mm | **0.398** (0.390, ECMWF ENS) | **0.300** (0.293, ECMWF AIFS) | **0.218** (0.216, ECMWF ENS) | 0.141 (0.156, ECMWF AIFS) |
| ROC area, 1 mm | **0.912** (0.908, ECMWF ENS) | **0.865** (0.853, ECMWF AIFS) | **0.754** (0.743, ECMWF AIFS) | 0.679 (0.681, ECMWF ENS) |
| CRPS | **0.905** (0.919, ECMWF ENS) | **1.096** (1.119, ECMWF ENS) | **1.187** (1.201, ECMWF ENS) | 1.434 (1.426, ECMWF AIFS) |
| RMSE | **3.921** (3.968, ECMWF ENS) | **4.720** (4.796, ECMWF ENS) | **5.085** (5.116, ECMWF ENS) | 5.853 (5.822, ECMWF AIFS) |
| ACC | **+0.717** (+0.711, ECMWF ENS) | **+0.588** (+0.581, ECMWF AIFS) | +0.388 (+0.400, ECMWF AIFS) | +0.216 (+0.283, ECMWF AIFS raw) |

Online weights for light rain on days 1-3, at the end of each quarter of the test year:

| After | ECMWF ENS | NOAA GEFS | ECMWF HRES | ECMWF AIFS | NOAA GraphCast | NOAA GFS | one network | the pool |
|---|---|---|---|---|---|---|---|---|
| quarter 1 | 0.05 | 0.03 | 0.02 | 0.03 | 0.02 | 0.02 | 0.10 | 0.73 |
| quarter 2 | 0.03 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.27 | 0.62 |
| quarter 3 | 0.05 | 0.01 | 0.02 | 0.02 | 0.00 | 0.01 | 0.16 | 0.73 |
| quarter 4 | 0.05 | 0.01 | 0.04 | 0.06 | 0.01 | 0.01 | 0.08 | 0.75 |

### US Great Plains (mid-latitude storms)

1,500 places.

| Score | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Brier skill, 1 mm | **+0.479** (+0.463, ECMWF AIFS) | **+0.312** (+0.289, ECMWF AIFS) | **+0.139** (+0.120, ECMWF ENS) | **+0.033** (+0.028, ECMWF ENS) |
| Brier skill, 10 mm | **+0.406** (+0.392, ECMWF AIFS) | **+0.243** (+0.218, ECMWF AIFS) | **+0.096** (+0.077, ECMWF ENS) | **+0.019** (+0.018, ECMWF ENS) |
| Brier skill, 25 mm | **+0.282** (+0.273, ECMWF AIFS) | **+0.145** (+0.125, ECMWF AIFS) | **+0.054** (+0.043, ECMWF ENS) | **+0.007** (+0.006, ECMWF ENS) |
| Brier skill, 50 mm | **+0.129** (+0.114, ECMWF AIFS) | +0.029 (+0.035, ECMWF ENS raw) | +0.015 (+0.017, ECMWF AIFS) | +0.003 (+0.004, ECMWF AIFS) |
| CSI, 1 mm | **0.549** (0.537, ECMWF AIFS) | **0.435** (0.424, ECMWF AIFS) | **0.333** (0.329, ECMWF ENS) | **0.229** (0.227, NOAA GFS) |
| CSI, 10 mm | **0.430** (0.412, NOAA GraphCast) | **0.310** (0.293, ECMWF AIFS) | 0.154 (0.161, ECMWF ENS) | 0.043 (0.090, ECMWF ENS) |
| ROC area, 1 mm | **0.937** (0.930, ECMWF AIFS) | **0.885** (0.870, ECMWF AIFS) | **0.805** (0.794, ECMWF ENS) | **0.722** (0.718, ECMWF ENS) |
| CRPS | **1.249** (1.280, ECMWF AIFS) | **1.486** (1.527, ECMWF AIFS) | **1.784** (1.815, ECMWF ENS) | **1.895** (1.899, ECMWF ENS) |
| RMSE | **5.442** (5.524, ECMWF AIFS raw) | **6.186** (6.302, ECMWF AIFS) | **6.921** (7.012, ECMWF ENS) | **7.206** (7.222, ECMWF ENS) |
| ACC | **+0.730** (+0.720, ECMWF AIFS raw) | +0.575 (+0.579, NOAA GraphCast raw) | +0.415 (+0.430, ECMWF AIFS raw) | +0.171 (+0.285, ECMWF ENS raw) |

Online weights for light rain on days 1-3, at the end of each quarter of the test year:

| After | ECMWF ENS | NOAA GEFS | ECMWF HRES | ECMWF AIFS | NOAA GraphCast | NOAA GFS | one network | the pool |
|---|---|---|---|---|---|---|---|---|
| quarter 1 | 0.01 | 0.01 | 0.01 | 0.04 | 0.02 | 0.01 | 0.14 | 0.77 |
| quarter 2 | 0.01 | 0.01 | 0.01 | 0.04 | 0.02 | 0.01 | 0.16 | 0.73 |
| quarter 3 | 0.04 | 0.03 | 0.02 | 0.02 | 0.03 | 0.03 | 0.12 | 0.72 |
| quarter 4 | 0.05 | 0.01 | 0.02 | 0.04 | 0.01 | 0.01 | 0.10 | 0.77 |

### Southern Andes and Chile (southern mid-latitudes)

1,500 places.

| Score | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Brier skill, 1 mm | **+0.424** (+0.417, ECMWF ENS) | **+0.290** (+0.279, ECMWF ENS) | **+0.141** (+0.131, ECMWF ENS) | +0.045 (+0.051, ECMWF ENS) |
| Brier skill, 10 mm | **+0.464** (+0.449, ECMWF ENS) | **+0.303** (+0.285, ECMWF ENS) | **+0.112** (+0.107, ECMWF ENS) | +0.026 (+0.030, ECMWF ENS) |
| Brier skill, 25 mm | **+0.419** (+0.387, ECMWF ENS) | **+0.257** (+0.235, ECMWF ENS) | **+0.074** (+0.071, ECMWF ENS) | +0.014 (+0.018, ECMWF ENS) |
| Brier skill, 50 mm | **+0.283** (+0.228, ECMWF AIFS) | **+0.133** (+0.107, ECMWF ENS) | **+0.032** (+0.029, ECMWF ENS) | +0.006 (+0.006, ECMWF ENS) |
| CSI, 1 mm | **0.535** (0.528, ECMWF ENS) | **0.455** (0.448, ECMWF ENS) | **0.374** (0.365, ECMWF ENS) | 0.307 (0.316, ECMWF ENS) |
| CSI, 10 mm | **0.499** (0.493, ECMWF ENS) | **0.392** (0.381, ECMWF ENS) | **0.249** (0.247, ECMWF ENS) | 0.152 (0.180, ECMWF ENS raw) |
| ROC area, 1 mm | **0.933** (0.930, ECMWF ENS) | **0.895** (0.892, ECMWF ENS) | **0.841** (0.838, ECMWF ENS) | 0.798 (0.800, ECMWF ENS) |
| CRPS | **0.898** (0.937, ECMWF ENS) | **1.144** (1.170, ECMWF ENS) | **1.423** (1.430, ECMWF ENS) | 1.532 (1.526, ECMWF ENS) |
| RMSE | **4.471** (4.737, ECMWF AIFS) | **5.548** (5.654, ECMWF ENS) | **6.677** (6.685, ECMWF ENS) | 6.976 (6.956, ECMWF ENS) |
| ACC | **+0.742** (+0.733, ECMWF ENS) | **+0.583** (+0.573, ECMWF ENS) | **+0.380** (+0.376, ECMWF ENS) | **+0.197** (+0.191, ECMWF ENS) |

Online weights for light rain on days 1-3, at the end of each quarter of the test year:

| After | ECMWF ENS | NOAA GEFS | ECMWF HRES | ECMWF AIFS | NOAA GraphCast | NOAA GFS | one network | the pool |
|---|---|---|---|---|---|---|---|---|
| quarter 1 | 0.06 | 0.02 | 0.03 | 0.03 | 0.02 | 0.01 | 0.18 | 0.65 |
| quarter 2 | 0.06 | 0.01 | 0.03 | 0.03 | 0.02 | 0.01 | 0.14 | 0.70 |
| quarter 3 | 0.06 | 0.08 | 0.02 | 0.02 | 0.01 | 0.03 | 0.15 | 0.63 |
| quarter 4 | 0.05 | 0.02 | 0.02 | 0.03 | 0.01 | 0.02 | 0.15 | 0.70 |

