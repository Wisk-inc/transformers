# Pluvium against the operational systems

Pluvium against ECMWF ENS, ECMWF HRES, ECMWF AIFS, NOAA GEFS, NOAA GraphCast and NOAA GFS. Only real, archived 00 UTC
runs are used, with no reanalysis anywhere. The question: what is the chance of at least 1, 10, 25 and 50 mm of rain
on each of the next three days, at 450 land cells of 8 km covering Jamaica? Truth is CMORPH.

## The short answer

Not golden tier yet. Pluvium is the best system on Jamaica for "will it rain" on days 1 and 2 in one of the two test
periods. Its lead does not yet hold against every rival at 95% confidence, and it lost the other period.

**July and August 2025.** This is the clean test: the protocol was frozen and committed before any of its data were
downloaded.

* **Rain at all (at least 1 mm).** Pluvium has the highest skill of any system on days 1 and 2. Its Brier skill
  against climatology is +0.127 and +0.118. The best operational system scores +0.087 and +0.091, and the blend of
  all six +0.090 and +0.095.
  * With 95% confidence it beats ECMWF ENS and ECMWF HRES on all three days. It beats AIFS and GraphCast on days 1
    and 2, and the blend of all six on day 1.
  * Against GEFS and GFS it is better on days 1 and 2 in 86-97% of bootstrap resamples, but the 95% interval still
    includes no difference.
  * On day 3 it ties everything except ENS and HRES.
* **10 mm or more.** A tie with the best system on every day.
* **25 and 50 mm.** No system, Pluvium included, does meaningfully better than a forecast that never says rain (the
  largest margin is under 0.01). Rain this heavy at one 8 km cell on one day is too rare (0.5% and 0.1% of cell-days)
  for any of them.
* **Daily totals.** Pluvium's CRPS is 40-50% smaller than ECMWF ENS's and GEFS's on every day. This compares raw
  outputs, and only Pluvium was trained on CMORPH, so part of the gap is calibration.
* **Added information.** Adding Pluvium to the blend of all six raises skill at 1 mm from +0.090 to +0.136 on day 1,
  and from +0.095 to +0.134 on day 2.

**May and June 2025.** This was the first test, with its protocol fixed in advance. Pluvium lost.

* After calibration it was worse than the best operational system on 9 of the 12 threshold-day comparisons (95%
  confidence) and tied on 3.
* Its raw forecasts still had the best discrimination: ROC area 0.76 at 1 mm and 0.84 at 10 mm on day 1.
* They also had the best CRPS: 0.94 mm against 1.82 for ECMWF ENS on day 1.

## Why the two tests differ, and why the protocol changed

Every system is calibrated on the 30 runs before each test day, so early-May forecasts were calibrated on April.

* April 2025 was unusually wet on Jamaica. CMORPH gave 2.6 mm per cell and day, more than May's 0.8 mm.
* Pluvium forecast 0.6 mm. It had never trained on an April: stage B ran from May 2024 to March 2025. Its
  climatology input also came from a single April, April 2024.
* The calibration therefore learned that any rain Pluvium forecasts means a heavy event. On the first days of May it
  turned modest Pluvium forecasts into 96-98% chances of 25 mm, on days when at most 6% of cells saw that much.
* The first protocol used isotonic regression, which also overfit: one wet day dominated each window's highest
  chances, so Pluvium's calibrated 80-90% chances verified about a third of the time.

The protocol was then changed to logistic regression (two parameters, standard in precipitation post-processing). It
was frozen in a commit before July and August were downloaded. Both protocols are reported for both periods below,
with Brier skill for at least 1 mm:

| Test period | Calibration | Pluvium, days 1 / 2 / 3 | Best operational system, days 1 / 2 / 3 |
|---|---|---|---|
| May-June 2025 | isotonic | -0.032 / -0.034 / +0.021 | +0.082 / +0.083 / +0.066 |
| May-June 2025 | logistic | +0.051 / +0.026 / -0.048 | +0.097 / +0.107 / +0.072 |
| July-August 2025 | isotonic | +0.126 / +0.118 / +0.014 | +0.083 / +0.084 / +0.067 |
| July-August 2025 | logistic (frozen) | +0.127 / +0.118 / +0.052 | +0.087 / +0.091 / +0.063 |

The second test's calibration windows contain no April. That is luck of the calendar, not a fix. The lesson for
Pluvium is in the first item below.

## What would make it golden tier

1. **Train on every month, over several years.** Stage B never saw an April. Replace the one-year climatology input
   with a multi-year one.
2. **Give Pluvium the ensembles.** GEFS and ENS earn their day 3 skill from their spread. Pluvium can already mix
   GEFS-driven chains live, and ensemble statistics could become inputs.
3. **Train the real model.** This test used pluvium-small (0.9M parameters) trained for about two hours on a 4-core
   CPU. Pluvium-500 is 60 times larger and is designed for a day on an RTX PRO 6000.
4. **Calibrate on the same weeks of earlier years.** Season-matched windows, as reforecast-based calibration does,
   replace the trailing 30 days, which a dry-to-wet transition breaks.

Run it again with `pluvium operational all --root RUN --geography GEO --checkpoint CKPT --stations STATIONS`. Add
`--test-period 2025-07-01T00,2025-09-01T00 --calibration logistic` for the second test.

## Results: July and August 2025 (the frozen protocol, logistic calibration)

Test: 62 daily 00 UTC runs, 2025-07-01 to 2025-08-31, at 450 land cells of 8 km covering Jamaica. Truth: CMORPH. Every forecast is a real, archived operational run; none uses a reanalysis.

Brier skill score against the climatology of the same weeks (higher is better; 0 is climatology). Each system is calibrated the same causal way (logistic regression on the 30 previous runs). The raw score is in brackets.

### 24-hour rain of at least 1 mm

| System | Day 1 | Day 2 | Day 3 |
|---|---|---|---|
| **Pluvium (GFS-driven)** | +0.127 (+0.127) | +0.118 (+0.116) | +0.052 (+0.046) |
| ECMWF ENS | +0.056 (-1.493) | +0.029 (-1.951) | +0.014 (-1.999) |
| NOAA GEFS | +0.069 (-2.043) | +0.091 (-1.683) | +0.043 (-1.732) |
| ECMWF HRES | +0.030 (-2.261) | +0.011 (-2.916) | -0.001 (-2.924) |
| ECMWF AIFS | +0.082 (-2.421) | +0.069 (-2.734) | +0.063 (-2.827) |
| NOAA GraphCast | +0.073 (-2.351) | +0.064 (-2.122) | +0.049 (-2.069) |
| NOAA GFS | +0.087 (-2.846) | +0.091 (-2.932) | +0.058 (-3.103) |
| Blend of the 6 operational systems | +0.090 | +0.095 | +0.061 |
| Blend of the 6 + Pluvium | +0.136 | +0.134 | +0.081 |
| *A forecast that never says rain* | *-0.201* | *-0.209* | *-0.210* |

Share of cell-days with at least 1 mm on day 1: 17.3%. A system shows skill at this threshold only where it beats the forecast that never says rain.

### 24-hour rain of at least 10 mm

| System | Day 1 | Day 2 | Day 3 |
|---|---|---|---|
| **Pluvium (GFS-driven)** | +0.066 (+0.058) | +0.088 (+0.071) | +0.092 (+0.074) |
| ECMWF ENS | +0.047 (+0.080) | +0.074 (+0.049) | +0.088 (+0.011) |
| NOAA GEFS | +0.068 (-0.057) | +0.068 (+0.058) | +0.091 (+0.075) |
| ECMWF HRES | +0.063 (-0.195) | +0.072 (-0.433) | +0.080 (-0.278) |
| ECMWF AIFS | +0.077 (+0.038) | +0.090 (+0.061) | +0.093 (+0.071) |
| NOAA GraphCast | +0.074 (-0.029) | +0.070 (-0.006) | +0.082 (+0.004) |
| NOAA GFS | +0.088 (-0.247) | +0.035 (-0.535) | +0.061 (-0.249) |
| Blend of the 6 operational systems | +0.059 | +0.038 | +0.054 |
| Blend of the 6 + Pluvium | +0.047 | +0.036 | +0.038 |
| *A forecast that never says rain* | *+0.063* | *+0.061* | *+0.071* |

Share of cell-days with at least 10 mm on day 1: 2.3%. A system shows skill at this threshold only where it beats the forecast that never says rain.

### 24-hour rain of at least 25 mm

| System | Day 1 | Day 2 | Day 3 |
|---|---|---|---|
| **Pluvium (GFS-driven)** | +0.111 (+0.082) | +0.116 (+0.087) | +0.118 (+0.101) |
| ECMWF ENS | +0.118 (+0.111) | +0.112 (+0.094) | +0.115 (+0.102) |
| NOAA GEFS | +0.113 (+0.110) | +0.113 (+0.112) | +0.114 (+0.113) |
| ECMWF HRES | +0.117 (+0.070) | +0.110 (+0.068) | +0.112 (+0.057) |
| ECMWF AIFS | +0.116 (+0.112) | +0.113 (+0.109) | +0.114 (+0.112) |
| NOAA GraphCast | +0.115 (+0.112) | +0.114 (+0.109) | +0.114 (+0.112) |
| NOAA GFS | +0.113 (+0.112) | +0.109 (-0.197) | +0.117 (+0.112) |
| Blend of the 6 operational systems | +0.097 | +0.111 | +0.114 |
| Blend of the 6 + Pluvium | +0.073 | +0.107 | +0.116 |
| *A forecast that never says rain* | *+0.112* | *+0.109* | *+0.112* |

Share of cell-days with at least 25 mm on day 1: 0.5%. A system shows skill at this threshold only where it beats the forecast that never says rain.

### 24-hour rain of at least 50 mm

| System | Day 1 | Day 2 | Day 3 |
|---|---|---|---|
| **Pluvium (GFS-driven)** | +0.119 (+0.112) | +0.121 (+0.100) | +0.122 (+0.113) |
| ECMWF ENS | +0.119 (+0.119) | +0.121 (+0.120) | +0.122 (+0.121) |
| NOAA GEFS | +0.119 (+0.119) | +0.120 (+0.120) | +0.121 (+0.121) |
| ECMWF HRES | +0.119 (+0.119) | +0.121 (+0.120) | +0.121 (+0.121) |
| ECMWF AIFS | +0.119 (+0.119) | +0.121 (+0.120) | +0.122 (+0.121) |
| NOAA GraphCast | +0.119 (+0.119) | +0.121 (+0.120) | +0.122 (+0.121) |
| NOAA GFS | +0.119 (+0.119) | +0.121 (+0.120) | +0.122 (+0.121) |
| Blend of the 6 operational systems | +0.119 | +0.120 | +0.121 |
| Blend of the 6 + Pluvium | +0.119 | +0.120 | +0.121 |
| *A forecast that never says rain* | *+0.119* | *+0.120* | *+0.121* |

Share of cell-days with at least 50 mm on day 1: 0.1%. A system shows skill at this threshold only where it beats the forecast that never says rain.

### Pluvium against each system

Difference in Brier score, Pluvium minus the other system, after calibration (negative: Pluvium better), with a 95% moving-block bootstrap interval over days.

| Against | Day 1, 1 mm | Day 1, 10 mm | Day 2, 1 mm | Day 2, 10 mm | Day 3, 1 mm | Day 3, 10 mm |
|---|---|---|---|---|---|---|
| ECMWF ENS | **-0.0102** [-0.0169, -0.0047] | -0.0005 [-0.0024, +0.0009] | **-0.0128** [-0.0191, -0.0076] | -0.0003 [-0.0010, +0.0002] | **-0.0054** [-0.0106, -0.0001] | -0.0001 [-0.0006, +0.0004] |
| NOAA GEFS | -0.0082 [-0.0178, +0.0004] | +0.0001 [-0.0017, +0.0016] | -0.0039 [-0.0101, +0.0020] | -0.0005 [-0.0021, +0.0005] | -0.0014 [-0.0068, +0.0045] | -0.0000 [-0.0005, +0.0004] |
| ECMWF HRES | **-0.0139** [-0.0210, -0.0074] | -0.0001 [-0.0016, +0.0012] | **-0.0154** [-0.0211, -0.0114] | -0.0004 [-0.0010, +0.0001] | **-0.0076** [-0.0120, -0.0029] | -0.0003 [-0.0008, +0.0001] |
| ECMWF AIFS | **-0.0064** [-0.0136, -0.0004] | +0.0003 [-0.0004, +0.0011] | **-0.0071** [-0.0125, -0.0026] | +0.0001 [-0.0004, +0.0005] | +0.0016 [-0.0036, +0.0082] | +0.0000 [-0.0006, +0.0008] |
| NOAA GraphCast | **-0.0077** [-0.0131, -0.0030] | +0.0002 [-0.0011, +0.0016] | **-0.0078** [-0.0115, -0.0043] | -0.0005 [-0.0020, +0.0007] | -0.0004 [-0.0057, +0.0068] | -0.0003 [-0.0017, +0.0014] |
| NOAA GFS | -0.0057 [-0.0139, +0.0010] | +0.0006 [-0.0008, +0.0021] | -0.0039 [-0.0109, +0.0027] | -0.0013 [-0.0040, +0.0004] | +0.0008 [-0.0066, +0.0112] | -0.0008 [-0.0041, +0.0016] |
| Blend of the 6 operational systems | **-0.0052** [-0.0126, -0.0002] | -0.0002 [-0.0020, +0.0017] | -0.0033 [-0.0086, +0.0020] | -0.0013 [-0.0030, +0.0002] | +0.0013 [-0.0042, +0.0097] | -0.0009 [-0.0029, +0.0008] |
| Blend of the 6 + Pluvium | +0.0013 [-0.0015, +0.0039] | -0.0005 [-0.0031, +0.0018] | +0.0023 [-0.0011, +0.0064] | **-0.0013** [-0.0028, -0.0000] | +0.0042 [+0.0006, +0.0097] | -0.0013 [-0.0036, +0.0003] |

Bold: Pluvium better with 95% confidence. Differences at 25 and 50 mm are left out: there no system, Pluvium included, beats a forecast that never says rain, so the differences are noise.

### CRPS of the 24-hour totals (mm, lower is better; raw forecasts)

| System | Day 1 | Day 2 | Day 3 |
|---|---|---|---|
| Pluvium (GFS-driven) | 0.87 | 0.89 | 0.91 |
| ECMWF ENS | 1.45 | 1.67 | 1.68 |
| NOAA GEFS | 1.80 | 1.58 | 1.60 |
| ECMWF HRES | 1.94 | 2.41 | 2.36 |
| ECMWF AIFS | 1.76 | 1.92 | 1.94 |
| NOAA GraphCast | 1.83 | 1.76 | 1.73 |
| NOAA GFS | 2.42 | 2.52 | 2.56 |

### ROC area after calibration (day 1)

| System | 1 mm | 10 mm | 25 mm | 50 mm |
|---|---|---|---|---|
| Pluvium (GFS-driven) | 0.758 | 0.712 | 0.641 | 0.463 |
| ECMWF ENS | 0.669 | 0.742 | 0.589 | 0.425 |
| NOAA GEFS | 0.702 | 0.667 | 0.501 | 0.363 |
| ECMWF HRES | 0.624 | 0.585 | 0.566 | 0.435 |
| ECMWF AIFS | 0.688 | 0.570 | 0.532 | 0.389 |
| NOAA GraphCast | 0.701 | 0.693 | 0.533 | 0.438 |
| NOAA GFS | 0.725 | 0.536 | 0.514 | 0.424 |
| Blend of the 6 operational systems | 0.729 | 0.726 | 0.427 | 0.291 |
| Blend of the 6 + Pluvium | 0.767 | 0.766 | 0.453 | 0.291 |

## Results: May and June 2025 (the first protocol, isotonic calibration)

Test: 61 daily 00 UTC runs, 2025-05-01 to 2025-06-30, at 450 land cells of 8 km covering Jamaica. Truth: CMORPH. Every forecast is a real, archived operational run; none uses a reanalysis.

Brier skill score against the climatology of the same weeks (higher is better; 0 is climatology). Each system is calibrated the same causal way (isotonic regression on the 30 previous runs). The raw score is in brackets.

### 24-hour rain of at least 1 mm

| System | Day 1 | Day 2 | Day 3 |
|---|---|---|---|
| **Pluvium (GFS-driven)** | -0.032 (+0.080) | -0.034 (-0.009) | +0.021 (-0.196) |
| ECMWF ENS | +0.059 (-2.306) | +0.022 (-2.573) | -0.014 (-2.399) |
| NOAA GEFS | +0.071 (-3.042) | +0.067 (-2.514) | +0.041 (-2.191) |
| ECMWF HRES | +0.039 (-3.073) | -0.023 (-3.469) | -0.011 (-3.513) |
| ECMWF AIFS | +0.070 (-3.225) | +0.064 (-3.442) | +0.051 (-3.477) |
| NOAA GraphCast | +0.082 (-3.051) | +0.083 (-2.599) | +0.066 (-2.232) |
| NOAA GFS | +0.082 (-3.378) | +0.074 (-3.286) | +0.040 (-3.129) |
| Blend of the 6 operational systems | +0.087 | +0.094 | +0.068 |
| Blend of the 6 + Pluvium | +0.046 | +0.038 | +0.078 |
| *A forecast that never says rain* | *-0.212* | *-0.209* | *-0.204* |

Share of cell-days with at least 1 mm on day 1: 16.1%. A system shows skill at this threshold only where it beats the forecast that never says rain.

### 24-hour rain of at least 10 mm

| System | Day 1 | Day 2 | Day 3 |
|---|---|---|---|
| **Pluvium (GFS-driven)** | -0.039 (+0.049) | -0.022 (+0.009) | +0.013 (-0.674) |
| ECMWF ENS | +0.091 (+0.045) | +0.068 (-0.089) | +0.072 (-0.129) |
| NOAA GEFS | +0.085 (+0.033) | +0.108 (+0.095) | +0.109 (+0.046) |
| ECMWF HRES | +0.068 (-0.480) | +0.073 (-0.573) | +0.069 (-1.258) |
| ECMWF AIFS | +0.102 (-0.092) | +0.094 (+0.012) | +0.079 (-0.051) |
| NOAA GraphCast | +0.066 (-0.060) | +0.044 (+0.022) | +0.041 (+0.033) |
| NOAA GFS | +0.069 (-0.215) | +0.106 (-0.039) | +0.079 (-0.074) |
| Blend of the 6 operational systems | +0.070 | +0.126 | +0.100 |
| Blend of the 6 + Pluvium | +0.041 | +0.120 | +0.097 |
| *A forecast that never says rain* | *+0.055* | *+0.060* | *+0.072* |

Share of cell-days with at least 10 mm on day 1: 2.6%. A system shows skill at this threshold only where it beats the forecast that never says rain.

### 24-hour rain of at least 25 mm

| System | Day 1 | Day 2 | Day 3 |
|---|---|---|---|
| **Pluvium (GFS-driven)** | -0.098 (-0.006) | -0.095 (+0.014) | +0.018 (-1.167) |
| ECMWF ENS | +0.090 (+0.112) | +0.116 (+0.102) | +0.135 (+0.088) |
| NOAA GEFS | +0.111 (+0.110) | +0.130 (+0.120) | +0.142 (+0.135) |
| ECMWF HRES | +0.037 (-0.042) | +0.113 (+0.069) | +0.114 (+0.060) |
| ECMWF AIFS | +0.113 (+0.110) | +0.107 (+0.115) | +0.037 (+0.137) |
| NOAA GraphCast | +0.026 (+0.110) | +0.005 (+0.115) | +0.034 (+0.137) |
| NOAA GFS | +0.103 (+0.110) | +0.118 (+0.115) | +0.139 (+0.137) |
| Blend of the 6 operational systems | +0.115 | +0.133 | +0.139 |
| Blend of the 6 + Pluvium | +0.117 | +0.125 | +0.138 |
| *A forecast that never says rain* | *+0.110* | *+0.115* | *+0.137* |

Share of cell-days with at least 25 mm on day 1: 0.6%. A system shows skill at this threshold only where it beats the forecast that never says rain.

### 24-hour rain of at least 50 mm

| System | Day 1 | Day 2 | Day 3 |
|---|---|---|---|
| **Pluvium (GFS-driven)** | -0.282 (+0.072) | -0.239 (+0.099) | -0.031 (-2.251) |
| ECMWF ENS | +0.122 (+0.205) | +0.206 (+0.215) | +0.238 (+0.235) |
| NOAA GEFS | +0.207 (+0.206) | +0.217 (+0.217) | +0.110 (+0.240) |
| ECMWF HRES | +0.014 (+0.206) | +0.206 (+0.217) | +0.194 (+0.240) |
| ECMWF AIFS | +0.188 (+0.206) | +0.198 (+0.217) | -0.059 (+0.240) |
| NOAA GraphCast | +0.004 (+0.206) | -0.039 (+0.217) | +0.033 (+0.240) |
| NOAA GFS | +0.199 (+0.206) | +0.206 (+0.217) | +0.240 (+0.240) |
| Blend of the 6 operational systems | +0.206 | +0.217 | +0.240 |
| Blend of the 6 + Pluvium | +0.207 | +0.217 | +0.240 |
| *A forecast that never says rain* | *+0.206* | *+0.217* | *+0.240* |

Share of cell-days with at least 50 mm on day 1: 0.1%. A system shows skill at this threshold only where it beats the forecast that never says rain.

### Pluvium against each system

Difference in Brier score, Pluvium minus the other system, after calibration (negative: Pluvium better), with a 95% moving-block bootstrap interval over days.

| Against | Day 1, 1 mm | Day 1, 10 mm | Day 2, 1 mm | Day 2, 10 mm | Day 3, 1 mm | Day 3, 10 mm |
|---|---|---|---|---|---|---|
| ECMWF ENS | +0.0121 [-0.0085, +0.0389] | +0.0035 [+0.0002, +0.0063] | +0.0074 [-0.0099, +0.0294] | +0.0024 [-0.0001, +0.0046] | -0.0047 [-0.0125, +0.0043] | +0.0015 [-0.0005, +0.0044] |
| NOAA GEFS | +0.0136 [-0.0050, +0.0424] | +0.0034 [+0.0001, +0.0060] | +0.0134 [-0.0031, +0.0349] | +0.0034 [+0.0005, +0.0057] | +0.0026 [-0.0056, +0.0108] | +0.0024 [+0.0002, +0.0051] |
| ECMWF HRES | +0.0095 [-0.0109, +0.0353] | +0.0029 [-0.0004, +0.0056] | +0.0014 [-0.0145, +0.0247] | +0.0025 [+0.0004, +0.0046] | -0.0043 [-0.0148, +0.0044] | +0.0014 [-0.0004, +0.0041] |
| ECMWF AIFS | +0.0136 [-0.0072, +0.0401] | +0.0038 [+0.0002, +0.0071] | +0.0128 [-0.0028, +0.0348] | +0.0031 [+0.0010, +0.0052] | +0.0041 [-0.0050, +0.0120] | +0.0017 [+0.0001, +0.0045] |
| NOAA GraphCast | +0.0151 [-0.0047, +0.0420] | +0.0028 [-0.0004, +0.0052] | +0.0154 [-0.0000, +0.0381] | +0.0017 [-0.0005, +0.0037] | +0.0060 [-0.0037, +0.0163] | +0.0007 [-0.0013, +0.0038] |
| NOAA GFS | +0.0151 [-0.0020, +0.0423] | +0.0029 [-0.0003, +0.0054] | +0.0142 [-0.0016, +0.0371] | +0.0034 [+0.0004, +0.0057] | +0.0025 [-0.0062, +0.0113] | +0.0017 [-0.0000, +0.0043] |
| Blend of the 6 operational systems | +0.0158 [-0.0014, +0.0430] | +0.0030 [+0.0002, +0.0053] | +0.0169 [+0.0005, +0.0400] | +0.0039 [+0.0011, +0.0061] | +0.0063 [-0.0034, +0.0164] | +0.0022 [+0.0004, +0.0051] |
| Blend of the 6 + Pluvium | +0.0104 [+0.0015, +0.0247] | +0.0022 [+0.0002, +0.0045] | +0.0094 [+0.0021, +0.0170] | +0.0038 [+0.0011, +0.0058] | +0.0075 [-0.0013, +0.0177] | +0.0021 [+0.0004, +0.0050] |

Bold: Pluvium better with 95% confidence. Differences at 25 and 50 mm are left out: there no system, Pluvium included, beats a forecast that never says rain, so the differences are noise.

### CRPS of the 24-hour totals (mm, lower is better; raw forecasts)

| System | Day 1 | Day 2 | Day 3 |
|---|---|---|---|
| Pluvium (GFS-driven) | 0.94 | 0.98 | 1.47 |
| ECMWF ENS | 1.82 | 1.96 | 1.87 |
| NOAA GEFS | 2.12 | 1.84 | 1.68 |
| ECMWF HRES | 2.46 | 2.73 | 2.88 |
| ECMWF AIFS | 2.20 | 2.23 | 2.13 |
| NOAA GraphCast | 2.09 | 1.90 | 1.74 |
| NOAA GFS | 2.45 | 2.45 | 2.31 |

### ROC area after calibration (day 1)

| System | 1 mm | 10 mm | 25 mm | 50 mm |
|---|---|---|---|---|
| Pluvium (GFS-driven) | 0.739 | 0.782 | 0.800 | 0.565 |
| ECMWF ENS | 0.708 | 0.772 | 0.752 | 0.650 |
| NOAA GEFS | 0.732 | 0.696 | 0.567 | 0.650 |
| ECMWF HRES | 0.679 | 0.592 | 0.630 | 0.581 |
| ECMWF AIFS | 0.695 | 0.723 | 0.773 | 0.663 |
| NOAA GraphCast | 0.721 | 0.784 | 0.805 | 0.661 |
| NOAA GFS | 0.750 | 0.723 | 0.753 | 0.613 |
| Blend of the 6 operational systems | 0.758 | 0.772 | 0.627 | 0.494 |
| Blend of the 6 + Pluvium | 0.771 | 0.791 | 0.641 | 0.515 |

## How it was run

* **Pluvium**: pluvium-small (version 3, 935,838 parameters). Stage A: 32 minutes on ERA5 for June to September 2021 (the version benchmark's run). Stage B: 900 steps (80 minutes on a 4-core CPU) on 324 real 00 UTC GFS runs from 2024-05-01 to 2025-03-20, all ending before the first calibration day. At issue time it reads that morning's GFS run (0.5 degrees), the satellite rain of the previous 3 hours, the airports' reports of the previous 3 hours, terrain, and a CMORPH climatology of the 12 months before calibration starts.
* **Baselines**: each system's own 00 UTC run, read from its producer's archive, interpolated bilinearly to the cell centres. Ensembles give the share of members at or above each threshold.
* **Calibration** (the same for all, Pluvium included): for each test run, a logistic regression (May-June first used isotonic regression) per day and threshold fitted on the 30 previous runs whose outcomes were known 3 days before. Each system uses whichever predictor fits its window best: its point chance, its chance over everything within 0.5 degrees, or its mean amount.
* **Blends**: logistic regression on the calibrated chances, fitted on the same window.
* **Truth**: CMORPH 8 km rain (satellite estimates adjusted to rain gauges) summed over 00-24 UTC.
* **Reference**: the CMORPH climatology of 2017-2024: days within 15 days of the same date, cells within 25 km.
* **Intervals**: moving-block bootstrap over test days (blocks of 5 days, 2,000 resamples).

## Limits

* One island and two months per test. Skill elsewhere or in other seasons is untested.
* Truth is CMORPH, which Pluvium was also trained on. Calibration on the same truth puts every system on the same footing, but the raw scores favour Pluvium.
* Pluvium's "rain now" input comes from the CMORPH record, which is adjusted to gauges after the fact. Live, Pluvium reads GOES-East and radar instead, which are cruder.
* Pluvium ran at its small size, trained for about two hours on a CPU in total. Pluvium-500 is 60 times larger and is meant to train for a day on a GPU.
