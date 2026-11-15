# Sharp scenarios from smooth chances: ensemble copula coupling for multi-model rain forecasts

Pluvium technical paper 3. Nathan, November 2026.

## Summary

A calibrated chance of rain at each place makes a smooth map, and the map gets smoother the further ahead it looks, because it averages over every place a storm might be. For any one place that is the correct forecast. As a picture of a day it misleads, since no real day looks like the average of many. The first benchmark counted this as a weak spot: the sharpness of the expected-amount map fell with lead.

We leave that map alone and add scenarios. Each scenario takes its amounts, place by place, from Pluvium Blend's calibrated distribution and its arrangement in space from one member of the ECMWF ensemble (ensemble copula coupling; Schefzik, Thorarinsdottir and Gneiting, 2013), with ties among dry members broken by rain nearby.

Over 7 places, a coupled scenario on days 11-14 has 0.88 of the weather's spatial variability (a raw ENS member: 0.84). The CRPS of rain averaged over two-degree boxes is 2.26 mm for coupled scenarios, 2.93 mm for the same amounts with no arrangement and 2.57 mm for the raw ensemble. On days 1-3 the three numbers are 1.47, 1.69 and 2.16. The variogram score, which rewards the right differences between nearby places, agrees at every lead.

The rain test year (April 2025 to March 2026) is the year on which the weak spots were found. Every design choice was made inside the training year and written down before the test year was scored, but a second look at a year is weaker evidence than a fresh year. The scenarios have no settings fitted to that year.

## 1. The problem

Sharpness here is the spatial standard deviation of a map over that of the observed map. For the Blend's expected amount and for the mean of the raw ECMWF ensemble, averaged over the places:

| Map | days 1-3 | days 4-7 | days 8-10 | days 11-14 |
|---|---|---|---|---|
| Pluvium Blend, expected amount | 0.63 | 0.47 | 0.33 | 0.28 |
| Raw ECMWF ENS, ensemble mean | 0.73 | 0.58 | 0.45 | 0.38 |

Both fall with lead, as they must. At day 12 nobody knows where the rain band will lie, so every place gets a modest chance, and sharpening the map of expected amounts would make it wrong. At that range a user is better served by a set of days that could happen, each one looking like weather, which together carry the calibrated chances. A hydrologist needs such a set for rain over a basin, where the total depends on whether nearby places are wet together.

## 2. Method

### 2.1 Coupling

For one valid day, take the ensemble's `M` members at the places.

1. At every place, rank the members from driest to wettest.
2. Give the member ranked `r` the calibrated distribution's quantile at level `(r + 0.5) / M`.

Every scenario keeps its member's pattern: where that member was wet compared with the others, the scenario is wet. At every place the scenarios are an even sample of the calibrated distribution, so every score computed place by place is unchanged.

### 2.2 Ties

Rain needs one refinement. At most places most members are dry, so their ranks are ties. Broken at random, the ties scatter drizzle wherever calibration says rain is more likely than the ensemble did. Here a tie is broken by the member's rain in the neighbourhood: among dry members, the one with rain nearby is the next to be wet. Remaining ties are broken at random.

### 2.3 Without an ensemble

Where no physical ensemble is at hand, the arrangement can come from past observed days instead (the Schaake shuffle; Clark et al., 2004). `pluvium.scenarios.schaake` does that. It is not scored here.

### 2.4 Scores

Three sets of scenarios with the same number of members are compared: the raw ensemble, coupled scenarios, and uncoupled ones (the same amounts, each place drawn on its own).

The sharpness of one scenario is its spatial standard deviation over the observed one, so that 1 is as varied as the weather. The CRPS is computed place by place, where coupled and uncoupled scenarios are equal by construction. The areal CRPS is the CRPS of the mean rain over boxes two degrees wide, which only scenarios with the right structure get right. The energy score (Gneiting et al., 2008) and the variogram score (Scheuerer and Hamill, 2015) reward the right joint behaviour of nearby places.

## 3. Results

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

Coupled scenarios have the lowest areal CRPS in 28 of 28 place and lead-group cells. Against the raw ensemble the gain comes from calibration. Against uncoupled draws it comes from structure alone, since the amounts are the same and only their arrangement differs. That gap widens with lead, because independent draws average out over a box and real rain does not.

A single uncoupled draw is as sharp as a coupled one, or sharper. Sharpness by itself therefore says nothing about structure, which is what the areal and variogram scores measure.

## 4. Limits

- The arrangement is the ECMWF ensemble's. Where that ensemble puts the rain band in the wrong place, so does every scenario: coupling corrects the amounts and leaves the positions as the ensemble has them.
- There are as many scenarios as the ensemble has members.
- Only daily totals are coupled, and each day on its own. A scenario is not yet one story across the 14 days.
- Everything is scored against one satellite estimate, on quarter-degree cells.
- The test year is the one on which the weak spots were found (see the summary).

## 5. Reproducing

```python
from pluvium.scenarios.ecc import ecc, neighbourhood_tiebreak
near = neighbourhood_tiebreak(fields, lat, lon, place_lat, place_lon)    # each member's rain nearby
scenarios = ecc(chances, members, tiebreak=near)    # chances (places, thresholds), members (M, places)
```

The benchmark is `pluvium.benchmarks.scenario_bench`, run by `python -m pluvium.benchmarks.frontier_bench`.

## References

- Clark, M., Gangopadhyay, S., Hay, L., Rajagopalan, B., and Wilby, R. (2004). The Schaake shuffle: a method for reconstructing space-time variability in forecasted precipitation and temperature fields. *Journal of Hydrometeorology*, 5, 243-262. doi:10.1175/1525-7541(2004)005<0243:TSSAMF>2.0.CO;2
- Gneiting, T., Stanberry, L. I., Grimit, E. P., Held, L., and Johnson, N. A. (2008). Assessing probabilistic forecasts of multivariate quantities, with an application to ensemble predictions of surface winds. *TEST*, 17, 211-235. doi:10.1007/s11749-008-0114-x
- Schefzik, R., Thorarinsdottir, T. L., and Gneiting, T. (2013). Uncertainty quantification in complex simulation models using ensemble copula coupling. *Statistical Science*, 28, 616-640. doi:10.1214/13-STS443
- Scheuerer, M., and Hamill, T. M. (2015). Variogram-based proper scoring rules for probabilistic forecasts of multivariate quantities. *Monthly Weather Review*, 143, 1321-1334. doi:10.1175/MWR-D-14-00269.1
