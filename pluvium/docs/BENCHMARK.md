# Benchmark: Pluvium versions 1, 2 and 3

Real data, Jamaica: 560 CMORPH 8 km cells over and around the island, 17 airports; a 358-point mesh (about 22 km over land). Trained from scratch on issue times from 1 June to 23 September 2021 (ERA5 driving fields with forecast-error emulation); calibrated on half of the issue times from 1 October to 3 November 2021 (every 6 hours, alternating) and scored on the other half. Same seed, same data, same wall-clock budget (32 minutes) each, on a 4-core CPU, so the heavier version 3 takes fewer steps. Size: pluvium-small at each version (parameters and steps at the end).

**Reading.** With emulated forecast errors, version 3 is best or tied on 7 of the 11 scores. It nearly doubles
the skill of 24-hour totals (CRPS skill +44.6% against +23.8% for version 2), and it has the best hourly skill, the
best critical success index and the best 24-hour rain chances. It is worse than version 2 at "any rain in the next
6 hours" and at thunder. Heavy-rain hours remain below climatology for versions 2 and 3. Version 3 did this with 392
training steps against 592 for version 2, because its deeper engine costs more per step.

The operational benchmark against ECMWF, NOAA and AI forecasts is in [OPERATIONAL_BENCHMARK.md](OPERATIONAL_BENCHMARK.md).

## Driving fields with emulated forecast errors (closer to live)

| Score | Version 1 | Version 2 | Version 3 |
|---|---|---|---|
| hourly rain chance, skill vs climatology | +13.1% | +16.0% | +16.4% |
| hourly rain chance, skill vs the driving model's rain | +71.9% | +72.8% | +73.0% |
| yes/no at 30%: critical success index | 0.157 | 0.200 | 0.214 |
| any rain in 6 h, skill vs climatology | +31.4% | +33.2% | +31.1% |
| any rain in 24 h, skill vs climatology | +48.4% | +49.3% | +50.5% |
| rain start within 2 h of the truth | 0.454 | 0.464 | 0.463 |
| rain start, median error (h) | 3.0 h | 3.0 h | 3.0 h |
| intensity classes, skill vs climatology | +11.3% | +14.2% | +14.2% |
| heavy hours, skill vs climatology | +0.6% | -3.2% | -4.5% |
| 24 h totals, CRPS skill vs climatology | +15.2% | +23.8% | +44.6% |
| thunder at airports, skill vs climatology | n/a | +2.6% | +0.6% |

References on the same hours: climatology Brier 0.0683, persistence 0.1235, driving model's rain 0.2112; 2,692,800 target-hours, 7.4% wet.

## Reanalysis as a perfect forecast (upper bound on live skill)

| Score | Version 1 | Version 2 | Version 3 |
|---|---|---|---|
| hourly rain chance, skill vs climatology | +16.5% | +18.2% | +17.7% |
| hourly rain chance, skill vs the driving model's rain | +72.2% | +72.8% | +72.7% |
| yes/no at 30%: critical success index | 0.204 | 0.231 | 0.240 |
| any rain in 6 h, skill vs climatology | +31.9% | +33.1% | +31.3% |
| any rain in 24 h, skill vs climatology | +48.9% | +49.7% | +50.8% |
| rain start within 2 h of the truth | 0.461 | 0.478 | 0.468 |
| rain start, median error (h) | 3.0 h | 3.0 h | 3.0 h |
| intensity classes, skill vs climatology | +14.4% | +16.2% | +15.2% |
| heavy hours, skill vs climatology | +0.9% | -4.4% | -5.5% |
| 24 h totals, CRPS skill vs climatology | +14.3% | +20.8% | +43.7% |
| thunder at airports, skill vs climatology | n/a | +2.6% | -0.3% |

References on the same hours: climatology Brier 0.0683, persistence 0.1235, driving model's rain 0.2055; 2,692,800 target-hours, 7.4% wet.

Training: version 1: 549,202 parameters, 672 steps in 32 min; version 2: 587,177 parameters, 592 steps in 32 min; version 3: 935,838 parameters, 392 steps in 32 min
