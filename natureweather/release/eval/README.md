# Evaluation scripts for NatureV1

These produced every number in section 4 of the model card. Each takes one argument, a working directory `S`
laid out as:

```
S/hub/checkpoints/stage2/latest.pt     from this repository
S/hub/checkpoints/storm_bank_val.pt    from this repository
S/hub/staging/stats.json               from this repository
S/hurdat/hurdat2.txt                   https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt
S/hurdat/ibtracs_<BASIN>.csv           IBTrACS v04r01 per-basin CSV (for ood_run.py)
S/hub/checkpoints/storm_bank_train.pt  from this repository (for consistency.py)
```

| Script | What it does |
|---|---|
| `test_storms.py` | the held-out storm scorecard (2017/2019/2021) from the cached features |
| `per_storm.py` | the same, storm by storm, with predicted-versus-actual tables for single forecasts |
| `other_heads.py` | landfall, rapid intensification, peak wind, wind radii, radius of maximum wind |
| `e2e_maria.py`, `e2e_compare.py` | Maria 2017 through the full model from raw ERA5, compared with the cached features |
| `arco15.py` | rebuilds WeatherBench 2's 1.5° six-hourly ERA5 from ARCO-ERA5 at 0.25° (conservative regridding) |
| `validate15.py` | checks that rebuild against the WeatherBench 2 store on a date both cover |
| `elnino2023.py STORM,STORM,...` | runs the full model on 2023 storms from rebuilt ERA5, one bank per storm |
| `score2023.py` | scores those banks and compares them with the held-out seasons |
| `stress.py` | red team: impossible and corrupted inputs through the heads and `check_forecast` |
| `calib.py` | red team: calibration of track, wind, landfall and RI; intensity bias by category |
| `ablate.py` | red team: remove one input at a time and measure what the heads rely on |
| `consistency.py` | red team: data-leak audit and logical consistency of the outputs |
| `ood_run.py BASIN:NAME:YEAR ...` | red team: the full model on storms outside the Atlantic (IBTrACS tracks) |
| `score_ood.py` | scores those runs, with Otis 2023 forecast by forecast |

Needs `pip install naturev1 xarray zarr gcsfs`. On a CPU a full-model forecast takes about a minute.
