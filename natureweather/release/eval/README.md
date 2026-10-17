# Evaluation scripts for NatureV1

These produced every number in section 4 of the model card. Each takes one argument, a working directory `S`
laid out as:

```
S/hub/checkpoints/stage2/latest.pt     from this repository
S/hub/checkpoints/storm_bank_val.pt    from this repository
S/hub/staging/stats.json               from this repository
S/hurdat/hurdat2.txt                   https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt
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

Needs `pip install naturev1 xarray zarr gcsfs`. On a CPU a full-model forecast takes about a minute.
