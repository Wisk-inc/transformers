# Copyright 2026 Nathan. Apache-2.0.
"""The model card written next to a trained Pluvium: what it is, how it was trained, and how well it did."""

from __future__ import annotations

from .verify import WINDOWS


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:+.1%}"


def _num(x, digits=4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def scores_table(scores: dict) -> str:
    names = {"reanalysis_driven": "Reanalysis-driven (upper bound)", "emulated_errors": "Emulated forecast errors",
             "gfs_driven": "Driven by real GFS runs (live situation)"}
    rows = ["| Test | Hourly Brier | Climatology | Skill vs climatology | Any rain in 24 h: skill | Start hour within 3 h | "
            "24 h total CRPS skill | Intensity (RPS) skill | Heavy hours skill |", "|---|---|---|---|---|---|---|---|---|"]
    for key, label in names.items():
        r = scores.get(key)
        if not r:
            continue
        a24 = r.get("any_rain_24h", {})
        start = r.get("start_hour_error", {})
        within = "n/a" if not start else "{:.0%}".format(start["within_3h"])
        heavy = r.get("heavy", {})
        rows.append(f"| {label} | {_num(r.get('brier_p'))} | {_num(r.get('brier_clim'))} | {_pct(r.get('bss_vs_clim'))} "
                    f"| {_pct(a24.get('bss_vs_clim'))} | {within} | {_pct(r.get('crpss_24h_vs_clim'))} | "
                    f"{_pct(r.get('rpss_vs_clim'))} | {_pct(heavy.get('bss_vs_clim'))} |")
    return "\n".join(rows) if len(rows) > 2 else "_No scores were recorded for this release._"


def model_card(repo: str, model, scores: dict, years, holdout: int, stage_b: bool) -> str:
    c = model.config
    params = model.num_parameters() / 1e6
    train = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])
    return f"""---
license: apache-2.0
library_name: pluvium
tags:
- weather
- precipitation
- nowcasting
- caribbean
- probabilistic-forecasting
- ihelix
---

# {repo.split('/')[-1]}

Hour-by-hour rain forecasts for every town and village in the Caribbean, three days ahead: the chance of rain in each
hour, when rain is most likely to start (in local time) and how long it lasts, how much may fall, and what is causing
it (cold front or shear line, tropical wave, tropical cyclone, afternoon storms, trade-wind showers).

**This is a research model, not an official forecast.** Follow your national meteorological service.

* {params:.1f}M parameters, the NatureV1 / iHELIX engine: geodesic attention on a variable-resolution Caribbean mesh
  ({len(model.geo.mesh):,} points, about 11 km over land), index attention, and a gated delta-rule recurrence along the
  forecast's frames ({c.layers} blocks of width {c.hidden}).
* Inputs: driving-model fields (ERA5 in training; GFS, ECMWF IFS and ECMWF AIFS live) with convective indices, rain
  already falling (CMORPH in training, GOES-East live), the last three hours of airport reports, terrain, coasts,
  sun and season.
* Truth: CMORPH 8 km satellite rain (half-hourly, NOAA CDR) and hourly airport weather reports (NOAA ISD, read for
  intensity, drizzle/rain/showers and thunder).
* Outputs per town and hour: chance of dry/light/moderate/heavy rain, thunder, drizzle/rain/showers, expected rain;
  start, heaviest and stop times; totals and the wettest 1, 3 and 6 hours; flood risk; the cause.

## Experimental components

These have not been tested in published work; each can be switched off for an ablation, and `pluvium bench`
compares the full version 2 engine with version 1 on held-out data.

1. **An intensity chain.** For every hour and place the network writes a transition matrix between dry, light,
   moderate and heavy rain. Hourly chances, start and stop times, the hour rain turns heavy, the chance of any rain
   in a window and the totals all come from that one chain, so they agree with each other.
2. **Truth seen through its instrument.** Training scores the observed classes through a hidden Markov model in
   which the satellite and the airports each have their own learned confusion matrix.
3. **Thunder and drizzle/rain/showers heads**, trained on what airports report (METAR groups such as ``-SHRA``,
   ``+TSRA``, ``DZ``).
4. **Airport assimilation.** The last three hours of reports are read into the mesh by cross-attention, with
   dropout so the model works with or without them.
5. **Advected rain.** The rain falling now is carried along the steering wind for the first 12 hours (the idea of
   radar nowcasting, with satellite rain).
6. **Convective indices** among the inputs: K-index, total totals, theta-e difference, lifted index, 700 hPa
   humidity, moisture-flux convergence, 850-200 hPa shear, 200 hPa divergence.
7. **A differentiable orographic layer.** Smith-Barstad linear theory of rain over mountains, solved by FFT on 2 km
   tiles every forecast hour, with learned parameters.
8. **Island-heating convergence**, **front and shear-line arrival times**, **per-place daily cycles**, and
   **forecast-error emulation** during reanalysis training.
9. **A mixture of chains** live, one per driving model, so disagreement widens the windows.
10. **Online corrections.** Forecasts are checked against later airport reports and recalibrated per parish.

## Training

* Stage A: ERA5 years {train}, perfect prognosis with emulated forecast errors.
* Stage B: {'model output statistics on real GFS runs.' if stage_b else 'not run for this release (no fine-tuning on real forecast runs).'}
* Calibration: isotonic, per lead-time band, fitted on half of {holdout}; scores below are on the other half.

## Scores on held-out data ({holdout})

{scores_table(scores)}

* Brier score: mean squared error of the hourly chance of rain (lower is better). Skill: improvement over the
  climatological chance for that place, month and hour of day.
* "Reanalysis-driven" uses the reanalysis as if it were a perfect forecast, so it is an upper bound on live skill.
  "Emulated forecast errors" degrades it the way forecasts are degraded; "driven by real GFS runs" is the live
  situation (when stage B ran).
* A wet hour is at least 0.2 mm. Rain-start error is for places dry at the start of the forecast where rain began
  within 24 hours.

## Limitations

* Satellite rain (CMORPH) misses some light showers and is smoothed at 8 km; the model's truth inherits that.
* Live, the rain already falling comes from GOES-East infrared when its rain-rate product is unavailable; that
  estimate is cruder than the CMORPH used in training.
* Tropical cyclones are rare in any training period; rain in landfalling storms is less reliable than everyday rain.
* Towns are forecast at their coordinates; a town spread over a hillside can see very different rain at each end.

## Use

```bash
pip install pluvium
pluvium forecast --repo {repo}          # every town, tables in ./pluvium-forecast
pluvium place "Mandeville" --country JM  # hour by hour for one place
```

## Data and attribution

ERA5 (Copernicus Climate Change Service), CMORPH CDR and ISD (NOAA NCEI), GFS (NOAA NCEP), ECMWF open data
(CC-BY-4.0), GOES-East (NOAA), GeoNames (CC-BY-4.0), geoBoundaries (CC-BY-4.0), Terrarium elevation tiles (Mapzen
and its sources). Code: Apache-2.0, CorX Labs.
"""


__all__ = ["model_card", "scores_table", "WINDOWS"]
