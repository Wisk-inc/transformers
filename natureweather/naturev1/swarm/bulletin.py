# Copyright 2026 Nathan. Apache-2.0.
"""
The operations desk's record: one bulletin per storm per cycle (JSON and CSV), and the previous cycle found again.

A forecast that is not written down did not happen. :func:`write_bulletin` saves what a cycle issued -- the track,
intensity, probabilities, the landfall summary, every parish's numbers -- under ``runs/<storm>/<time>.json`` with the
parish table beside it as CSV; :func:`load_previous` finds the newest earlier cycle of the same storm so its error can
feed back into the next forecast (:func:`naturev1.cyclone.consensus.apply_track_feedback`), and :func:`history`
lists the cycles issued so far, which is how a forecast's drift from cycle to cycle is seen.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PARISH_COLUMNS = ("label", "p34", "p64", "p_gust64", "peak_kt", "peak_kt_if_hit", "peak_kt_p90", "peak_gust_kt", "gust_kt_if_hit",
                  "rain_mm", "rain_mm_if_hit", "rain_mm_p90", "rain_max_mm_if_hit", "p_rain100", "p_rain250", "p_flash",
                  "arrival_time", "hu_arrival_time", "peak_time")


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.datetime64,)):
        return str(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _stamp(t) -> str:
    return str(np.datetime64(t, "m")).replace(":", "")


def write_bulletin(run: dict, summary: dict | None, directory: str | Path, storm_id: str | None = None) -> dict:
    """
    Save a cycle. ``run`` is :func:`naturev1.swarm.live.forecast_storm`'s dict (``case``, ``result``, ``impact``,
    ``analysis_time``), ``summary`` the :func:`naturev1.swarm.landfall.landfall_summary` for the country of interest
    (optional). Returns ``{"json": path, "csv": path, "record": the saved dict}``.
    """
    case, result = run["case"], run["result"]
    sid = (storm_id or getattr(case, "storm_id", "") or "storm").lower()
    when = np.datetime64(run.get("analysis_time", case.time), "m")
    folder = Path(directory).expanduser() / sid
    folder.mkdir(parents=True, exist_ok=True)
    keys = ("leads", "track", "intensity", "cone_km", "cone_fixed_km", "ri_probability", "intensity_regime", "intensity_if_ri",
            "peak_probability", "category_probability", "consensus_table", "n_models", "model_spread_km", "model_deepening_hpa")
    record = {"storm_id": sid, "name": getattr(case, "name", ""), "time": str(when), "lat": float(case.lat), "lon": float(case.lon),
              "wind_kt": float(case.wind_kt), "issued_by": f"naturev1 {_version()}", "forecast": {k: _plain(result[k]) for k in keys if k in result},
              "models": _plain({k: v.get("title", k) if isinstance(v, dict) else str(v) for k, v in (run.get("models") or {}).items()}),
              "r34_km": _plain(run.get("r34_km")), "landfall": _plain(summary) if summary else None,
              "scenarios": _plain({k: v.get("probability") for k, v in (result.get("swarm", {}).get("scenarios") or {}).items()}) if result.get("swarm") else None}
    impact = run.get("impact")
    rows = []
    if impact is not None:
        for u in impact.units:
            rows.append({k: _plain(u.get(k)) for k in PARISH_COLUMNS if k in u})
        if summary and summary.get("units"):
            timed = {r["label"]: r for r in summary["units"]}
            for r in rows:
                for k in ("arrival_time", "hu_arrival_time", "peak_time"):
                    if r["label"] in timed and k in timed[r["label"]]:
                        r[k] = timed[r["label"]][k]
    record["parishes"] = rows
    jpath = folder / f"{_stamp(when)}.json"
    jpath.write_text(json.dumps(record, indent=1))
    cpath = folder / f"{_stamp(when)}_parishes.csv"
    if rows:
        cols = [c for c in PARISH_COLUMNS if any(c in r for r in rows)]
        with open(cpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    return {"json": jpath, "csv": cpath if rows else None, "record": record}


def history(directory: str | Path, storm_id: str) -> list[dict]:
    """Every saved cycle of a storm, oldest first (the records' ``forecast`` and ``landfall`` parts included)."""
    folder = Path(directory).expanduser() / storm_id.lower()
    out = []
    for p in sorted(folder.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


def load_previous(directory: str | Path, storm_id: str, before, max_gap_h: float = 48.0) -> dict | None:
    """
    The newest saved cycle of ``storm_id`` older than ``before`` (and not older than ``max_gap_h``), as the
    ``previous=`` argument of :func:`naturev1.cyclone.forecast.physics_forecast`: ``{"time", "track"}``.
    """
    t = np.datetime64(before, "m")
    best = None
    for rec in history(directory, storm_id):
        when = np.datetime64(rec["time"], "m")
        gap = float((t - when) / np.timedelta64(60, "m"))
        if 0.0 < gap <= max_gap_h and rec.get("forecast", {}).get("track"):
            if best is None or when > best[0]:
                best = (when, rec)
    if best is None:
        return None
    track = np.asarray([[np.nan if v is None else v for v in row] for row in best[1]["forecast"]["track"]], np.float64)
    return {"time": best[0], "track": track, "lat": best[1].get("lat"), "lon": best[1].get("lon"), "leads": best[1]["forecast"].get("leads"), "record": best[1]}


def _version() -> str:
    try:
        from .. import __version__

        return __version__
    except Exception:
        return "?"


__all__ = ["PARISH_COLUMNS", "history", "load_previous", "write_bulletin"]
