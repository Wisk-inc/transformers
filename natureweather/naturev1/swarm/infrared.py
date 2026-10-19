# Copyright 2026 Nathan. Apache-2.0.
"""
The satellite, read quantitatively: infrared predictors from a GOES band-13 scene around the centre.

These are the convective-structure quantities the operational rapid-intensification guidance uses (Kaplan, DeMaria &
Knaff 2010, the SHIPS-RII predictors: the share of the 50-200 km annulus colder than -30 C, the standard deviation of
brightness temperature 100-300 km) plus the coldest cloud top near the core, the symmetry of the eyewall ring and, with
an earlier scene, their trend -- a convective burst shows as cloud tops cooling and the cold share growing over a few
hours; an eye shows as a warm spot inside a cold ring.

They are **diagnostics** here: shown beside the forecast, not yet inside the rapid-intensification probability, because
no archive of storm-centred infrared was available to fit and test them on the held-out seasons. The function is
written so that the day such an archive is in hand the same numbers can be fitted.
"""

from __future__ import annotations

import numpy as np

from ..cyclone.geometry import great_circle_km

K0 = 273.15


def _wmean(v, w):
    w = np.where(np.isfinite(v), w, 0.0)
    return float(np.sum(np.nan_to_num(v) * w) / w.sum()) if w.sum() > 0 else float("nan")


def ir_metrics(scene, lat: float, lon: float, previous=None, previous_hours: float | None = None) -> dict:
    """
    Args:
        scene: a :class:`naturev1.satellite.SatelliteScene` with band 13 brightness temperature (K) in column 0.
        lat, lon: the centre (the latest fix or the analysis position).
        previous: an earlier scene around the same storm (optional), ``previous_hours`` hours older.

    Returns a dict: ``min_bt_c`` (coldest top within 100 km), ``pc30`` and ``pc50`` (area shares of the 50-200 km
    annulus colder than -30 / -50 C), ``sdo_k`` (std of BT 100-300 km), ``mean_bt_c`` (50-200 km), ``ring_asymmetry_k``
    (std over octants of the 30-100 km mean BT), ``eye_warm_k`` (how much warmer the inner 25 km is than the 40-80 km
    ring; an eye when clearly positive), ``samples``; with a previous scene also ``d_pc50_per_h``, ``d_min_bt_per_h``;
    and ``text``, one line for a report.
    """
    la, lo = np.asarray(scene.latitude, np.float64), np.asarray(scene.longitude, np.float64)
    bt = np.asarray(scene.values[:, 0], np.float64)
    area = np.asarray(getattr(scene, "area_km2", np.ones_like(bt)), np.float64)
    ok = np.isfinite(bt) & np.isfinite(la) & np.isfinite(lo)
    la, lo, bt, area = la[ok], lo[ok], bt[ok], area[ok]
    d = great_circle_km(la, lo, lat, lon)
    out = {"samples": int(bt.size), "time": getattr(scene, "timestamp", None), "satellite": getattr(scene, "satellite", "")}
    if bt.size < 50 or not (d <= 200).any():
        out["text"] = "GOES IR: no usable scene around the centre"
        return out
    inner, ann, outer = d <= 100.0, (d >= 50.0) & (d <= 200.0), (d >= 100.0) & (d <= 300.0)
    out["min_bt_c"] = float(np.nanmin(bt[inner])) - K0 if inner.any() else float("nan")
    out["pc30"] = _wmean((bt[ann] < K0 - 30.0).astype(float), area[ann]) if ann.any() else float("nan")
    out["pc50"] = _wmean((bt[ann] < K0 - 50.0).astype(float), area[ann]) if ann.any() else float("nan")
    out["mean_bt_c"] = _wmean(bt[ann], area[ann]) - K0 if ann.any() else float("nan")
    out["sdo_k"] = float(np.nanstd(bt[outer])) if outer.sum() > 10 else float("nan")
    # eyewall ring symmetry: the mean brightness temperature of the 30-100 km ring in eight octants
    ring = (d >= 30.0) & (d <= 100.0)
    if ring.sum() > 16:
        east = (lo[ring] - lon) * np.cos(np.radians(lat)); north = la[ring] - lat
        octant = ((np.degrees(np.arctan2(east, north)) + 360.0) % 360.0 // 45.0).astype(int)
        means = [np.nanmean(bt[ring][octant == o]) for o in range(8) if (octant == o).sum() >= 2]
        out["ring_asymmetry_k"] = float(np.nanstd(means)) if len(means) >= 4 else float("nan")
    else:
        out["ring_asymmetry_k"] = float("nan")
    core, wall = d <= 25.0, (d >= 40.0) & (d <= 80.0)
    out["eye_warm_k"] = float(np.nanmean(bt[core]) - np.nanmean(bt[wall])) if core.any() and wall.any() else float("nan")
    if previous is not None and previous_hours and previous_hours > 0:
        earlier = ir_metrics(previous, lat, lon)
        for key, name in (("pc50", "d_pc50_per_h"), ("min_bt_c", "d_min_bt_per_h"), ("pc30", "d_pc30_per_h")):
            if np.isfinite(out.get(key, np.nan)) and np.isfinite(earlier.get(key, np.nan)):
                out[name] = (out[key] - earlier[key]) / float(previous_hours)
    bits = [f"coldest top {out['min_bt_c']:.0f} C within 100 km", f"{out['pc50']:.0%} of 50-200 km below -50 C ({out['pc30']:.0%} below -30 C)",
            f"ring asymmetry {out['ring_asymmetry_k']:.1f} K"]
    if np.isfinite(out.get("eye_warm_k", np.nan)) and out["eye_warm_k"] > 15.0:
        bits.append(f"eye visible (+{out['eye_warm_k']:.0f} K)")
    if "d_pc50_per_h" in out:
        bits.append(f"cold share {out['d_pc50_per_h'] * 100:+.1f} pts/h, tops {out['d_min_bt_per_h']:+.1f} C/h")
    out["text"] = "GOES IR (diagnostic, not in the probabilities): " + "; ".join(bits)
    return out


__all__ = ["ir_metrics"]
