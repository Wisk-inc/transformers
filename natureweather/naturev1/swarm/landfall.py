# Copyright 2026 Nathan. Apache-2.0.
"""
Landfall, answered the way a parish needs it: when, where, how strong -- with the uncertainty that is really there.

From the swarm's worldlines and the consensus, for one country (or any set of units):

* **when the centre comes ashore** -- the distribution of first-landfall times over the worlds (median, the 10-90%
  range, the chance of no landfall within the forecast), and the consensus track's own landfall time;
* **where** -- the probability that each parish is the first the centre crosses;
* **how strong at landfall** -- the intensity forecast at the consensus landfall time, with the calibrated 10-90%
  band from the storm's regime (:data:`naturev1.cyclone.coefficients.INTENSITY_SPREAD`), and the peak intensity of
  the forecast and when it comes;
* **each parish's own numbers** -- from the impact painting: expected peak sustained wind and gust, the 90th
  percentile of the peak, when the peak comes, when tropical-storm and hurricane force arrive, rain.

Every time is UTC and comes with its date. Research model, not an operational forecast.
"""

from __future__ import annotations

import numpy as np


def _select(region, where) -> np.ndarray:
    """Unit ids of a country name, a list of labels / names, or every unit when None."""
    if where is None:
        return np.arange(len(region.names))
    if isinstance(where, str):
        ids = [u for u, c in enumerate(region.countries) if c == where]
        if not ids:
            ids = [u for u, n in enumerate(region.names) if n == where]
        return np.asarray(ids, int)
    wanted = set(where)
    return np.asarray([u for u in range(len(region.names)) if region.label(u) in wanted or region.names[u] in wanted], int)


def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    c = np.cumsum(w) / max(w.sum(), 1e-12)
    return float(np.interp(q, c, v))


def landfall_summary(region, result: dict, case, impact=None, where="Jamaica", analysis_time=None) -> dict:
    """
    Args:
        region: the :class:`naturev1.swarm.geography.RegionMap`.
        result: a :func:`naturev1.cyclone.forecast.physics_forecast` result with a swarm.
        case: the storm's :class:`naturev1.cyclone.data.Case`.
        impact: the :class:`naturev1.swarm.impact.ImpactResult` for the same forecast (parish numbers), optional.
        where: country name, unit labels, or None for anywhere.
        analysis_time: the forecast's initial time (``case.time`` when None), to turn hours into dates.

    Returns a dict with ``p_landfall``, ``hours`` (median, p10, p90), ``times`` (the same as UTC strings),
    ``first_unit`` ({label: probability}), ``consensus`` (hour, time, unit, intensity_kt, intensity_band_kt),
    ``peak`` (intensity_kt, hour, time, band_kt) and ``units`` (the parishes' own numbers, hardest hit first).
    """
    t0 = np.datetime64(case.time if analysis_time is None else analysis_time, "m")
    stamp = lambda hours: str(t0 + np.timedelta64(int(round(float(hours) * 60)), "m")).replace("T", " ") + " UTC" if np.isfinite(hours) else "-"
    ids = _select(region, where)
    chosen = np.zeros(len(region.names) + 1, bool)
    chosen[ids] = True
    fc = result["swarm"]["forecast"]
    track, hours, weights = fc.track, np.asarray(fc.track_hours, float), np.asarray(fc.weights, float)
    weights = weights / weights.sum()
    lat, lon = track[..., 0], track[..., 1]
    unit = region.unit_of(lat, lon)                                        # (N, T), -1 at sea
    inside = chosen[np.where(unit >= 0, unit, len(region.names))]
    first = np.argmax(inside, axis=1)
    hit = inside.any(axis=1)
    out = {"where": where, "p_landfall": float(weights[hit].sum()), "horizon_h": float(hours[-1])}
    if hit.any():
        lh = hours[first[hit]]
        wh = weights[hit] / weights[hit].sum()
        q = {k: _weighted_quantile(lh, wh, p) for k, p in (("median", 0.5), ("p10", 0.1), ("p90", 0.9))}
        out["hours"] = q
        out["times"] = {k: stamp(v) for k, v in q.items()}
        counts = {}
        for u, w in zip(unit[hit, first[hit]], wh):
            counts[u] = counts.get(u, 0.0) + float(w)
        out["first_unit"] = {region.label(u): round(p * out["p_landfall"], 4) for u, p in sorted(counts.items(), key=lambda kv: -kv[1])}
    # the consensus track, hourly, and the intensity along it
    leads = np.asarray(result["leads"], float)
    ct = np.vstack([[case.lat, case.lon], result["track"]])
    ok = np.isfinite(ct[:, 0])
    ch = np.concatenate([[0.0], leads])[ok]
    hh = np.arange(0.0, ch[-1] + 1e-9, 1.0)
    clat = np.interp(hh, ch, ct[ok, 0])
    clon = np.degrees(np.interp(hh, ch, np.unwrap(np.radians(ct[ok, 1]))))
    cunit = region.unit_of(clat, clon)
    cin = chosen[np.where(cunit >= 0, cunit, len(region.names))]
    kt = np.asarray(result.get("intensity", np.full(leads.size, np.nan)), float)
    good = np.isfinite(kt)
    ikt = lambda h: float(np.interp(h, np.concatenate([[0.0], leads[good]]), np.concatenate([[case.wind_kt], kt[good]]))) if good.any() else float("nan")
    cat = result.get("category_probability")
    band = None
    spread = None
    try:
        from ..cyclone import coefficients

        table = coefficients.INTENSITY_SPREAD
        spread = (np.asarray(table["leads"], float), np.asarray(table["quantiles"][result.get("intensity_regime", "normal")], float),
                  np.asarray(table["quantile_levels"], float))
    except Exception:
        pass

    def band_at(h):
        if spread is None:
            return None
        lh, q, levels = spread
        row = np.array([np.interp(h, np.concatenate([[0.0], lh]), np.concatenate([[0.0], q[:, j]])) for j in range(q.shape[1])])
        v = ikt(h)
        return (round(v + float(np.interp(0.1, levels, row)), 0), round(v + float(np.interp(0.9, levels, row)), 0))

    cons = {"landfall": None}
    if cin.any():
        k = int(np.argmax(cin))
        cons = {"hour": float(hh[k]), "time": stamp(hh[k]), "unit": region.label(cunit[k]), "lat": round(float(clat[k]), 2),
                "lon": round(float(clon[k]), 2), "intensity_kt": round(ikt(hh[k]), 0), "intensity_band_kt": band_at(hh[k])}
    out["consensus"] = cons
    if result.get("peak_probability"):
        out["peak_probability"] = result["peak_probability"]
    if_ri = result.get("intensity_if_ri")
    p_ri = result.get("ri_probability")
    if if_ri is not None and p_ri is not None and cin.any() and np.isfinite(if_ri).any():
        kk = int(np.argmax(cin))
        g2 = np.isfinite(if_ri)
        at = float(np.interp(hh[kk], np.concatenate([[0.0], leads[g2]]), np.concatenate([[case.wind_kt], np.asarray(if_ri)[g2]])))
        j = int(np.nanargmax(if_ri))
        out["if_rapid"] = {"probability": float(p_ri), "landfall_kt": round(at, 0), "peak_kt": round(float(np.nanmax(if_ri)), 0),
                           "peak_time": stamp(leads[j])}
    if good.any():
        kk = int(np.nanargmax(np.concatenate([[case.wind_kt], kt[good]])))
        ph = np.concatenate([[0.0], leads[good]])[kk]
        out["peak"] = {"intensity_kt": round(float(np.concatenate([[case.wind_kt], kt[good]])[kk]), 0), "hour": float(ph), "time": stamp(ph),
                       "band_kt": band_at(ph) if ph > 0 else None, "strengthens": bool(ph > 0),
                       "forecast_max_kt": round(float(np.nanmax(kt)), 0), "forecast_max_time": stamp(leads[good][int(np.nanargmax(kt[good]))])}
    if impact is not None:
        rows = []
        for u in impact.units:
            if u["unit"] in set(ids.tolist()):
                row = dict(u)
                for key, name in (("arrival_h", "arrival_time"), ("hu_arrival_h", "hu_arrival_time"), ("peak_hour", "peak_time")):
                    if key in row:
                        row[name] = stamp(row[key])
                rows.append(row)
        rows.sort(key=lambda r: (-np.nan_to_num(r.get("peak_kt_if_hit", r.get("peak_kt", 0.0))), -r.get("p64", 0.0)))
        out["units"] = rows
    return out


def format_landfall(summary: dict, top: int = 14) -> str:
    """The summary as plain text."""
    s = summary
    lines = [f"Landfall on {s['where']}: probability {s['p_landfall']:.0%} within {s['horizon_h']:.0f} h"]
    if "times" in s:
        lines.append(f"  centre comes ashore: median {s['times']['median']}  (10-90%: {s['times']['p10']} to {s['times']['p90']})")
        first = list(s.get("first_unit", {}).items())[:4]
        lines.append("  first parish crossed: " + ", ".join(f"{k.split(',')[0]} {v:.0%}" for k, v in first))
    c = s.get("consensus") or {}
    if c.get("time"):
        band = c.get("intensity_band_kt")
        lines.append(f"  consensus track: {c['time']} near {c['unit'].split(',')[0]} ({c['lat']}N {abs(c['lon']):.2f}W), "
                     f"{c['intensity_kt']:.0f} kt" + (f" (10-90%: {band[0]:.0f}-{band[1]:.0f} kt)" if band else ""))
    r = s.get("if_rapid")
    if r:
        lines.append(f"  if it intensifies rapidly (probability {r['probability']:.0%}): landfall at {r['landfall_kt']:.0f} kt, "
                     f"peak {r['peak_kt']:.0f} kt at {r['peak_time']}")
    pp = s.get("peak_probability")
    if pp:
        names = {64: "hurricane", 96: "major (96 kt)", 113: "category 4 (113 kt)", 137: "category 5 (137 kt)"}
        lines.append("  chance the storm peaks at or above: " + ", ".join(f"{names.get(int(t), f'{t:.0f} kt')} {p:.0%}" for t, p in zip(pp["thresholds_kt"], pp["probability"])))
    p = s.get("peak")
    if p and p.get("strengthens", True):
        band = p.get("band_kt")
        lines.append(f"  peak intensity: {p['intensity_kt']:.0f} kt at {p['time']}" + (f" (10-90%: {band[0]:.0f}-{band[1]:.0f} kt)" if band else ""))
    elif p:
        lines.append(f"  peak intensity: the storm is at its forecast peak now ({p['intensity_kt']:.0f} kt); the forecast does not strengthen it "
                     f"(forecast maximum {p['forecast_max_kt']:.0f} kt at {p['forecast_max_time']})")
    if s.get("units"):
        head = (f"  {'parish':22s} {'P(HU)':>6s} {'peak kt if hit':>14s} {'p90 kt':>7s} {'gust kt if hit':>14s} {'rain if hit':>11s} {'max mm':>7s} "
                f"{'TS arrives (UTC)':>17s} {'HU arrives (UTC)':>17s} {'peak at (UTC)':>14s}")
        lines += ["", head, "  " + "-" * (len(head) - 2)]
        for r in s["units"][:top]:
            lines.append(f"  {r['label'].split(',')[0][:22]:22s} {r['p64']:6.0%} {r.get('peak_kt_if_hit', r['peak_kt']):14.0f} {r.get('peak_kt_p90', float('nan')):7.0f} "
                         f"{r.get('gust_kt_if_hit', r.get('peak_gust_kt', float('nan'))):14.0f} {r.get('rain_mm_if_hit', r['rain_mm']):11.0f} {r.get('rain_max_mm_if_hit', float('nan')):7.0f} {r.get('arrival_time', '-')[5:16]:>17s} "
                         f"{r.get('hu_arrival_time', '-')[5:16]:>17s} {r.get('peak_time', '-')[5:16]:>14s}")
        lines.append("  'if hit' = over the worlds that bring at least tropical-storm force to the parish; p90 = 90th percentile over all worlds; "
                     "rain = parish area mean, max = its wettest cell (what a gauge reports)")
    return "\n".join(lines)


__all__ = ["format_landfall", "landfall_summary"]
