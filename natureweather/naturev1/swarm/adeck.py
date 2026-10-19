# Copyright 2026 Nathan. Apache-2.0.
"""
The NHC's operational guidance for a storm, read from the public ATCF.

* **The a-deck** (``aid_public/a{id}.dat.gz`` during the season, ``archive/{year}/`` after it): every aid's track and
  intensity for every cycle -- the official forecast (OFCL), the consensus aids (TVCN, HCCA, IVCN), the hurricane
  models (HWRF, HMON, HAFS-A and -B), the global models' interpolated runs (GFS, the GEFS mean, UKMET, CMC) and the
  statistical intensity models (SHIPS with decay, LGEM). :func:`aids_at` interpolates them to the forecast leads;
  :mod:`naturev1.cyclone.aids` blends them in with the fitted tables.
* **The rapid-intensification probabilities**: the SHIPS text (``stext/``) carries the SHIPS-RII, logistic, Bayesian,
  DTOPS and consensus schemes live; the season archive's e-deck (``RIOC`` and friends) carries the same after the
  fact. Reported beside our own probability (:func:`nhc_ri_probability`); not a member.
"""

from __future__ import annotations

import numpy as np

from .fixes import ATCF, _get, fetch_deck

AID_TITLES = {"OFCL": "NHC official forecast", "OFCI": "previous official forecast (interpolated)", "CARQ": "operational analysis",
              "TVCN": "track consensus", "TVCA": "track consensus (all)", "TVCX": "track consensus with ECMWF", "HCCA": "HFIP corrected consensus",
              "IVCN": "intensity consensus", "IVDR": "intensity consensus (dynamical)", "HWFI": "HWRF", "HMNI": "HMON", "HFAI": "HAFS-A",
              "HFBI": "HAFS-B", "HWRF": "HWRF (late)", "HMON": "HMON (late)", "HFSA": "HAFS-A (late)", "HFSB": "HAFS-B (late)",
              "AVNI": "GFS", "AVNO": "GFS (late)", "AEMI": "GEFS mean", "AEMN": "GEFS mean (late)", "EMXI": "ECMWF", "EMX": "ECMWF (late)",
              "UKXI": "UKMET", "EGRI": "UKMET", "CMCI": "CMC", "CTCI": "COAMPS-TC", "CTCX": "COAMPS-TC (late)", "NVGI": "NAVGEM",
              "DSHP": "SHIPS with decay", "SHIP": "SHIPS", "LGEM": "LGEM", "OCD5": "CLIPER/SHIFOR", "CLP5": "CLIPER", "SHF5": "SHIFOR",
              "XTRP": "extrapolation", "RVCN": "wind-radii consensus", "GPCC": "GPCE climatology cone"}
TRACK_AIDS = ("OFCL", "HCCA", "TVCN", "HFAI", "HFBI", "HWFI", "HMNI", "AVNI", "AEMI", "EMXI", "UKXI", "EGRI", "CMCI", "CTCI")
INTENSITY_AIDS = ("OFCL", "IVCN", "HCCA", "HFAI", "HFBI", "HWFI", "HMNI", "DSHP", "LGEM", "SHIP", "AVNI", "EMXI", "CTCI")


def fetch_adeck(storm_id: str, archive: bool = False) -> str:
    """The text of the storm's a-deck (``aid_public`` live, the season archive with ``archive=True``)."""
    return fetch_deck(storm_id, "a", archive)


def fetch_edeck(storm_id: str, archive: bool = True) -> str:
    """The text of the storm's e-deck (the probabilistic aids). Only the season archive is public."""
    sid = storm_id.lower()
    return _get(f"{ATCF}/archive/{sid[-4:]}/e{sid}.dat.gz").decode("latin-1")


def _coord(text: str) -> float:
    """ATCF tenths of a degree with a hemisphere letter, ``"164N"`` -> 16.4, ``"773W"`` -> -77.3."""
    text = text.strip()
    if len(text) < 2 or text[-1] not in "NSEW":
        return float("nan")
    value = float(text[:-1]) / 10.0
    return -value if text[-1] in "SW" else value


def _cycle(text: str):
    t = text.strip()
    if len(t) < 10 or not t[:10].isdigit():
        return None
    return np.datetime64(f"{t[:4]}-{t[4:6]}-{t[6:8]}T{t[8:10]}:00")


def parse_adeck(text: str) -> dict:
    """
    ``{cycle (np.datetime64, minutes): {tech: {tau_h: (lat, lon, vmax_kt, mslp_hpa)}}}`` -- the first row of each
    tau (the 34-kt wind-radii row, or the only one). A 0/0 position, a 0 wind and a 0 pressure read as missing.
    """
    out: dict = {}
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 9:
            continue
        when = _cycle(f[2])
        if when is None:
            continue
        try:
            tau = int(f[5])
            lat, lon = _coord(f[6]), _coord(f[7])
            vmax = float(f[8]) if f[8] else float("nan")
            mslp = float(f[9]) if len(f) > 9 and f[9] else float("nan")
        except ValueError:
            continue
        d = out.setdefault(when, {}).setdefault(f[4], {})
        if tau not in d:
            no_position = not (np.isfinite(lat) and np.isfinite(lon)) or (lat == 0.0 and lon == 0.0)
            d[tau] = (float("nan") if no_position else lat, float("nan") if no_position else lon,
                      vmax if vmax > 0 else float("nan"), mslp if mslp > 0 else float("nan"))
    return out


def interpolate(aid: dict, leads) -> np.ndarray:
    """(L, 3) latitude, longitude, wind at ``leads`` from an aid's taus, linear between them (NaN where a lead is not
    bracketed by two taus with the value, so an aid is never extrapolated)."""
    taus = sorted(t for t in aid if t >= 0)
    out = np.full((len(leads), 3), np.nan)
    for k, h in enumerate(leads):
        lo = [t for t in taus if t <= h]
        hi = [t for t in taus if t >= h]
        if not lo or not hi:
            continue
        a, b = lo[-1], hi[0]
        pa, pb = aid[a], aid[b]
        w = 0.0 if b == a else (h - a) / (b - a)
        for i in range(3):
            if np.isfinite(pa[i]) and np.isfinite(pb[i]):
                va, vb = pa[i], pb[i]
                if i == 1 and abs(vb - va) > 180.0:                 # across the dateline
                    vb = vb + 360.0 if vb < va else vb - 360.0
                v = va + w * (vb - va)
                out[k, i] = ((v + 180.0) % 360.0) - 180.0 if i == 1 else v
    return out


def aids_at(deck: dict, when, leads, techs=None) -> dict:
    """
    The aids of one cycle at the forecast leads.

    Returns ``{"cycle", "track": {tech: (L, 2)}, "intensity": {tech: (L,)}, "available": [tech, ...]}``; ``techs``
    restricts the aids (None: every aid with at least two taus). An intensity aid's positions are kept too (SHIPS and
    LGEM run along the official track), which is harmless: the blend tables name the aids they use.
    """
    when = np.datetime64(when, "m")
    cycle = deck.get(when) or {}
    track, intensity = {}, {}
    for tech, aid in cycle.items():
        if techs is not None and tech not in techs:
            continue
        if len([t for t in aid if t >= 0]) < 2:
            continue
        v = interpolate(aid, leads)
        if np.isfinite(v[:, 0]).any():
            track[tech] = v[:, :2]
        if np.isfinite(v[:, 2]).any():
            intensity[tech] = v[:, 2]
    return {"cycle": when, "track": track, "intensity": intensity, "available": sorted(set(track) | set(intensity))}


def parse_edeck_ri(text: str) -> dict:
    """The e-deck's rapid-intensification rows: ``{cycle: {tech: {(tau_h, change_kt): probability}}}`` (fractions)."""
    out: dict = {}
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 10 or f[3] != "RI":
            continue
        when = _cycle(f[2])
        if when is None:
            continue
        try:
            tau, prob, change = int(f[5]), float(f[8]) / 100.0, float(f[9])
        except ValueError:
            continue
        out.setdefault(when, {}).setdefault(f[4], {})[(tau, change)] = prob
    return out


def parse_ships_ri(text: str) -> dict | None:
    """The ``Matrix of RI probabilities`` of a SHIPS text file: ``{scheme: {"30/24": probability, ...}}`` (fractions,
    thresholds as kt/h), None when the file has no matrix."""
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if "RI (kt / h)" in line), None)
    if header is None:
        return None
    cols = [c.strip() for c in lines[header].split("|")[1:] if c.strip()]
    out: dict = {}
    for line in lines[header + 1:]:
        if ":" not in line:
            if out and not line.strip():
                break
            continue
        name, _, rest = line.partition(":")
        values = [v for v in rest.replace("%", " ").split()]
        try:
            numbers = [float(v) / 100.0 for v in values]
        except ValueError:
            break
        if len(numbers) != len(cols):
            break
        out[name.strip()] = dict(zip(cols, numbers))
    return out or None


def fetch_ships_text(storm_id: str, when) -> str:
    """The SHIPS text for a cycle, ``stext/{yymmddhh}{BASIN}{nn}{yy}_ships.txt`` (live; the archive keeps the e-deck instead)."""
    sid = storm_id.lower()
    stamp = str(np.datetime64(when, "h")).replace("-", "").replace("T", "")
    return _get(f"{ATCF}/stext/{stamp[2:10]}{sid[:2].upper()}{sid[2:4]}{sid[-2:]}_ships.txt").decode("latin-1")


def nhc_ri_probability(storm_id: str, when, archive: bool = False) -> dict | None:
    """
    The NHC's probability of +30 kt in 24 h for the cycle: the SHIPS text live, the e-deck from the season archive.
    DTOPS first (held-out 2020-2022, 453 cases: Brier skill 0.17 against 0.07 for the SHIPS-RII consensus and 0.13 for
    ours; the e-deck archive carries no RI rows before 2020, so nothing here was chosen on the tuning seasons), then
    the consensus. ``{"source", "p30_24", "schemes": {scheme: p}}``, None when the cycle has none.
    """
    when = np.datetime64(when, "m")
    if archive:
        cycle = parse_edeck_ri(fetch_edeck(storm_id)).get(when) or {}
        schemes = {tech: v[(24, 30.0)] for tech, v in cycle.items() if (24, 30.0) in v}
        for tech in ("DTOP", "RIOC", "RIOD", "SDCN"):
            if tech in schemes:
                return {"source": f"NHC e-deck {tech}", "p30_24": schemes[tech], "schemes": schemes}
        return None
    table = parse_ships_ri(fetch_ships_text(storm_id, when))
    if not table:
        return None
    schemes = {k: v.get("30/24") for k, v in table.items() if v.get("30/24") is not None}
    for name in ("DTOPS", "Consensus", "SHIPS-RII"):
        if name in schemes:
            return {"source": f"SHIPS text {name}", "p30_24": schemes[name], "schemes": schemes}
    return None


__all__ = ["AID_TITLES", "INTENSITY_AIDS", "TRACK_AIDS", "aids_at", "fetch_adeck", "fetch_edeck", "fetch_ships_text", "interpolate",
           "nhc_ri_probability", "parse_adeck", "parse_edeck_ri", "parse_ships_ri"]
