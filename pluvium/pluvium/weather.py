# Copyright 2026 Nathan. Apache-2.0.
"""
What the weather *is*, in the words people use: intensity classes, drizzle versus rain versus showers, thunder.

**Intensity classes** (per hour, from the rate): dry below 0.2 mm/h; light 0.2-2.5; moderate 2.5-7.6; heavy 7.6 and
over (the American Meteorological Society's boundaries, also in common international use); "torrential" is the heavy
class above 50 mm/h (the usual threshold for violent rain).

**Airport reports** carry what an observer or sensor saw: ``-SHRA`` (light rain showers), ``+TSRA`` (thunderstorm
with heavy rain), ``DZ`` (drizzle), ``VCTS`` (thunderstorm in the vicinity). :func:`metar_weather` reads those groups
from a raw METAR, and :func:`code` packs what one hour's reports say into one small integer, which is how the
staged airport archive stores it and how live reports are fed to the model:

======  =====================================================================
bits    meaning
======  =====================================================================
0-1     intensity class (0 dry, 1 light, 2 moderate, 3 heavy)
2       thunder (at or near the airport)
3-4     kind of precipitation (0 not said, 1 drizzle, 2 rain, 3 showers)
5       intensity stated (otherwise "wet, intensity not said": class 1 is a placeholder)
======  =====================================================================

``-1`` is "no report this hour".
"""

from __future__ import annotations

import re

import numpy as np

EDGES = (0.2, 2.5, 7.6)            # mm/h: dry | light | moderate | heavy
TORRENTIAL = 50.0
CLASS_NAMES = ("dry", "light", "moderate", "heavy")
KIND_NAMES = ("", "drizzle", "rain", "showers")
UNKNOWN = -1

#: features of one hour's report given to the model (station assimilation)
OBS_FEATURES = ("known", "wet", "light", "moderate", "heavy", "thunder", "drizzle", "showers")


def rate_class(rate, edges=EDGES) -> np.ndarray:
    """Intensity class of rates (mm/h); NaN gives -1."""
    r = np.asarray(rate, np.float64)
    out = np.searchsorted(np.asarray(edges), np.nan_to_num(r, nan=-1.0), side="right").astype(np.int8)
    return np.where(np.isfinite(r), out, -1).astype(np.int8)


def code(intensity: int, thunder: bool = False, kind: int = 0, stated: bool = True) -> int:
    return int(intensity) | (4 if thunder else 0) | (int(kind) << 3) | (32 if stated else 0)


def decode(codes) -> dict:
    """Arrays of packed codes -> ``known, intensity, thunder, kind, stated`` (all int8/bool arrays)."""
    c = np.asarray(codes).astype(np.int16)
    known = c >= 0
    cc = np.where(known, c, 0)
    return {"known": known, "intensity": (cc & 3).astype(np.int8), "thunder": ((cc >> 2) & 1).astype(bool) & known,
            "kind": ((cc >> 3) & 3).astype(np.int8), "stated": ((cc >> 5) & 1).astype(bool) & known,
            "wet": known & ((cc & 3) > 0)}


def obs_features(codes) -> np.ndarray:
    """Packed codes ``(...)`` -> ``(..., len(OBS_FEATURES))`` float32 for the model."""
    d = decode(codes)
    f = np.zeros(np.shape(codes) + (len(OBS_FEATURES),), np.float32)
    f[..., 0] = d["known"]
    f[..., 1] = d["wet"]
    for k in (1, 2, 3):
        f[..., 1 + k] = d["wet"] & (d["intensity"] == k) & d["stated"]
    f[..., 5] = d["thunder"]
    f[..., 6] = d["kind"] == 1
    f[..., 7] = d["kind"] == 3
    return f


def class_mask(codes, states: int) -> np.ndarray:
    """Which of ``states`` hidden-state classes each report allows: ``(..., states)`` bool. Unknown: all; dry: class 0;
    wet with stated intensity: that class (K = 4) or class 1 (K = 2); wet, intensity not said: every wet class."""
    d = decode(codes)
    shape = np.shape(codes)
    m = np.ones(shape + (states,), bool)
    known = d["known"]
    dry = known & ~d["wet"]
    wet = d["wet"]
    m[dry] = False
    m[..., 0] = np.where(dry, True, np.where(wet, False, True))
    if states == 2:
        m[..., 1] = np.where(dry, False, True)
        return m
    stated = wet & d["stated"]
    for k in range(1, states):
        m[..., k] = np.where(dry, False, np.where(stated, d["intensity"] == min(k, 3), True))
    return m


# ------------------------------------------------------------------------------------------- METAR --

_WX = re.compile(r"^(?P<int>[+-]|VC)?(?P<desc>MI|PR|BC|DR|BL|SH|TS|FZ)?(?P<desc2>SH|TS)?"
                 r"(?P<precip>(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP)+)?(?P<obsc>BR|FG|FU|VA|DU|SA|HZ|PY)?(?P<other>PO|SQ|FC|SS|DS)?$")
_RECENT = re.compile(r"^RE(?P<desc>SH|TS)?(?P<precip>(?:DZ|RA|GR|GS|UP)+)?$")


def metar_weather(raw: str) -> dict | None:
    """
    Present weather from a raw METAR/SPECI: ``{"wet", "intensity", "thunder", "kind", "stated", "recent_wet",
    "recent_thunder"}``, or None if the text is not a report. Groups in the vicinity (``VC``) count for thunder but
    not for rain at the airport; ``RE..`` groups describe the previous hour.
    """
    if not raw:
        return None
    text = raw.replace("=", " ")
    i = text.find("METAR")
    j = text.find("SPECI")
    k = min(x for x in (i, j) if x >= 0) if (i >= 0 or j >= 0) else -1
    if k < 0:
        return None
    body = text[k:].split(" RMK")[0].split()
    remark = text[k:].split(" RMK", 1)[1] if " RMK" in text[k:] else ""
    wet, thunder, intensity, kind, stated = False, False, 0, 0, False
    recent_wet = recent_thunder = False
    missing = False
    for tok in body[2:]:
        if set(tok) == {"/"}:                       # "//" weather not observed, "////" visibility missing
            missing = True
            continue
        if tok in ("NOSIG", "TEMPO", "BECMG"):
            break                                    # trend groups describe the future
        m = _RECENT.match(tok)
        if m and (m.group("precip") or m.group("desc")):
            recent_wet |= bool(m.group("precip"))
            recent_thunder |= m.group("desc") == "TS"
            continue
        m = _WX.match(tok)
        if not m or not any(m.group(g) for g in ("desc", "desc2", "precip", "other")):
            continue
        descs = {m.group("desc"), m.group("desc2")}
        if "TS" in descs:
            thunder = True
        precip = m.group("precip") or ""
        if m.group("int") == "VC" or not any(p in precip for p in ("DZ", "RA", "GR", "GS", "UP")):
            continue
        wet = True
        sign = m.group("int") or ""
        if "DZ" in precip and "RA" not in precip:
            level = 1 if sign != "+" else 2
            this_kind = 1
        else:
            level = {"-": 1, "": 2, "+": 3}.get(sign, 2)
            this_kind = 3 if ("SH" in descs or "TS" in descs) else 2
        if level >= intensity:
            intensity, kind = level, this_kind
        stated = True
    if "LTG" in remark or "TS " in remark or remark.strip().startswith("TS"):
        thunder = True
    if missing and not wet and not thunder:
        return {"wet": False, "intensity": 0, "thunder": False, "kind": 0, "stated": False, "unknown": True,
                "recent_wet": recent_wet, "recent_thunder": recent_thunder}
    return {"unknown": False, "wet": wet, "intensity": intensity if wet else 0, "thunder": thunder, "kind": kind if wet else 0,
            "stated": stated or not wet, "recent_wet": recent_wet, "recent_thunder": recent_thunder}


def code_from_metar(raw: str) -> tuple[int, int] | None:
    """``(code for the report's hour, code for the previous hour or -1)`` from a raw METAR."""
    w = metar_weather(raw)
    if w is None:
        return None
    now = UNKNOWN if w.get("unknown") else code(w["intensity"], w["thunder"], w["kind"], w["stated"])
    before = UNKNOWN
    if w["recent_wet"] or w["recent_thunder"]:
        before = code(1 if w["recent_wet"] else 0, w["recent_thunder"], 0, stated=False)
    return now, before


# ---------------------------------------------------------------------- WMO present-weather codes --

def code_from_wmo(value: int, automated: bool) -> tuple[int | None, int]:
    """WMO 4677 (manual, ``automated=False``) or 4680 (automatic) present weather -> ``(code now or None if the code
    says nothing about precipitation, code for the previous hour or -1)``."""
    v = int(value)
    if not automated:
        if v in (13, 17):                                   # lightning seen / thunder heard, no precipitation
            return code(0, True), UNKNOWN
        if 20 <= v <= 29:
            before = code(1 if v in (20, 21, 23, 24, 25, 26, 27) else 0, v == 29, 1 if v == 20 else 0,
                          stated=False) if v in (20, 21, 23, 24, 25, 26, 27, 29) else UNKNOWN
            return code(0), before
        if 50 <= v <= 59:
            level = {50: 1, 51: 1, 52: 2, 53: 2, 54: 3, 55: 3, 58: 1, 59: 2}.get(v, 1)
            return code(min(level, 2) if v < 58 else level, False, 1 if v < 58 else 2), UNKNOWN
        if 60 <= v <= 69:
            level = {60: 1, 61: 1, 62: 2, 63: 2, 64: 3, 65: 3}.get(v, 2)
            return code(level, False, 2), UNKNOWN
        if 80 <= v <= 90:
            level = {80: 1, 81: 2, 82: 3}.get(v, 2)
            return code(level, False, 3), UNKNOWN
        if 91 <= v <= 94:
            return code(1 if v in (91, 93) else 2, False, 2), code(0, True, stated=False)
        if 95 <= v <= 99:
            level = {95: 2, 96: 2, 97: 3, 98: 0, 99: 3}[v]
            return code(level, True, 3 if level else 0, stated=v != 98), UNKNOWN
        if v <= 49:
            return code(0), UNKNOWN
        return None, UNKNOWN
    # automatic stations (WMO 4680)
    if 21 <= v <= 26:
        return code(0), code(1 if v != 26 else 0, v == 26, stated=False)
    if 40 <= v <= 42:
        return code({40: 1, 41: 1, 42: 3}[v], stated=v != 40), UNKNOWN
    if 50 <= v <= 58:
        level = {51: 1, 52: 2, 53: 2}.get(v, 1)
        return code(level, False, 1 if v <= 53 else 2), UNKNOWN
    if 60 <= v <= 68:
        level = {61: 1, 62: 2, 63: 3}.get(v, 2)
        return code(level, False, 2), UNKNOWN
    if 80 <= v <= 89:
        level = {80: 1, 81: 1, 82: 2, 83: 3, 84: 3}.get(v, 2)
        return code(level, False, 3, stated=v != 80), UNKNOWN
    if 90 <= v <= 99:
        level = {90: 0, 91: 0, 92: 2, 93: 2, 94: 0, 95: 3, 96: 3}.get(v, 0)
        return code(level, True, 3 if level else 0, stated=v not in (90,)), UNKNOWN
    if v < 40:
        return code(0), UNKNOWN
    return None, UNKNOWN


def code_from_gauge(depth_mm: float) -> int:
    """One-hour gauge depth -> code (intensity from the hour's total)."""
    k = int(rate_class(np.array([depth_mm]))[0])
    return code(max(k, 0), stated=True)


def merge(a: int, b: int) -> int:
    """Combine two reports for the same hour: wetter and stormier wins; a stated intensity beats an unstated one."""
    if a < 0:
        return b
    if b < 0:
        return a
    da, db = decode(np.array([a]))["intensity"][0], decode(np.array([b]))["intensity"][0]
    ta, tb = bool(a & 4), bool(b & 4)
    sa, sb = bool(a & 32), bool(b & 32)
    level = max(da, db)
    if da > 0 and db > 0 and sa != sb:
        level = da if sa else db
    kind = (a >> 3) & 3 if ((a >> 3) & 3) else (b >> 3) & 3
    stated = (sa and da == level) or (sb and db == level) or level == 0
    return code(level, ta or tb, kind if level else 0, stated)


# ----------------------------------------------------------------------------------------- wording --

def describe(p_state: np.ndarray, p_thunder: float, p_kind: np.ndarray | None = None, p_torrential: float = 0.0) -> str:
    """Words for one hour from its state probabilities ``(K,)`` (dry first), thunder chance and kind chances
    ``(3,)`` (drizzle, rain, showers given rain)."""
    p_state = np.asarray(p_state, np.float64)
    wet = 1.0 - p_state[0]
    if wet < 0.15 and p_thunder < 0.15:
        return "dry"
    chance = "slight chance of " if wet < 0.3 else "chance of " if wet < 0.6 else ""
    if p_thunder >= max(0.25, 0.5 * wet):
        return f"{chance}thunderstorms" + (" with torrential rain" if p_torrential > 0.15 else "")
    if p_state.size >= 4:
        given = p_state[1:] / max(wet, 1e-9)
        level = int(np.argmax(given * np.array([1.0, 1.15, 1.3])))     # a little weight towards the heavier word
        if p_torrential > 0.2:
            word = "torrential rain"
        else:
            word = ("light", "moderate", "heavy")[level]
    else:
        word = ""
    kind = "rain"
    if p_kind is not None:
        kind = ("drizzle", "rain", "showers")[int(np.argmax(p_kind))]
        if kind == "drizzle" and word in ("moderate", "heavy"):
            kind = "rain"
    if word == "torrential rain":
        return chance + word
    return (chance + (f"{word} {kind}" if word else kind)).strip()


__all__ = ["CLASS_NAMES", "EDGES", "KIND_NAMES", "OBS_FEATURES", "TORRENTIAL", "UNKNOWN", "class_mask", "code",
           "code_from_gauge", "code_from_metar", "code_from_wmo", "decode", "describe", "merge", "metar_weather",
           "obs_features", "rate_class"]
