# Copyright 2026 Nathan. Apache-2.0.
"""
Forecasts in words: what each hour will be like at a place ("light showers", "thunderstorms", "heavy rain"), and a
sentence per town and per parish ("Showers likely from about 2 pm (between 1 and 5 pm), heaviest around 3 pm with
thunder possible; 5-20 mm, locally 35 mm. Flash-flood risk moderate.").
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from .weather import describe


def _zone(name: str):
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        return None


def local_time(hour: float, zone: str, lon: float, with_day: bool = True) -> str:
    """``"Fri 2 pm"`` for an absolute hour (hours since 1970), in a place's time zone ("" for NaN)."""
    if hour is None or not np.isfinite(hour):
        return ""
    when = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(hours=float(hour))
    z = _zone(str(zone))
    local = when.astimezone(z) if z is not None else when + timedelta(hours=round(float(lon) / 15.0))
    h12 = local.hour % 12 or 12
    text = f"{h12} {'am' if local.hour < 12 else 'pm'}"
    return f"{local.strftime('%a')} {text}" if with_day else text


def hour_words(classes: np.ndarray, thunder: np.ndarray | None, kind: np.ndarray | None,
               torrential: np.ndarray | None) -> list[str]:
    """Words for each hour from ``classes (T, 4)``, ``thunder (T,)``, ``kind (T, 3)``, ``torrential (T,)``."""
    t = classes.shape[0]
    return [describe(classes[h], 0.0 if thunder is None else float(thunder[h]), None if kind is None else kind[h],
                     0.0 if torrential is None else float(torrential[h])) for h in range(t)]


def any_in_window(p_hourly: np.ndarray, hours: int, block: int = 3) -> np.ndarray:
    """Chance of at least one occurrence in the first ``hours`` from hourly chances of a process that persists for
    a few hours (thunder): independent ``block``-hour blocks, each as likely as its likeliest hour."""
    p = np.asarray(p_hourly)[..., :hours]
    pad = (-p.shape[-1]) % block
    if pad:
        p = np.concatenate([p, np.zeros(p.shape[:-1] + (pad,))], -1)
    blocks = p.reshape(p.shape[:-1] + (-1, block)).max(-1)
    return 1 - np.prod(1 - blocks, -1)


def town_sentence(row: dict) -> str:
    """One sentence for a town's next 24 hours from its table row."""
    p24 = row["any_rain_24h"]
    if p24 < 0.15:
        text = "Mostly dry."
    else:
        likely = "likely" if p24 >= 0.6 else "possible" if p24 >= 0.3 else "a slight chance"
        what = row.get("main_weather") or "rain"
        if likely == "a slight chance":
            text = f"A slight chance of {what}"
        else:
            text = f"{what[0].upper() + what[1:]} {likely}"
        if row.get("likely_start"):
            text += f" from about {row['likely_start']}"
            if row.get("start_window"):
                text += f" ({row['start_window']})"
        if row.get("heaviest_hour") and row.get("rain_24h_mm_p90", 0) >= 5:
            text += f", heaviest around {row['heaviest_hour']}"
        if row.get("thunder_24h", 0) >= 0.3:
            text += " with thunder" + (" likely" if row["thunder_24h"] >= 0.6 else " possible")
        text += "."
    lo, hi = row.get("rain_24h_mm_median", 0.0), row.get("rain_24h_mm_p90", 0.0)
    if hi >= 1:
        text += f" {lo:.0f}-{hi:.0f} mm" if lo >= 1 else f" Up to {hi:.0f} mm"
        text += " in 24 hours."
    if row.get("flood_risk") and row["flood_risk"] != "low":
        text += f" Flash-flood risk {row['flood_risk']}."
    if row.get("lightning_now", 0) > 0:
        text += f" Lightning detected within 10 km in the last 15 minutes ({row['lightning_now']} flashes)."
    return text


def parish_sentence(row: dict) -> str:
    when = f", most likely from {row['earliest_likely_start']}" if row.get("earliest_likely_start") else ""
    thunder = " Thunderstorms possible." if row.get("thunder_24h_highest", 0) >= 0.3 else ""
    flood = f" Flash-flood risk up to {row['flood_highest']}." if row.get("flood_highest", "low") != "low" else ""
    return (f"{row['any_rain_24h_typical']:.0%} chance of rain in the next 24 hours (up to {row['any_rain_24h_highest']:.0%}"
            f" at {row['wettest_place']}){when}; wettest places could see {row['rain_24h_mm_p90_highest']:.0f} mm."
            f"{thunder}{flood}")


__all__ = ["any_in_window", "hour_words", "local_time", "parish_sentence", "town_sentence"]
