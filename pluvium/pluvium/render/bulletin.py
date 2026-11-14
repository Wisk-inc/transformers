# Copyright 2026 Nathan. Apache-2.0.
"""
A national rain bulletin.

A headline for the country (the parishes where rain is most likely, thunder, any flood risk), then each parish in a
sentence. For Jamaica the bulletin can also be written in Jamaican Patois, built from the same numbers with fixed
phrasing (never generated), so the facts are identical in both languages.
"""

from __future__ import annotations

PATOIS_LEVEL = {"likely": "a go fall", "possible": "might fall", "unlikely": "nah look like"}


def _level(p: float) -> str:
    return "likely" if p >= 0.6 else "possible" if p >= 0.3 else "unlikely"


def bulletin(parishes: list[dict], country: str, issued: str, language: str = "en") -> str:
    rows = [r for r in parishes if r["country"] == country]
    if not rows:
        return f"No places forecast for {country}."
    rows.sort(key=lambda r: -r["any_rain_24h_typical"])
    wettest = [r["parish"] for r in rows if r["any_rain_24h_typical"] >= 0.5][:4]
    thunder = any(r.get("thunder_24h_highest", 0) >= 0.3 for r in rows)
    flood = [r["parish"] for r in rows if r.get("flood_highest", "low") in ("high", "very high")]
    if language == "jam":
        head = (f"Rain bulletin fi {country}, {issued}. " + (f"Rain a go fall inna {', '.join(wettest)}. " if wettest
                else "Most place a go stay dry. ") + ("Tunda an lightning possible. " if thunder else "")
                + (f"Watch out fi flood inna {', '.join(flood)}. " if flood else ""))
        lines = [head, ""]
        for r in rows:
            when = f" from bout {r['earliest_likely_start']}" if r.get("earliest_likely_start") else ""
            lines.append(f"{r['parish']}: rain {PATOIS_LEVEL[_level(r['any_rain_24h_typical'])]}{when} "
                         f"({r['any_rain_24h_typical']:.0%} chance). Wettest spot could get {r['rain_24h_mm_p90_highest']:.0f} mm.")
        return "\n".join(lines) + "\n"
    head = (f"Rain bulletin for {country}, issued {issued}. " + (f"Rain is most likely in {', '.join(wettest)}. "
            if wettest else "Most places should stay dry. ") + ("Thunderstorms are possible. " if thunder else "")
            + (f"Flash-flood risk is high in {', '.join(flood)}. " if flood else ""))
    lines = [head, ""] + [f"{r['parish']}: {r.get('summary', '')}" for r in rows]
    return "\n".join(lines) + "\n"


__all__ = ["bulletin"]
