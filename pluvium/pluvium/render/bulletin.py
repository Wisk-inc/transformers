# Copyright 2026 Nathan. Apache-2.0.
"""
A national rain bulletin.

A headline for the country (the parishes or provinces where rain is most likely, thunder, any flood risk), then each
parish in a sentence. Bulletins are written in each country's languages: English, Spanish, French, and for Jamaica also
Jamaican Patois. Every language is built from the same numbers with fixed phrasing (never generated), so the facts
are identical in all of them.
"""

from __future__ import annotations

PATOIS_LEVEL = {"likely": "a go fall", "possible": "might fall", "unlikely": "nah look like"}
LEVEL = {"es": {"likely": "probable", "possible": "posible", "unlikely": "poco probable"},
         "fr": {"likely": "probable", "possible": "possible", "unlikely": "peu probable"}}
#: languages by country where GeoNames cannot be asked (offline); everything else defaults to English
KNOWN_LANGUAGES = {"JM": ("en", "jam"), "HT": ("fr",), "DO": ("es",), "CU": ("es",), "PR": ("es", "en"), "MX": ("es",),
                   "GT": ("es",), "HN": ("es",), "SV": ("es",), "NI": ("es",), "CR": ("es",), "PA": ("es",),
                   "CO": ("es",), "VE": ("es",), "GP": ("fr",), "MQ": ("fr",), "MF": ("fr",), "BL": ("fr",),
                   "GF": ("fr",), "SN": ("fr",), "CI": ("fr",), "CM": ("fr",), "ML": ("fr",), "BF": ("fr",)}


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
    if language in LEVEL:
        return _latin(rows, country, issued, language, wettest, thunder, flood)
    head = (f"Rain bulletin for {country}, issued {issued}. " + (f"Rain is most likely in {', '.join(wettest)}. "
            if wettest else "Most places should stay dry. ") + ("Thunderstorms are possible. " if thunder else "")
            + (f"Flash-flood risk is high in {', '.join(flood)}. " if flood else ""))
    lines = [head, ""] + [f"{r['parish']}: {r.get('summary', '')}" for r in rows]
    return "\n".join(lines) + "\n"


def _latin(rows, country, issued, language, wettest, thunder, flood) -> str:
    """Spanish and French, from the same numbers as the English."""
    lv = LEVEL[language]
    if language == "es":
        head = (f"Boletín de lluvia para {country}, emitido {issued}. "
                + (f"La lluvia es más probable en {', '.join(wettest)}. " if wettest
                   else "La mayoría de los lugares deberían permanecer secos. ")
                + ("Posibles tormentas eléctricas. " if thunder else "")
                + (f"Riesgo alto de inundaciones repentinas en {', '.join(flood)}. " if flood else ""))
        line = "{parish}: lluvia {level}{when} ({p:.0%} de probabilidad). El punto más lluvioso podría recibir {mm:.0f} mm."
        since = " desde aproximadamente {t}"
    else:
        head = (f"Bulletin de pluie pour {country}, émis {issued}. "
                + (f"La pluie est la plus probable à {', '.join(wettest)}. " if wettest
                   else "La plupart des endroits devraient rester secs. ")
                + ("Orages possibles. " if thunder else "")
                + (f"Risque élevé de crues soudaines à {', '.join(flood)}. " if flood else ""))
        line = "{parish} : pluie {level}{when} ({p:.0%} de probabilité). L'endroit le plus arrosé pourrait recevoir {mm:.0f} mm."
        since = " à partir d'environ {t}"
    lines = [head, ""]
    for r in rows:
        when = since.format(t=r["earliest_likely_start"]) if r.get("earliest_likely_start") else ""
        lines.append(line.format(parish=r["parish"], level=lv[_level(r["any_rain_24h_typical"])], when=when,
                                 p=r["any_rain_24h_typical"], mm=r["rain_24h_mm_p90_highest"]))
    return "\n".join(lines) + "\n"


def languages_of(country: str) -> tuple:
    """A country's bulletin languages: GeoNames' official languages that Pluvium writes, else a built-in table."""
    try:
        from ..area import country_info, languages_for

        info = country_info()
        if info:
            return languages_for(country, info)
    except Exception:
        pass
    return KNOWN_LANGUAGES.get(country, ("en",))


def bulletins(parishes: list[dict], country: str, issued: str, languages=None) -> dict:
    """``{file name: text}``: ``bulletin_<CC>.txt`` in the first language, ``bulletin_<CC>_<language>.txt`` for the
    others (Jamaican Patois keeps its name, ``bulletin_JM_patois.txt``)."""
    langs = tuple(languages or languages_of(country))
    out = {}
    for k, lang in enumerate(langs):
        suffix = "" if k == 0 else "_patois" if lang == "jam" else f"_{lang}"
        out[f"bulletin_{country}{suffix}.txt"] = bulletin(parishes, country, issued, lang)
    return out


__all__ = ["bulletin", "bulletins", "languages_of"]
