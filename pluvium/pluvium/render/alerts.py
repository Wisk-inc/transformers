# Copyright 2026 Nathan. Apache-2.0.
"""
Alerts in the Common Alerting Protocol (OASIS CAP 1.2).

Caribbean meteorological services and disaster offices exchange warnings as CAP messages. For each parish where a
threshold is met, an alert is written with the event (heavy rain, thunderstorms, flash-flood risk), severity,
certainty from the probability (likely above 50%, possible above 20%), the expected onset in UTC, an area described
by the parish name and the bounding circle of its places, and the forecast sentence as the description. These are
advisory products of a research model: the ``status`` is "Exercise" unless an operator sets it otherwise, so they are
never mistaken for official warnings.
"""

from __future__ import annotations

import uuid
from xml.sax.saxutils import escape

import numpy as np


def _certainty(p: float) -> str:
    return "Likely" if p >= 0.5 else "Possible" if p >= 0.2 else "Unlikely"


def cap_alerts(parishes: list[dict], towns: list[dict], issued_iso: str, status: str = "Exercise",
               sender: str = "pluvium@corx-labs.example") -> list[str]:
    """CAP XML strings, one per parish and hazard that meets its threshold."""
    out = []
    for r in parishes:
        places = [t for t in towns if t["country"] == r["country"] and t["parish"] == r["parish"]]
        if not places:
            continue
        lat = np.array([t["lat"] for t in places])
        lon = np.array([t["lon"] for t in places])
        c_lat, c_lon = float(lat.mean()), float(lon.mean())
        radius = float(max(5.0, 111.195 * np.max(np.hypot(lat - c_lat, (lon - c_lon) * np.cos(np.radians(c_lat))))))
        hazards = []
        heavy = max((t.get("heavy_rain_24h") or 0.0) for t in places)
        if heavy >= 0.2:
            hazards.append(("Heavy rain", "Moderate" if heavy < 0.5 else "Severe", heavy))
        thunder = r.get("thunder_24h_highest", 0.0)
        if thunder >= 0.3:
            hazards.append(("Thunderstorms", "Moderate", thunder))
        level = r.get("flood_highest", "low")
        if level in ("moderate", "high", "very high"):
            hazards.append(("Flash flood risk", {"moderate": "Moderate", "high": "Severe", "very high": "Extreme"}[level],
                            0.5 if level == "moderate" else 0.7))
        for event, severity, p in hazards:
            out.append(f"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>pluvium-{uuid.uuid4()}</identifier>
  <sender>{escape(sender)}</sender>
  <sent>{issued_iso}</sent>
  <status>{status}</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>en</language>
    <category>Met</category>
    <event>{escape(event)}</event>
    <urgency>Expected</urgency>
    <severity>{severity}</severity>
    <certainty>{_certainty(p)}</certainty>
    <senderName>Pluvium research forecast (not an official warning)</senderName>
    <headline>{escape(event)} for {escape(r['parish'])}, {escape(r['country'])}</headline>
    <description>{escape(r.get('summary', ''))}</description>
    <instruction>Follow the advice of your national meteorological service and disaster office.</instruction>
    <area>
      <areaDesc>{escape(r['parish'])}, {escape(r['country'])}</areaDesc>
      <circle>{c_lat:.3f},{c_lon:.3f} {radius:.1f}</circle>
    </area>
  </info>
</alert>""")
    return out


__all__ = ["cap_alerts"]
