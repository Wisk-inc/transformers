# Copyright 2026 Nathan. Apache-2.0.
"""
From chances to warnings: three tiers for heavy rain, and what they have meant.

A chance of 50 mm of rain is the right thing to forecast and an awkward thing to act on. Tiers turn it into a message:

* **watch**: at least a 10% chance of the amount;
* **warning**: at least 30%;
* **extreme**: at least 60%.

A tier is only worth issuing if people know what it has meant. :func:`verify` reports, for a set of past forecasts,
how often each tier was issued, how often the rain then came (its hit rate), and what share of all heavy-rain cases
it caught. A separate flag marks forecasts **beyond the record**: the raw systems call for more than anything in the
training period, so the amount is a lower bound and the calibrated chance has been handed over
(:mod:`pluvium.tails.override`).
"""

from __future__ import annotations

import numpy as np

from ..blend import exceed

TIERS = (("watch", 0.10), ("warning", 0.30), ("extreme", 0.60))
WORDS = {0: "", 1: "Heavy rain possible", 2: "Heavy rain likely", 3: "Extreme rain expected"}


def tier(p: np.ndarray, amount: float = 50.0, tiers: tuple = TIERS) -> np.ndarray:
    """The tier per row, 0 (none) to ``len(tiers)``, from chances ``p`` (N, K) of exceeding the thresholds."""
    chance = exceed(p, amount)
    out = np.zeros(chance.shape[0], np.int8)
    for k, (_, cut) in enumerate(tiers):
        out[chance >= cut] = k + 1
    return out


def wording(level: int, beyond_record: bool = False, amount: float = 50.0) -> str:
    """A line for a forecast table: the tier in words, and a note when the forecast is beyond the record."""
    text = WORDS.get(int(level), "")
    if text:
        text += f" ({amount:g} mm or more)"
    if beyond_record:
        text += ("; " if text else "") + "beyond anything in the training period: treat amounts as a lower bound"
    return text


def verify(levels: np.ndarray, y: np.ndarray, amount: float = 50.0, tiers: tuple = TIERS) -> dict:
    """What each tier meant on past forecasts: ``{tier: {"issued", "share", "hit_rate", "caught"}}``.

    ``issued`` counts forecasts at that tier or higher, ``hit_rate`` the share of them with at least ``amount`` mm,
    ``caught`` the share of all such cases they covered. ``base_rate`` is the share of all forecasts with the rain."""
    ok = np.isfinite(y)
    levels, event = np.asarray(levels)[ok], np.asarray(y)[ok] >= amount
    out = {"forecasts": int(ok.sum()), "events": int(event.sum()), "base_rate": float(event.mean()) if ok.any() else float("nan")}
    for k, (name, _) in enumerate(tiers):
        at = levels >= k + 1
        out[name] = {"issued": int(at.sum()), "share": float(at.mean()) if ok.any() else float("nan"),
                     "hit_rate": float(event[at].mean()) if at.any() else float("nan"),
                     "caught": float((event & at).sum() / event.sum()) if event.any() else float("nan")}
    return out


def reliability(p: np.ndarray, y: np.ndarray, amount: float = 50.0,
                bins=(0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001)) -> list:
    """Chance given against how often it happened: ``[{"from", "to", "n", "forecast", "observed"}]``."""
    ok = np.isfinite(y)
    chance, event = exceed(p[ok], amount), np.asarray(y)[ok] >= amount
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (chance >= lo) & (chance < hi)
        if m.sum():
            out.append({"from": lo, "to": min(hi, 1.0), "n": int(m.sum()), "forecast": float(chance[m].mean()),
                        "observed": float(event[m].mean())})
    return out


__all__ = ["TIERS", "WORDS", "reliability", "tier", "verify", "wording"]
