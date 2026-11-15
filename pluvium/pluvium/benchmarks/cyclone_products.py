# Copyright 2026 Nathan. Apache-2.0.
"""
Are the uncertainty products honest? The cone, the passage chances and the scenarios, verified.

The track benchmark (:mod:`.cyclone_bench`) scores where the consensus puts the centre. This one scores what Pluvium
Cyclone says about its own uncertainty, season by season, with everything learned causally:

* **cone**: how often the centre was inside the two-in-three circle (it should be two times in three), and how large
  the circle was, for the static cone and for the one that follows the systems' spread;
* **passage chances**: the chance given that the centre passes within 150 km of a place in the next five days,
  for every place within 2,000 km of the storm, against whether it did. Scored with the Brier skill score against the
  season's base rate, for the mixture cloud (:func:`pluvium.cyclone.cloud.mixture_cloud`), the same cloud resized to
  the errors verified so far, the raw member cloud, and a yes/no read off the consensus track;
* **scenarios**: how often the cloud split, and whether the truth then lay nearer a scenario than the consensus.
"""

from __future__ import annotations

import numpy as np

from ..cyclone.approach import observed_passage
from ..cyclone.basins import BASINS
from ..cyclone.geodesy import distance_km
from ..cyclone.places import PLACES
from ..cyclone.products import Forecaster
from ..cyclone.verify import LEADS, case_errors

BINS = (0.0, 0.02, 0.1, 0.25, 0.5, 0.75, 1.0001)


def brier_skill(p: np.ndarray, y: np.ndarray) -> float:
    base = float(y.mean())
    if not 0 < base < 1:
        return float("nan")
    return float(1.0 - np.mean((p - y) ** 2) / (base * (1 - base)))


def reliability(p: np.ndarray, y: np.ndarray, bins=BINS) -> list:
    """``[{"from", "to", "n", "forecast", "observed"}]`` per bin of forecast chance."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            out.append({"from": lo, "to": min(hi, 1.0), "n": int(m.sum()), "forecast": float(p[m].mean()),
                        "observed": float(y[m].mean())})
    return out


def evaluate_products(by_season: dict, systems, learner: dict, radius_km: float = 150.0, horizon: int = 120,
                      sized: bool = True) -> dict:
    """Step one :class:`Forecaster` through every season's cases in time order and score its products per season.

    ``by_season`` maps a season (with ``label``, ``role`` and ``archive``) to its cases."""
    forecaster = Forecaster(list(systems), learner, radius_km, horizon, sized=sized)
    season_of = {id(c): s for s, cases in by_season.items() for c in cases}
    rows = {s.label: {"cone": [], "passage": [], "split": []} for s in by_season}
    for case in sorted((c for cases in by_season.values() for c in cases), key=lambda c: (c.init, c.storm.id)):
        season = season_of[id(case)]
        basin = season.archive if season.archive in PLACES else None
        g = forecaster.step(case, basin)
        acc = rows[season.label]
        errors = case_errors(g.track, case.storm.track, case.init)
        for lead, km in errors.items():
            acc["cone"].append((lead, km, g.cone[lead], g.cone_static[lead], g.spread[lead]))
        for name, v in g.places.items():
            plat, plon = PLACES[basin][name]
            hit = observed_passage(case.storm.track, case.init, plat, plon, radius_km, horizon)
            if hit is not None and np.isfinite(v["chance"]):
                yes = float(np.isfinite(v["track_km"]) and v["track_km"] <= radius_km)
                acc["passage"].append((v["chance"], v["raw_chance"], yes, float(hit), v["sized_chance"],
                                       v["mixture_chance"]))
        if len(g.scenarios) > 1:
            lead = max((k for k in errors if all(k in s.points for s in g.scenarios)), default=None)
            if lead is not None:
                obs = case.storm.track.position(case.init + lead)
                away = [float(distance_km(s.points[lead][0], s.points[lead][1], obs[0], obs[1])) for s in g.scenarios]
                acc["split"].append((lead, errors[lead], min(away), g.scenarios[0].weight,
                                     float(int(np.argmin(away)) == 0)))
        elif errors:
            acc["split"].append(None)
    out = {}
    for label, acc in rows.items():
        cone = np.array(acc["cone"], np.float64).reshape(-1, 5)
        by_lead = {}
        for lead in LEADS:
            m = cone[:, 0] == lead
            if m.sum() >= 5:
                by_lead[str(lead)] = {
                    "n": int(m.sum()),
                    "coverage": float(np.mean(cone[m, 1] <= cone[m, 2])),
                    "coverage_static": float(np.mean(cone[m, 1] <= cone[m, 3])),
                    "radius_nm": float(cone[m, 2].mean() / 1.852),
                    "radius_static_nm": float(cone[m, 3].mean() / 1.852),
                    "spread_error_corr": float(np.corrcoef(cone[m, 4], cone[m, 1])[0, 1])
                    if np.isfinite(cone[m, 4]).all() and m.sum() >= 10 else float("nan")}
        res = {"cone": by_lead}
        if cone.size:
            res["cone_all"] = {"coverage": float(np.mean(cone[:, 1] <= cone[:, 2])),
                               "coverage_static": float(np.mean(cone[:, 1] <= cone[:, 3])), "n": int(cone.shape[0])}
        if acc["passage"]:
            p = np.array(acc["passage"], np.float64)
            res["passage"] = {"n": int(p.shape[0]), "events": int(p[:, 3].sum()), "base_rate": float(p[:, 3].mean()),
                              "bss": brier_skill(p[:, 0], p[:, 3]), "bss_raw": brier_skill(p[:, 1], p[:, 3]),
                              "bss_sized": brier_skill(p[:, 4], p[:, 3]), "bss_mixture": brier_skill(p[:, 5], p[:, 3]),
                              "bss_track": brier_skill(p[:, 2], p[:, 3]), "reliability": reliability(p[:, 0], p[:, 3]),
                              "reliability_raw": reliability(p[:, 1], p[:, 3])}
        splits = [s for s in acc["split"] if s is not None]
        res["scenarios"] = {"cases": len(acc["split"]), "split": len(splits)}
        if splits:
            s = np.array(splits, np.float64)
            res["scenarios"].update({"consensus_nm": float(s[:, 1].mean() / 1.852),
                                     "nearest_scenario_nm": float(s[:, 2].mean() / 1.852),
                                     "main_weight": float(s[:, 3].mean()), "main_verified": float(s[:, 4].mean())})
        out[label] = res
    out["slope"] = forecaster.cone.slope()
    out["basins"] = {s.label: BASINS[s.archive].name for s in by_season}
    return out


def products_markdown(products: dict, roles: dict) -> list:
    lines = ["## The uncertainty products", "",
             "Everything below was learned causally, from forecasts that had verified before each forecast was made. "
             f"The cone's dependence on spread ended at a slope of {products['slope']:.2f} (0 would be a static cone).", ""]
    order = sorted((k for k in products if k in roles), key=lambda k: (roles[k] != "test", k))
    for label in order:
        p = products[label]
        lines += [f"### {label} ({roles[label]})", ""]
        if p.get("cone"):
            lines += ["**Cone.** Share of forecasts whose centre fell inside the two-in-three circle (target 67%), and "
                      "the circle's mean radius in nautical miles.", "",
                      "| | " + " | ".join(f"{k} h" for k in LEADS) + " |", "|---|" + "---|" * len(LEADS)]
            for title, a, b in (("Inside, cone that follows spread", "coverage", None),
                                ("Inside, static cone", "coverage_static", None),
                                ("Radius, cone that follows spread", "radius_nm", 1),
                                ("Radius, static cone", "radius_static_nm", 1),
                                ("Correlation of spread with error", "spread_error_corr", 2)):
                cells = []
                for lead in LEADS:
                    v = p["cone"].get(str(lead))
                    if not v or not np.isfinite(v[a]):
                        cells.append("-")
                    else:
                        cells.append(f"{100 * v[a]:.0f}%" if b is None else (f"{v[a]:.0f}" if b == 1 else f"{v[a]:+.2f}"))
                lines.append(f"| {title} | " + " | ".join(cells) + " |")
            lines.append("")
        if p.get("passage"):
            q = p["passage"]
            lines += [f"**Passage chances.** {q['n']} forecasts for places within 2,000 km; the centre came within "
                      f"150 km in {q['events']} ({100 * q['base_rate']:.1f}%). Brier skill against that base rate: "
                      f"Pluvium's cloud {q['bss']:+.3f} (the mixture before resizing {q['bss_mixture']:+.3f}), "
                      f"raw member cloud {q['bss_raw']:+.3f}, yes/no from the consensus track {q['bss_track']:+.3f}.", "",
                      "| Chance given | Forecasts | Mean chance | Happened |", "|---|---|---|---|"]
            for r in q["reliability"]:
                lines.append(f"| {100 * r['from']:.0f}-{100 * r['to']:.0f}% | {r['n']} | {100 * r['forecast']:.0f}% | "
                             f"{100 * r['observed']:.0f}% |")
            lines.append("")
        s = p.get("scenarios", {})
        if s.get("cases"):
            text = f"**Scenarios.** The cloud split into more than one scenario in {s['split']} of {s['cases']} forecasts."
            if s.get("split"):
                text += (f" In those, the main scenario held {100 * s['main_weight']:.0f}% of the weight on average and "
                         f"was the one nearest the truth {100 * s['main_verified']:.0f}% of the time. The consensus "
                         f"was {s['consensus_nm']:.0f} nm off at the longest lead; the nearest scenario's mean track "
                         f"{s['nearest_scenario_nm']:.0f} nm (the nearest of several is flattered by being picked "
                         f"afterwards).")
            lines += [text, ""]
    return lines


__all__ = ["BINS", "brier_skill", "evaluate_products", "products_markdown", "reliability"]
