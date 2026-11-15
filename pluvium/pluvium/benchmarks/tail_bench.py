# Copyright 2026 Nathan. Apache-2.0.
"""
The heavy end of the forecast, scored on its own.

From a place's store and the chances Pluvium Blend gave for the test year (kept by
:func:`pluvium.frontier.evaluate.evaluate`), by lead group:

* **tail CRPS**: the threshold-weighted CRPS above 25 and 50 mm (:mod:`pluvium.tails.twcrps`) for Pluvium Blend and
  for the raw consensus of the systems (one vote each), and Blend's skill against that consensus;
* **Brier scores** at 50 and 100 mm for both;
* **warnings**: how often each tier was issued for 50 mm, how often the rain came, and what share of all 50 mm cases
  it caught (:mod:`pluvium.tails.alerts`);
* **reliability** at 50 mm: the chance given against how often it happened;
* **case days**: on named days (a hurricane's landfall, say) the chance and amount by lead, against what fell.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import blend as B
from ..tails import alerts
from ..tails.twcrps import tw_crps
from .fullyear import place_name

LEAD_GROUPS = {"days 1-3": (1, 3), "days 4-7": (4, 7), "days 8-10": (8, 10), "days 11-14": (11, 14)}


def _day(hour: int) -> str:
    return str(np.datetime64(int(hour), "h"))[:10]


def run(store, keep_blocks: np.ndarray, keep_p: np.ndarray, amount: float = 50.0, case_days=()) -> dict:
    """Tail scores for the kept test blocks of ``store``."""
    sums = {g: {"n": 0, **{f"{who}_{k}": 0.0 for who in ("blend", "raw") for k in ("tw25", "tw50", "bs50", "bs100")}}
            for g in LEAD_GROUPS}
    levels = {g: [] for g in LEAD_GROUPS}
    truth = {g: [] for g in LEAD_GROUPS}
    chance = {g: [] for g in LEAD_GROUPS}
    cases = []
    for k, i in enumerate(keep_blocks):
        i = int(i)
        y = store.y[i]
        ok = np.isfinite(y)
        if ok.sum() == 0:
            continue
        lead = int(store.lead[i])
        g = next(name for name, (lo, hi) in LEAD_GROUPS.items() if lo <= lead <= hi)
        p, raw = keep_p[k].astype(np.float32), store.raw_p(i)
        acc = sums[g]
        acc["n"] += int(ok.sum())
        for who, q in (("blend", p), ("raw", raw)):
            acc[f"{who}_tw25"] += float(np.nansum(tw_crps(q, y, 25.0)))
            acc[f"{who}_tw50"] += float(np.nansum(tw_crps(q, y, 50.0)))
            acc[f"{who}_bs50"] += float(((B.exceed(q[ok], 50.0) - (y[ok] >= 50.0)) ** 2).sum())
            acc[f"{who}_bs100"] += float(((B.exceed(q[ok], 100.0) - (y[ok] >= 100.0)) ** 2).sum())
        levels[g].append(alerts.tier(p[ok], amount))
        truth[g].append(y[ok])
        chance[g].append(B.exceed(p[ok], amount).astype(np.float32))
        valid = _day(int(store.run[i]) + 24 * (lead - 1))
        if valid in case_days:
            cases.append({"valid": valid, "run": _day(int(store.run[i])), "lead": lead,
                          "observed_mean_mm": float(y[ok].mean()), "observed_max_mm": float(y[ok].max()),
                          "observed_share_50": float(np.mean(y[ok] >= 50.0)), "observed_share_100": float(np.mean(y[ok] >= 100.0)),
                          "blend_p50": float(B.exceed(p[ok], 50.0).mean()), "blend_p100": float(B.exceed(p[ok], 100.0).mean()),
                          "blend_mean_mm": float(B.quantiles(p[ok]).mean()),
                          "raw_p50": float(B.exceed(raw[ok], 50.0).mean()), "raw_p100": float(B.exceed(raw[ok], 100.0).mean()),
                          "raw_mean_mm": float(store.raw_mean[i][ok].mean()),
                          "extreme_share": float(np.mean(alerts.tier(p[ok], amount) == 3)),
                          "warning_share": float(np.mean(alerts.tier(p[ok], amount) >= 2))})
    out = {"region": store.folder.name, "amount_mm": amount, "groups": {}, "cases": cases}
    for g, acc in sums.items():
        if acc["n"] == 0:
            continue
        n = acc["n"]
        res = {key: acc[key] / n for key in acc if key != "n"}
        res["rows"] = n
        for key in ("tw25", "tw50", "bs50", "bs100"):
            ref = res[f"raw_{key}"]
            res[f"skill_{key}"] = float(1.0 - res[f"blend_{key}"] / ref) if ref > 0 else float("nan")
        y_all = np.concatenate(truth[g])
        res["alerts"] = alerts.verify(np.concatenate(levels[g]), y_all, amount)
        ch = np.concatenate(chance[g])
        rel = []
        bins = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001)
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (ch >= lo) & (ch < hi)
            if m.sum():
                rel.append({"from": lo, "to": min(hi, 1.0), "n": int(m.sum()), "forecast": float(ch[m].mean()),
                            "observed": float((y_all[m] >= amount).mean())})
        res["reliability"] = rel
        out["groups"][g] = res
    return out


def markdown(results: list) -> str:
    """Tail skill against the raw consensus and what the warnings meant, for every place."""
    lines = ["**Skill in the tail against the raw consensus of the six systems** (threshold-weighted CRPS above 25 mm; "
             "1 is perfect, 0 no better than the raw consensus)", "",
             "| Place | " + " | ".join(LEAD_GROUPS) + " |", "|---|" + "---|" * len(LEAD_GROUPS)]
    for r in results:
        cells = [f"{r['groups'][g]['skill_tw25']:+.3f}" if g in r["groups"] else "-" for g in LEAD_GROUPS]
        lines.append(f"| {place_name(r['region'])} | " + " | ".join(cells) + " |")
    lines += ["", "**What a warning for 50 mm meant, days 1-3** (issued: share of all place-days; came: how often 50 mm "
              "or more then fell; caught: share of all 50 mm cases covered)", "",
              "| Place | 50 mm days | Watch (10%+): issued / came / caught | Warning (30%+) | Extreme (60%+) |", "|---|---|---|---|---|"]
    for r in results:
        a = r["groups"].get("days 1-3", {}).get("alerts")
        if not a:
            continue
        cells = []
        for name in ("watch", "warning", "extreme"):
            t = a[name]
            cells.append("never issued" if t["issued"] == 0 else
                         f"{100 * t['share']:.2f}% / {100 * t['hit_rate']:.0f}% / {100 * t['caught']:.0f}%")
        lines.append(f"| {place_name(r['region'])} | {100 * a['base_rate']:.2f}% | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def case_table(result: dict, day: str) -> list:
    """One row per lead for a case day: Blend's and the raw consensus's chance of 50 and 100 mm, and amounts."""
    rows = sorted((c for c in result["cases"] if c["valid"] == day), key=lambda c: c["lead"])
    if not rows:
        return []
    o = rows[0]
    lines = [f"Observed on {day}: {o['observed_mean_mm']:.0f} mm on average, up to {o['observed_max_mm']:.0f} mm; "
             f"{100 * o['observed_share_50']:.0f}% of places had 50 mm or more, {100 * o['observed_share_100']:.0f}% had "
             "100 mm or more.", "",
             "| Days ahead | Blend: chance of 50 mm | of 100 mm | expected mm | places under an extreme warning | "
             "Raw consensus: chance of 50 mm | of 100 mm | mean mm |", "|---|---|---|---|---|---|---|---|"]
    for c in rows:
        lines.append(f"| {c['lead']} | {100 * c['blend_p50']:.0f}% | {100 * c['blend_p100']:.0f}% | {c['blend_mean_mm']:.0f} | "
                     f"{100 * c['extreme_share']:.0f}% | {100 * c['raw_p50']:.0f}% | {100 * c['raw_p100']:.0f}% | "
                     f"{c['raw_mean_mm']:.0f} |")
    return lines


def main(argv=None) -> None:
    import argparse

    from ..frontier.store import Store

    p = argparse.ArgumentParser(prog="pluvium tail-bench", description="Tail scores from a store and kept chances.")
    p.add_argument("--store", required=True)
    p.add_argument("--keep", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--case-day", action="append", default=[])
    a = p.parse_args(argv)
    d = np.load(a.keep)
    Path(a.out).write_text(json.dumps(run(Store(a.store), d["blocks"], d["p"], case_days=tuple(a.case_day)), indent=1))


if __name__ == "__main__":
    main()
