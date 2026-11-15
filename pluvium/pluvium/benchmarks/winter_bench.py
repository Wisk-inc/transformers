# Copyright 2026 Nathan. Apache-2.0.
"""
Winter, scored twice: against the satellite and against radar and gauges.

The first full-year benchmark found Pluvium Blend behind post-processed ECMWF ENS in the Great Plains winter. The
truth there is CMORPH, which is weak on frozen precipitation. Two things could be true: the Blend is worse in winter,
or the truth is. This benchmark scores the same forecasts for the same days against both CMORPH and MRMS
(:mod:`pluvium.sources.mrms_daily`), for a US place whose store and kept chances exist.

For a season and each lead group it reports, for Pluvium Blend and every post-processed component, the Brier skill
score against each truth's own base rate. Everything was trained on CMORPH, so against MRMS every model carries the
same handicap; what matters is whether the order between them changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import blend as B
from ..sources.mrms_daily import MRMSDaily

LEAD_GROUPS = {"days 1-3": (1, 3), "days 4-7": (4, 7), "days 8-10": (8, 10), "days 11-14": (11, 14)}
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def month_of(hour: int) -> int:
    return int(str(np.datetime64(int(hour), "h"))[5:7])


def skill(p: np.ndarray, event: np.ndarray) -> float:
    """Brier skill against the base rate of the rows given."""
    base = float(event.mean())
    if not 0 < base < 1:
        return float("nan")
    return float(1.0 - np.mean((p - event) ** 2) / (base * (1 - base)))


def run(store, keep_blocks: np.ndarray, keep_p: np.ndarray, mrms: MRMSDaily, season: str = "DJF",
        thresholds=(1.0, 10.0), download: bool = True, min_cells: int = 50) -> dict:
    """Score Blend and the components of ``store`` for one season against CMORPH and MRMS on the same rows."""
    at = {int(b): k for k, b in enumerate(keep_blocks)}
    names = list(store.names)
    rows = {g: {"cmorph": [], "mrms": [], **{m: [] for m in ["stack"] + names}} for g in LEAD_GROUPS}
    inside = mrms.covered()
    days = set()
    for q in store.quarters():
        d = store.quarter(q)
        for r, i in enumerate(d["test"]):
            i = int(i)
            lead = int(store.lead[i])
            valid = int(store.run[i]) + 24 * (lead - 1)
            if month_of(valid) not in SEASONS[season] or i not in at:
                continue
            radar = mrms.day(valid, download)
            if radar is None:
                continue
            ok = inside & np.isfinite(radar) & np.isfinite(store.y[i])
            if ok.sum() < min_cells:
                continue
            days.add(valid)
            g = next(name for name, (lo, hi) in LEAD_GROUPS.items() if lo <= lead <= hi)
            rows[g]["cmorph"].append(store.y[i][ok])
            rows[g]["mrms"].append(radar[ok])
            rows[g]["stack"].append(keep_p[at[i]][ok].astype(np.float32))
            z = d["test_z"][r].astype(np.float32)
            for k, name in enumerate(names):
                p = 1.0 / (1.0 + np.exp(-z[ok, k])) if d["test_present"][r, k] else np.full((int(ok.sum()), z.shape[2]), np.nan, np.float32)
                rows[g][name].append(p)
    out = {"region": store.folder.name, "season": season, "days": len(days), "groups": {}}
    for g, acc in rows.items():
        if not acc["cmorph"]:
            continue
        truth = {"cmorph": np.concatenate(acc["cmorph"]), "mrms": np.concatenate(acc["mrms"])}
        res = {"rows": int(truth["cmorph"].size),
               "wet_share": {k: {f"{t:g}": float(np.mean(v >= t)) for t in thresholds} for k, v in truth.items()},
               "truths_agree": {f"{t:g}": float(np.mean((truth["cmorph"] >= t) == (truth["mrms"] >= t))) for t in thresholds},
               "skill": {}}
        for m in ["stack"] + names:
            p = np.concatenate(acc[m])
            have = np.isfinite(p[:, 0])
            if have.sum() < 100:
                continue
            res["skill"][m] = {k: {f"{t:g}": skill(B.exceed(p[have], t), (v[have] >= t).astype(np.float64)) for t in thresholds}
                               for k, v in truth.items()}
            res["skill"][m]["rows"] = int(have.sum())
        out["groups"][g] = res
    return out


def markdown(result: dict, labels: dict | None = None) -> str:
    """Blend against the components, by lead group: skill against CMORPH / against MRMS."""
    labels = {"stack": "Pluvium Blend", "blend_ops": "One network over the six", **(labels or {})}
    groups = [g for g in LEAD_GROUPS if g in result["groups"]]
    lines = [f"{result['region'].replace('_', ' ').title()}, {result['season']}: {result['days']} days. Each cell: Brier "
             "skill against CMORPH / against MRMS, each against its own base rate. Bold is the best in the column for "
             "that truth.", ""]
    for t in next(iter(result["groups"].values()))["wet_share"]["cmorph"]:
        lines += [f"**{t} mm or more**", "", "| System | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
        models = list(next(iter(result["groups"].values()))["skill"])
        best = {g: {k: max((result["groups"][g]["skill"][m][k][t] for m in models if m in result["groups"][g]["skill"]),
                           default=float("nan")) for k in ("cmorph", "mrms")} for g in groups}
        for m in models:
            cells = []
            for g in groups:
                s = result["groups"][g]["skill"].get(m)
                if not s:
                    cells.append("-")
                    continue
                parts = [f"**{s[k][t]:+.3f}**" if s[k][t] == best[g][k] else f"{s[k][t]:+.3f}" for k in ("cmorph", "mrms")]
                cells.append(" / ".join(parts))
            lines.append(f"| {labels.get(m, m)} | " + " | ".join(cells) + " |")
        wet = " | ".join(f"{100 * result['groups'][g]['wet_share']['cmorph'][t]:.1f}% / "
                         f"{100 * result['groups'][g]['wet_share']['mrms'][t]:.1f}%" for g in groups)
        agree = " | ".join(f"{100 * result['groups'][g]['truths_agree'][t]:.0f}%" for g in groups)
        lines += [f"| Share of rows with the rain: CMORPH / MRMS | {wet} |", f"| Rows where the two truths agree | {agree} |", ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    import argparse

    from ..frontier.store import Store

    p = argparse.ArgumentParser(prog="pluvium winter-bench", description="Winter against CMORPH and MRMS.")
    p.add_argument("--store", required=True)
    p.add_argument("--keep", required=True)
    p.add_argument("--cache", required=True, help="folder for the MRMS days")
    p.add_argument("--season", default="DJF")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    store = Store(a.store)
    d = np.load(a.keep)
    result = run(store, d["blocks"], d["p"], MRMSDaily(a.cache, store.lat, store.lon, store.folder.name), a.season)
    Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
