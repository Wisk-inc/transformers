# Copyright 2026 Nathan. Apache-2.0.
"""
Do the scenarios keep the weather's shape? Coupled scenarios against raw ensembles and uncoupled draws.

A calibrated chance per place is smooth by nature: at day 12 nobody knows where the rain band will lie, so every place
gets a modest chance and the map of expected amounts looks washed out. That is honest for a single place and wrong as
a picture of a day: on the day, the band is somewhere, with sharp edges. Scenarios restore the picture. Ensemble copula
coupling (:mod:`pluvium.scenarios.ecc`) gives every scenario the calibrated amounts place by place and the raw
ensemble's arrangement in space.

This benchmark takes Pluvium Blend's chances for the test year (kept by :func:`pluvium.frontier.evaluate.evaluate`)
and ECMWF ENS as the template, and scores three sets of scenarios with the same number of members:

* **raw**: the ENS members as they come;
* **coupled**: Pluvium's amounts in the ENS arrangement;
* **uncoupled**: Pluvium's amounts with no arrangement (each place drawn on its own).

Scores, by lead group:

* **sharpness of one scenario**: the spatial variance of a scenario over the observed spatial variance, as a ratio of
  standard deviations (1 is a scenario as varied as the weather; the benchmark's sharpness of the *mean* map is well
  below 1 at long leads by construction);
* **CRPS** place by place (coupled and uncoupled are equal by construction: same amounts);
* **energy score** and **variogram score**, which reward the right joint behaviour of nearby places;
* **areal CRPS**: the CRPS of the mean rain over boxes two degrees wide, which only scenarios with the right
  structure get right.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..scenarios.ecc import ecc, independent
from ..scenarios.scores import areal_crps, box_groups, crps_ensemble, energy_score, nearby_pairs, variogram_score
from ..util import log

LEAD_GROUPS = {"days 1-3": (1, 3), "days 4-7": (4, 7), "days 8-10": (8, 10), "days 11-14": (11, 14)}
KINDS = ("raw", "coupled", "uncoupled")


def members_at(z: dict, lead: int, lat: np.ndarray, lon: np.ndarray) -> np.ndarray | None:
    """A system's members at places for one lead day, ``(M, N)`` mm (members with gaps dropped)."""
    from ..dataset import bilinear
    from .fullyear import daily_members

    day = daily_members(z, lead)
    if day is None:
        return None
    finite = np.isfinite(day).reshape(day.shape[0], -1).all(1)
    if finite.sum() < 5:
        return None
    return bilinear(day[finite], np.asarray(z["lat"], np.float64), np.asarray(z["lon"], np.float64), lat, lon)


class Accumulator:
    """Sums of every score for one kind of scenario and one lead group."""

    def __init__(self) -> None:
        self.n = 0
        self.sums = {"crps": 0.0, "energy": 0.0, "variogram": 0.0, "areal": 0.0, "var": 0.0, "obs_var": 0.0}
        self.areas = 0

    def add(self, scenarios: np.ndarray, obs: np.ndarray, pairs: np.ndarray, groups: np.ndarray) -> None:
        ok = np.isfinite(obs)
        if ok.sum() < 20:
            return
        self.n += 1
        self.sums["crps"] += float(crps_ensemble(scenarios[:, ok], obs[ok]).mean())
        self.sums["energy"] += energy_score(scenarios, obs) / np.sqrt(ok.sum())
        self.sums["variogram"] += variogram_score(scenarios, obs, pairs)
        area = areal_crps(scenarios, obs, groups)
        good = np.isfinite(area)
        self.sums["areal"] += float(area[good].sum())
        self.areas += int(good.sum())
        self.sums["var"] += float(np.var(scenarios[:, ok], axis=1).mean())
        self.sums["obs_var"] += float(np.var(obs[ok]))

    def result(self) -> dict:
        if self.n == 0:
            return {}
        return {"days": self.n, "crps": self.sums["crps"] / self.n, "energy": self.sums["energy"] / self.n,
                "variogram": self.sums["variogram"] / self.n, "areal_crps": self.sums["areal"] / max(self.areas, 1),
                "sharpness": float(np.sqrt(self.sums["var"] / max(self.sums["obs_var"], 1e-12)))}


def run(store, keep_blocks: np.ndarray, keep_p: np.ndarray, reader, region: str, system: str = "ens",
        every: int = 2, max_members: int = 50, seed: int = 0, pair_km: float = 150.0) -> dict:
    """Score the three kinds of scenarios on the kept test blocks of ``store``: every ``every``-th of the runs that
    reach the longest leads, so every lead group is covered.

    ``keep_blocks`` and ``keep_p`` are the block indices and chances ``(B, n, K)`` kept by the evaluation; ``reader``
    is the :class:`pluvium.sources.archive.ArchiveReader` the store was built from and ``region`` its region name."""
    rng = np.random.default_rng(seed)
    lat, lon = np.asarray(store.lat, np.float64), np.asarray(store.lon, np.float64)
    pairs = nearby_pairs(lat, lon, pair_km, rng=rng)
    groups = box_groups(lat, lon, 2.0)
    at = {int(b): k for k, b in enumerate(keep_blocks)}
    long_runs = np.unique(store.run[keep_blocks][store.lead[keep_blocks] > 3])   # runs that reach beyond day 3
    runs = (long_runs if long_runs.size else np.unique(store.run[keep_blocks]))[::every]
    acc = {g: {kind: Accumulator() for kind in KINDS} for g in LEAD_GROUPS}
    for k, run_hour in enumerate(runs):
        z = reader.load(region, system, int(run_hour))
        if z is None:
            continue
        for i in np.flatnonzero(store.run == run_hour):
            if int(i) not in at:
                continue
            lead = int(store.lead[i])
            members = members_at(z, lead, lat, lon)
            if members is None:
                continue
            members = members[:max_members]
            p = keep_p[at[int(i)]].astype(np.float32)
            sets = {"raw": members, "coupled": ecc(p, members, rng=rng),
                    "uncoupled": independent(p, members.shape[0], rng)}
            group = next(g for g, (lo, hi) in LEAD_GROUPS.items() if lo <= lead <= hi)
            for kind, scenarios in sets.items():
                acc[group][kind].add(scenarios, store.y[i], pairs, groups)
        if k % 20 == 0:
            log(f"  scenarios {store.folder.name}: {k + 1} of {runs.size} runs")
    return {"region": store.folder.name, "system": system, "pairs": int(pairs.shape[0]),
            "areas": int(groups.max() + 1), "groups": {g: {kind: a.result() for kind, a in kinds.items()}
                                                         for g, kinds in acc.items()}}


def markdown(results: list) -> str:
    """One table per score, places as rows and lead groups as columns: raw / coupled / uncoupled."""
    titles = {"sharpness": "Sharpness of one scenario (1 = as varied as the weather)",
              "crps": "CRPS place by place, mm (lower is better)",
              "areal_crps": "CRPS of the mean rain over two-degree boxes, mm (lower is better)",
              "energy": "Energy score per root place, mm (lower is better)",
              "variogram": "Variogram score for places within 150 km (lower is better)"}
    lines = ["Each cell: raw ECMWF ENS / Pluvium coupled / Pluvium uncoupled.", ""]
    for key, title in titles.items():
        lines += [f"**{title}**", "", "| Place | " + " | ".join(LEAD_GROUPS) + " |", "|---|" + "---|" * len(LEAD_GROUPS)]
        for r in results:
            cells = []
            for g in LEAD_GROUPS:
                row = r["groups"].get(g, {})
                if not all(row.get(kind) for kind in KINDS):
                    cells.append("-")
                    continue
                vals = [row[kind][key] for kind in KINDS]
                fmt = "{:.2f}" if key != "variogram" else "{:.3f}"
                best = int(np.argmin(vals)) if key != "sharpness" else int(np.argmin(np.abs(np.array(vals) - 1.0)))
                cells.append(" / ".join(f"**{fmt.format(v)}**" if j == best else fmt.format(v) for j, v in enumerate(vals)))
            lines.append(f"| {r['region'].replace('_', ' ').title()} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    import argparse

    from ..frontier.store import Store
    from ..sources.archive import ArchiveReader

    p = argparse.ArgumentParser(prog="pluvium scenario-bench", description="Coupled scenarios against raw ensembles.")
    p.add_argument("--store", required=True, help="a place's store folder")
    p.add_argument("--keep", required=True, help="the chances kept by the evaluation (.npz with blocks and p)")
    p.add_argument("--archive", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    d = np.load(a.keep)
    result = run(Store(a.store), d["blocks"], d["p"], ArchiveReader(a.archive), a.region)
    Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
