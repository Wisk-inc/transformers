# Copyright 2026 Nathan. Apache-2.0.
"""
The component store: every component's forecasts for a place, kept so that pools can be fitted and scored in minutes.

Fitting the components is the expensive part of the benchmark: seven small networks, fitted three times each (once on
everything, twice for out-of-fold forecasts) at four quarterly refits. None of it depends on how the components are
then combined. So it is done once per place and stored:

* ``meta.npz``: every block's run, lead and rain, the raw consensus mean, and the places;
* ``clim.npz``: the climatology by month;
* ``raw.json``: the raw systems' scores (computed while the table is read, as before);
* ``raw_test.npz``: the raw consensus chances for every test block (for the record-storm hand-over);
* ``q0.npz`` ... ``q3.npz``, one per quarterly refit: out-of-fold log-odds of every component on the training rows
  kept for the pool (every heavy-rain row, a share of the rest, with weights), and the components' log-odds for every
  test block of the quarter.

Everything is causal exactly as in the first benchmark: a quarter's components see only blocks verified before it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .. import blend as B
from ..benchmarks import fullyear as F
from ..sources.archive import ARCHIVE_SYSTEMS, ArchiveReader, run_set
from ..tails.stratify import table_order, table_rows
from ..util import hour_of, log
from .oof import block_logits, components_for, fit_component, has, oof_rows, row_arrays, usable


@dataclass
class FrontierConfig:
    archive: str
    truth: str
    out: str
    train_start: str = "2024-04-01T00"
    test_start: str = "2025-04-01T00"
    test_end: str = "2026-04-01T00"
    refit_months: int = 3
    max_cells: int = 1500
    max_rows: int = 400_000
    #: train the components on every heavy-rain row with inverse-probability weights (see pluvium.tails.stratify)
    stratified: bool = True
    #: rows kept per refit for fitting pools (tail-stratified)
    oof_rows: int = 400_000
    thresholds: tuple = B.SCORED
    seed: int = 0
    case_days: tuple = ("2025-10-27", "2025-10-28", "2025-10-29")
    #: hours after a day ends before its rain is known (the interim CMORPH arrives within two days)
    latency_hours: int = 48


def refits(cfg) -> list:
    """The hours at which components are refitted, and the end of the test period."""
    out, t, end = [], hour_of(cfg.test_start), hour_of(cfg.test_end)
    while t < end:
        out.append(t)
        day = str(np.datetime64(int(t), "h"))
        y_, m_ = int(day[:4]), int(day[5:7]) - 1 + cfg.refit_months
        t = hour_of(f"{y_ + m_ // 12}-{m_ % 12 + 1:02d}-01T00")
    return out + [end]


def place_table(cfg: FrontierConfig, place: str, country: dict | None = None):
    """Cells, table (with the raw consensus) and the raw systems' scores for a region or a country.

    ``country``: ``{"geography", "cmorph", "forecasts", "clim_span", "region"}`` for a country at Pluvium's own
    resolution with its network as a seventh system; ``None`` for one of the archive's regions."""
    reader = ArchiveReader(cfg.archive)
    runs, long = run_set(cfg.train_start, cfg.test_end, every=2, long_every=4)
    scores = F.Scores(cfg.thresholds, cfg.case_days)
    window = (hour_of(cfg.test_start), hour_of(cfg.test_end))
    if country is None:
        cells = F.region_cells(cfg, place, reader)
        cells.climatology(np.arange(hour_of(cfg.train_start), hour_of(cfg.test_start), 24))
        table = F.build_table(cells, reader, place, runs, set(int(r) for r in long), scores=scores, test_window=window,
                              keep_raw=True)
    else:
        cells = F.country_cells(place, country["geography"], country["cmorph"])
        span = country["clim_span"]
        cells.climatology(np.arange(hour_of(span[0]), hour_of(span[1]), 24))
        table = F.build_table(cells, reader, country.get("region", "caribbean"), runs, set(int(r) for r in long),
                              neural=F.neural_reader(country["forecasts"]), scores=scores, test_window=window,
                              keep_raw=True)
    return cells, table, scores


def summaries(scores: F.Scores, names: list, thresholds) -> dict:
    """A model's scores in the form the reports use: by lead, by lead group, by season, and Brier per run."""
    out = {"leads": {}, "groups": {}, "seasons": {}, "by_run": {}}
    for name in names:
        if name not in scores.data:
            continue
        out["leads"][name] = {str(lead): scores.summary(name, [lead]) for lead in range(1, 15) if scores.data[name].get(lead)}
        out["groups"][name] = {g: scores.summary(name, leads) for g, leads in F.LEAD_GROUPS.items()}
        out["seasons"][name] = {g: {s: scores.summary(name, leads, s) for s in F.SEASONS} for g, leads in F.LEAD_GROUPS.items()}
        out["by_run"][name] = {g: {f"{t:g}": {str(r): v for r, v in scores.by_run(name, leads, t).items()}
                                   for t in thresholds} for g, leads in F.LEAD_GROUPS.items()}
    return out


def csi_cuts(net, table: dict, blocks: np.ndarray, members: list, n: int, thresholds, seed: int = 0) -> dict:
    """The chance cut that maximises CSI on training rows, per scored threshold."""
    x, y, _, _ = table_rows(table, blocks, members, n, 100_000, seed=seed + 3, stratified=False)
    if x.shape[0] == 0:
        return {f"{t:g}": 0.5 for t in thresholds}
    p = B.predict(net, x)
    return {f"{t:g}": F.best_cut(B.exceed(p, t), y >= t) for t in thresholds}


def build_store(cfg: FrontierConfig, place: str, country: dict | None = None) -> Path:
    """Fit and store every component's forecasts for ``place`` (skipped when the store is complete)."""
    out = Path(cfg.out) / "store" / place
    if (out / "done").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    cells, table, raw_scores = place_table(cfg, place, country)
    n = len(cells)
    systems = list(ARCHIVE_SYSTEMS)
    with_neural = country is not None
    models = components_for(systems, with_neural)
    names = list(models)
    all_systems = sorted({s for members in models.values() for s in members})
    runs, leads = np.asarray(table["run"]), np.asarray(table["lead"])
    log(f"{place}: {n} cells, {len(runs)} blocks, components {', '.join(names)}")

    raw_names = [f"{s}_raw" for s in systems] + (["neural_raw"] if with_neural else [])
    raw = summaries(raw_scores, raw_names, cfg.thresholds)
    raw["cases"] = raw_scores.cases
    (out / "raw.json").write_text(json.dumps(raw, default=float))
    del raw_scores
    np.savez(out / "meta.npz", run=runs, lead=leads, y=np.stack(table["y"]).astype(np.float32),
             raw_mean=np.stack(table["raw_mean"]).astype(np.float32), raw_n=np.asarray(table["raw_n"], np.int16),
             lat=cells.lat, lon=cells.lon, names=np.array(names), systems=np.array(systems))
    np.savez(out / "clim.npz", **{f"{k}__{m:02d}": v for m, entry in cells.clim.items() for k, v in entry.items()})
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))

    dates = refits(cfg)
    valid_end = runs + 24 * leads
    test_all = np.flatnonzero((runs >= dates[0]) & (runs < dates[-1]))
    np.savez_compressed(out / "raw_test.npz", blocks=test_all, raw_p=np.stack([table["raw_p"][i] for i in test_all]))
    for q, (a, b) in enumerate(zip(dates[:-1], dates[1:])):
        train = np.flatnonzero(valid_end <= a)
        test = test_all[(runs[test_all] >= a) & (runs[test_all] < b)]
        if test.size == 0 or train.size == 0:
            continue
        log(f"  {place}: quarter {q}: {train.size} training blocks ({train.size * n:,} rows), {test.size} test blocks")
        nets, cuts = {}, {}
        for name, members in models.items():
            mine = usable(table, train, members, len(members) == 1)
            nets[name] = fit_component(table, mine, members, n, cfg.max_rows, cfg.seed, cfg.stratified,
                                       signal_systems=all_systems)
            if nets[name] is not None:
                cuts[name] = csi_cuts(nets[name], table, mine, members, n, cfg.thresholds, cfg.seed)
        order, weights = table_order(table, train, n, cfg.oof_rows, cfg.seed + 7, signal_systems=all_systems)
        z, present = oof_rows(table, order, models, n, cfg.max_rows, cfg.seed, cfg.stratified)
        rows = row_arrays(table, order)
        raw_p = np.concatenate([table["raw_p"][i][c] for i, c in order])
        raw_mean = np.concatenate([table["raw_mean"][i][c] for i, c in order])
        test_z = np.zeros((test.size, n, len(names), len(B.THRESHOLDS)), np.float16)
        test_present = np.zeros((test.size, len(names)), bool)
        for r, i in enumerate(test):
            for k, (name, members) in enumerate(models.items()):
                if nets[name] is None or not has(table, int(i), members, len(members) == 1):
                    continue
                test_z[r, :, k] = block_logits(nets[name], table, int(i), members, n)
                test_present[r, k] = True
        np.savez(out / f"q{q}.npz", oof_z=z, oof_present=present, oof_w=weights, oof_raw_p=raw_p.astype(np.float16),
                 oof_raw_mean=raw_mean.astype(np.float32), test=test, test_z=test_z, test_present=test_present,
                 cuts=json.dumps(cuts), **{f"oof_{k}": v for k, v in rows.items()})
        log(f"  {place}: quarter {q} stored ({z.shape[0]:,} out-of-fold rows)")
    (out / "done").write_text("ok")
    return out


class Store:
    """A place's stored components, read back."""

    def __init__(self, folder) -> None:
        self.folder = Path(folder)
        meta = np.load(self.folder / "meta.npz")
        self.run, self.lead, self.y = meta["run"], meta["lead"], meta["y"]
        self.raw_mean, self.raw_n = meta["raw_mean"], meta["raw_n"]
        self.lat, self.lon = meta["lat"], meta["lon"]
        self.names = [str(x) for x in meta["names"]]
        self.systems = [str(x) for x in meta["systems"]]
        clim = np.load(self.folder / "clim.npz")
        self.clim = {}
        for key in clim.files:
            name, m = key.split("__")
            self.clim.setdefault(int(m), {})[name] = clim[key]
        self.raw = json.loads((self.folder / "raw.json").read_text())
        self.config = json.loads((self.folder / "config.json").read_text())
        self._raw_test = None

    @property
    def n(self) -> int:
        return self.y.shape[1]

    def quarters(self) -> list:
        return sorted(int(p.stem[1:]) for p in self.folder.glob("q*.npz"))

    def quarter(self, q: int) -> dict:
        d = np.load(self.folder / f"q{q}.npz")
        out = {k: d[k] for k in d.files if k != "cuts"}
        out["cuts"] = json.loads(str(d["cuts"]))
        return out

    def raw_p(self, block: int) -> np.ndarray:
        """The raw consensus chances ``(n, K)`` for a test block."""
        if self._raw_test is None:
            d = np.load(self.folder / "raw_test.npz")
            self._raw_test = (d["raw_p"], {int(b): k for k, b in enumerate(d["blocks"])})
        return self._raw_test[0][self._raw_test[1][int(block)]].astype(np.float32)

    def table(self) -> dict:
        """The light table the scoring functions take: run, lead, rain and climatology per block."""
        months = [int(str(np.datetime64(int(r + 24 * (ld - 1)), "h"))[5:7]) for r, ld in zip(self.run, self.lead)]
        return {"run": self.run, "lead": self.lead, "y": list(self.y), "clim": [self.clim[m] for m in months]}


__all__ = ["FrontierConfig", "Store", "build_store", "csi_cuts", "place_table", "refits", "summaries"]
