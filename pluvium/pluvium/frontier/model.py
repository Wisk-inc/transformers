# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium Blend as benchmarked on the frontier, for any country, every day.

:class:`FrontierBlend` is :class:`pluvium.blendlive.BlendModel` with the four changes of the frontier benchmark:

* **components see every heavy day**: each is trained on all rows with observed or forecast rain of 25 mm or more,
  the rest thinned, with weights that undo the thinning (:mod:`pluvium.tails.stratify`);
* **the pool** is fitted on the same kind of rows (:mod:`pluvium.pool.gated`);
* **online weights**: the components and the pool are re-weighed after every verified day
  (:mod:`pluvium.pool.online`). The weights are part of the saved model and move on with :meth:`FrontierBlend.learn`;
* **record storms**: a forecast outside anything the training period held is handed back, in part, to what the raw
  systems say (:mod:`pluvium.tails.override`), but only where that helped on storms staged inside the training
  period: its wettest days are withheld from training, then forecast with and without the hand-over
  (:mod:`pluvium.tails.withheld`). Where it did not help, the calibration keeps the forecast.

Its forecast table also says, for every place and day, which alert tier applies (:mod:`pluvium.tails.alerts`),
whether the forecast is beyond the training record, and how rare the expected amount is at that place
(:mod:`pluvium.tails.returns`).

A model saved by :class:`pluvium.blendlive.BlendModel` loads as before; :func:`load_blend` returns whichever kind the
folder holds.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .. import blend as B
from ..blendlive import BlendModel, read_state
from ..pool import OnlineAggregator, fit_pool, predict_pool, run_online
from ..pool.gated import GatedPool
from ..sources.archive import ARCHIVE_SYSTEMS, DAYS, LIMIT, ArchiveReader
from ..tails.audit import audit_handover
from ..tails.consensus import raw_consensus
from ..tails.override import agree_share_from, blend_raw, handover
from ..tails.returns import ReturnLevels
from ..tails.stratify import table_order
from ..tails.support import Support
from ..tails.withheld import valid_days, verdict_line, withheld_experiment
from ..tails.withheld import verdict as staged_verdict
from ..util import hour_of, hour_to_datetime64, log
from .oof import components_for, fit_component, oof_rows, row_arrays, usable

BANDS = (0, 5, 11)
HANDOVER_MODES = ("auto", "always", "never")
#: days of training below which no storms are staged (the verdict then keeps the hand-over)
MIN_DAYS_TO_STAGE = 90
HEAVY_INDEX = int(np.searchsorted(B.THRESHOLDS, 25.0))
ENSEMBLES = {"ens", "gefs"}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z.astype(np.float32)))


class FrontierBlend(BlendModel):
    def __init__(self, nets: dict, pool, systems: list, area_json: dict, clim: dict, grid: tuple, meta: dict,
                 models: dict, online: OnlineAggregator | None = None, support: Support | None = None,
                 cells: tuple | None = None, returns: ReturnLevels | None = None) -> None:
        super().__init__(nets, pool, systems, area_json, clim, grid, meta)
        self.models = models                # component -> the systems it reads, in the pool's order
        self.online, self.support = online, support
        self.cells = cells                  # (lat, lon) of the places it was trained at, where it keeps learning
        self.returns = returns              # how rare an amount is at each grid cell (the grid flattened)

    def components(self) -> dict:
        return {k: v for k, v in self.models.items() if k in self.nets}

    @property
    def hands_over(self) -> bool:
        """Whether record storms are handed to the raw systems. With the mode ``"auto"`` (the default): only where
        the storms staged at training were forecast better with the hand-over (:mod:`pluvium.tails.withheld`).
        ``"always"`` and ``"never"`` overrule that verdict: see :meth:`set_handover`. A model saved before the verdict
        existed hands over."""
        verdict = self.meta.get("handover", {})
        mode = verdict.get("mode", "auto")
        if self.support is None or mode == "never":
            return False
        return True if mode == "always" else bool(verdict.get("keep", True))

    def set_handover(self, mode: str) -> None:
        """Choose how record storms are treated: ``"auto"`` (follow the verdict of the storms staged at training),
        ``"always"`` or ``"never"``.

        The verdict rests on four storms of one training period, so it can be wrong. A place that would rather hand
        every record forecast to the raw systems than see one shrunk can set ``"always"``; a place where the raw
        systems are known to overcall can set ``"never"``."""
        if mode not in HANDOVER_MODES:
            raise ValueError(f"handover must be one of {HANDOVER_MODES}, not {mode!r}")
        self.meta["handover"] = {**self.meta.get("handover", {}), "mode": mode}

    # ------------------------------------------------------------------------------------- train --
    @classmethod
    def from_table(cls, table: dict, n: int, systems: list, area_json: dict, clim: dict, grid: tuple, cells: tuple,
                   max_rows: int = 400_000, pool_rows: int = 300_000, seed: int = 0, eta: float = 1.0,
                   share: float = 0.02, pool_prior: float = 0.5, latency_hours: int = 48,
                   record_quantile: float = 0.999, meta: dict | None = None, handover: str = "auto") -> FrontierBlend:
        """Fit everything from a training table built with ``keep_raw=True``. ``handover``: ``"auto"``,
        ``"always"`` or ``"never"`` (:meth:`set_handover`); the storms are staged and the verdict saved whichever is chosen."""
        if handover not in HANDOVER_MODES:
            raise ValueError(f"handover must be one of {HANDOVER_MODES}, not {handover!r}")
        models = components_for(systems)
        every = np.arange(len(table["run"]))
        nets = {}
        for name, members in models.items():
            net = fit_component(table, usable(table, every, members, len(members) == 1), members, n, max_rows, seed,
                                stratified=True, signal_systems=systems)
            if net is not None:
                nets[name] = net
        models = {k: v for k, v in models.items() if k in nets}
        order, weights = table_order(table, every, n, pool_rows, seed + 7, signal_systems=systems)
        z, present = oof_rows(table, order, models, n, max_rows, seed, True)
        rows = row_arrays(table, order)
        zf = z.astype(np.float32)
        lead = rows["lead"].astype(np.float32)
        pool = fit_pool(zf, present, lead, rows["y"], None, weights, bands=BANDS, smooth=0.0, iters=120)
        # the online layer starts from how the experts did over the training period, in time order
        prior = np.concatenate([np.full(len(models), (1 - pool_prior) / len(models)), [pool_prior]])
        online = OnlineAggregator(len(models) + 1, eta, share, prior)
        p_exp = np.concatenate([sigmoid(zf), predict_pool(pool, zf, present, lead)[:, None, :]], 1)
        pres = np.concatenate([present.astype(bool), np.ones((present.shape[0], 1), bool)], 1)
        issued = rows["valid"].astype(np.int64) - 24 * (rows["lead"].astype(np.int64) - 1)
        run_online(online, p_exp, pres, rows["lead"], rows["y"], issued, rows["valid"] + 24 + latency_hours, weights)
        raw_mean = np.concatenate([table["raw_mean"][i][c] for i, c in order])
        support = Support.from_rows(raw_mean, weights, record_quantile)
        # is the hand-over worth having here? the place's own storms decide: the wettest days of the training period
        # are withheld, forecast without having been seen, and scored with and without the hand-over
        if np.unique(valid_days(table)).size >= MIN_DAYS_TO_STAGE:
            staged = staged_verdict(withheld_experiment(table, n, list(systems), top=4, quantiles=(record_quantile,),
                                                        widths=(support.width,), max_rows=max_rows, seed=seed))
        else:
            staged = staged_verdict({})                             # too short a period to stage storms in: kept
        log(f"blend: {verdict_line(staged)}")
        # the yearly check: what the hand-over did to the training period's out-of-fold forecasts (reported only)
        raw_p = np.concatenate([table["raw_p"][i][c] for i, c in order]).astype(np.float32)
        yearly = audit_handover(p_exp[:, -1], raw_p, raw_mean, rows["y"], support, weights, rows["valid"] // 24)
        info = {"components": list(models), "cells": int(n), "kind": "frontier", "latency_hours": latency_hours,
                "learned_days": [], "handover": {"keep": bool(staged["keep"]), "mode": handover, "staged": staged,
                                                 "yearly": yearly.to_json()}, **(meta or {})}
        return cls(nets, pool, list(systems), area_json, clim, grid, info, models, online, support, cells)

    @classmethod
    def train(cls, area, root, start: str = "2024-04-01T00", end: str = "2025-04-01T00", workers: int = 8,
              max_cells: int = 1500, max_rows: int = 400_000, handover: str = "auto") -> FrontierBlend:
        root = Path(root).expanduser()
        data = cls.prepare(area, root, start, end, workers, max_cells, keep_raw=True)
        cells, table, lat, lon = data["cells"], data["table"], data["lat"], data["lon"]
        clim = cls.grid_climatology(area, data["truth"], lat, lon, start, end)
        model = cls.from_table(table, len(cells), list(ARCHIVE_SYSTEMS), area.to_json(), clim, (lat, lon),
                               (cells.lat, cells.lon), max_rows, meta={"trained": [start, end], "runs": int(len(data["runs"]))},
                               handover=handover)
        days = np.arange(hour_of(start), hour_of(end), 24)              # how rare an amount is, from the same record
        model.returns = ReturnLevels.fit(data["truth"].days(area.name, days).reshape(days.size, -1))
        model.save(root / "blend")
        return model

    # ---------------------------------------------------------------------------------------- io --
    def save(self, directory) -> Path:
        d = Path(directory).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        # tensors and plain values only, so the file can be opened without running code (blendlive.read_state)
        online = None if self.online is None else {
            k: torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else int(v)
            for k, v in self.online.state().items()}
        torch.save({"nets": {k: net.state_dict() for k, net in self.nets.items()},
                    "n_in": {k: int(net.mu.numel()) for k, net in self.nets.items()},
                    "pool": self.pool.state_dict(), "components": list(self.components()),
                    "online": online}, d / "blend.pt")
        extra = {} if self.cells is None else {"cell_lat": self.cells[0], "cell_lon": self.cells[1]}
        np.savez_compressed(d / "climatology.npz", lat=self.grid[0], lon=self.grid[1], **extra,
                            **{f"{k}_{m:02d}": v for m, entry in self.clim.items() for k, v in entry.items()})
        online = None if self.online is None else {"eta": self.online.eta, "share": self.online.share,
                                                   "prior": self.online.prior.tolist()}
        (d / "blend.json").write_text(json.dumps({
            "kind": "frontier", "systems": self.systems, "area": self.area_json, "meta": self.meta, "models": self.models,
            "thresholds": B.THRESHOLDS.tolist(), "bands": list(BANDS), "online": online,
            "support": None if self.support is None else {"edge_mm": self.support.edge_mm, "width": self.support.width}},
            indent=1))
        if self.returns is not None:
            self.returns.save(d / "returns.npz")
        return d

    @classmethod
    def load(cls, directory, trusted: bool = True) -> FrontierBlend:
        """The blend saved in ``directory``. ``trusted=False`` for files that came from somewhere else: they are
        opened only if that cannot run code (:func:`pluvium.blendlive.read_state`)."""
        d = Path(directory).expanduser()
        meta = json.loads((d / "blend.json").read_text())
        state = read_state(d / "blend.pt", trusted)
        nets = {}
        for k, sd in state["nets"].items():
            nets[k] = B.BlendNet(int(state["n_in"][k]))
            nets[k].load_state_dict(sd)
            nets[k].eval()
        pool = GatedPool(len(state["components"]), bands=tuple(meta.get("bands", BANDS)))
        pool.load_state_dict(state["pool"])
        pool.eval()
        online = None
        if meta.get("online") and state.get("online") is not None:
            o = meta["online"]
            online = OnlineAggregator(len(state["components"]) + 1, o["eta"], o["share"], np.asarray(o["prior"]))
            online.load_state({k: v.numpy() if torch.is_tensor(v) else v for k, v in state["online"].items()})
        support = None if not meta.get("support") else Support(meta["support"]["edge_mm"], meta["support"]["width"])
        z = np.load(d / "climatology.npz")
        clim, cells = {}, None
        if "cell_lat" in z.files:
            cells = (z["cell_lat"], z["cell_lon"])
        for key in z.files:
            if key in ("lat", "lon", "cell_lat", "cell_lon"):
                continue
            name, m = key.rsplit("_", 1)
            clim.setdefault(int(m), {})[name] = z[key]
        models = {k: v for k, v in meta["models"].items() if k in nets}
        returns = ReturnLevels.load(d / "returns.npz") if (d / "returns.npz").exists() else None
        return cls(nets, pool, meta["systems"], meta["area"], clim, (z["lat"], z["lon"]), meta["meta"], models, online,
                   support, cells, returns)

    # ---------------------------------------------------------------------------------- forecast --
    def experts(self, per: dict, clim: dict, lead: float, n: int) -> tuple:
        """Every expert's chances ``(n, components + 1, K)`` (the pool last) and which are there ``(n, components + 1)``."""
        comps = self.components()
        z = np.zeros((n, len(comps), len(B.THRESHOLDS)), np.float32)
        present = np.zeros((n, len(comps)), bool)
        for k, (name, members) in enumerate(comps.items()):
            single = len(members) == 1
            if (single and per.get(members[0]) is None) or not any(per.get(s) is not None for s in members):
                continue
            x = B.assemble({s: per.get(s) for s in members}, members, ENSEMBLES, clim, lead, n)
            z[:, k] = B.to_logits(B.predict(self.nets[name], x))
            present[:, k] = True
        p_pool = predict_pool(self.pool, z, present, float(lead))
        p = np.concatenate([sigmoid(z), p_pool[:, None, :]], 1)
        return p, np.concatenate([present, np.ones((n, 1), bool)], 1)

    def chances(self, per: dict, clim: dict, lead: float, n: int, raw: dict | None = None) -> np.ndarray:
        """P(rain >= THRESHOLDS) at ``n`` places: the pool, re-weighed online, with record storms handed over."""
        p_exp, present = self.experts(per, clim, lead, n)
        p = p_exp[:, -1] if self.online is None else self.online.combine(p_exp, present, np.full(n, int(round(lead))))
        if self.hands_over and raw:
            p_raw, raw_mean = raw_consensus(raw, n)
            g = handover(self.support, raw_mean, agree_share_from(p_raw, HEAVY_INDEX))
            if g.max() > 1e-4:
                p = blend_raw(p, p_raw, g)
        return p

    def extras(self, p: np.ndarray, q: np.ndarray, raw: dict, lat: np.ndarray, lon: np.ndarray) -> dict:
        """The alert tier, and two more: whether the raw systems call for more than anything in the training period
        (``beyond_record``), and how rare the expected amount is at each place (``rarity_years`` and, in words,
        ``rarity``)."""
        out = super().extras(p, q, raw, lat, lon)
        if self.support is not None and raw:
            out["beyond_record"] = self.support.outside(raw_consensus(raw, lat.size)[1])
        if self.returns is not None:
            i, j = self._grid_index(lat, lon)
            years = self.returns.period(q.mean(1), index=i * self.grid[1].size + j)
            out["rarity_years"] = years
            out["rarity"] = [self.returns.wording(x) for x in years]
        return out

    # ------------------------------------------------------------------------------------- learn --
    def learn_rows(self, p_exp: np.ndarray, present: np.ndarray, lead: np.ndarray, y: np.ndarray) -> None:
        """Update the online weights from verified forecasts: the experts' chances, and the rain that fell."""
        if self.online is not None:
            self.online.update(p_exp, present, lead, y)

    def first_unlearned(self) -> int:
        """The first day (hours since 1970) the online weights have not seen: the day after the newest one learned
        from, or, for a blend that has only been trained, the first day forecast entirely by runs after its training
        period."""
        done = self.meta.get("learned_days", [])
        if done:
            return hour_of(max(done) + "T00") + 24
        end = (self.meta.get("trained") or [None, None])[1]
        return hour_of(end) + 24 * (DAYS + 1) if end else 0

    def learn(self, cache, truth, now: int | None = None, leads=(1, 2, 3, 5, 7, 10, 14), max_days: int = 14,
              since: int | None = None, until: int | None = None) -> list:
        """Learn from days whose rain is known and not yet learned from, at most ``max_days`` of them, oldest first.

        Without ``since`` the newest ``max_days`` days are looked at, which suits a truth that arrives within two
        days. The satellite record Pluvium trains on arrives months late, so the command line passes ``since``
        (:meth:`first_unlearned`) and ``until`` (the last day the record holds) and each call catches up a little.

        For each day the runs that forecast it at ``leads`` days ahead are read (from ``cache``, downloaded when
        missing), the experts' chances at the training places are rebuilt, and the weights are updated against
        ``truth`` (a :class:`pluvium.sources.cmorph_daily.DailyTruth`, or anything with ``days(name, [hour])``).
        Returns the days learned from. Call :meth:`save` afterwards."""
        from ..benchmarks.fullyear import CellSet, system_at_cells

        if self.online is None or self.cells is None:
            return []
        now = int(datetime.now(timezone.utc).timestamp() // 3600) if now is None else int(now)
        name = self.area_json["name"]
        lat, lon = self.cells
        cells = CellSet(lat, lon, None, name)
        reader = ArchiveReader(cache, {name: tuple(self.area_json["target_box"])})
        glat, glon = self.grid
        i = np.clip(np.round((lat - glat[0]) / (glat[1] - glat[0])).astype(int), 0, glat.size - 1)
        j = np.clip(np.round((lon - glon[0]) / (glon[1] - glon[0])).astype(int), 0, glon.size - 1)
        done = set(self.meta.get("learned_days", []))
        last = now - now % 24 - 24 - int(self.meta.get("latency_hours", 48))        # the newest day that can be known
        if until is not None:
            last = min(last, int(until) - int(until) % 24)
        first = last - 24 * (max_days - 1) if since is None else int(since) - int(since) % 24
        learned = []
        for day in range(first, last + 1, 24):
            if len(learned) >= max_days:
                break
            stamp = str(hour_to_datetime64(day))[:10]
            if stamp in done:
                continue
            try:
                y = np.asarray(truth.days(name, [day])[0])[i, j]
            except Exception:
                continue
            if np.isfinite(y).mean() < 0.5:
                continue
            clim = self._clim_at(lat, lon, int(stamp[5:7]))
            used = 0
            for lead in leads:
                init = day - 24 * (lead - 1)
                per = {}
                for system in self.systems:
                    if lead > LIMIT.get(system, DAYS) or not reader.read(system, init, min(DAYS, LIMIT.get(system, DAYS))):
                        continue
                    f = system_at_cells(reader.load(name, system, init), int(lead), cells)
                    if f is not None:
                        f.pop("_members")
                        per[system] = f
                if per:
                    p_exp, present = self.experts(per, clim, float(lead), lat.size)
                    self.learn_rows(p_exp, present, np.full(lat.size, lead), y)
                    used += 1
            if used:                                                # a day no run could be read for is tried again
                learned.append(stamp)
        self.meta["learned_days"] = sorted(done | set(learned))[-400:]
        if learned:
            log(f"blend for {name}: learned from {len(learned)} day(s), {learned[0]} to {learned[-1]}")
        return learned

    def weights(self) -> dict:
        """The online weights for light rain at days 1-3: ``{expert: weight}`` (the pool last)."""
        if self.online is None:
            return {}
        names = list(self.components()) + ["pool"]
        return {n: float(w) for n, w in zip(names, self.online.weights()[0, 0])}


def load_blend(directory, trusted: bool = True) -> BlendModel:
    """The blend saved in a folder, whichever kind it is. ``trusted=False`` for a folder that was downloaded."""
    meta = json.loads((Path(directory).expanduser() / "blend.json").read_text())
    kind = FrontierBlend if meta.get("kind") == "frontier" else BlendModel
    return kind.load(directory, trusted)


__all__ = ["BANDS", "HANDOVER_MODES", "FrontierBlend", "load_blend", "read_state"]
