# Copyright 2026 Nathan. Apache-2.0.
"""
The live forecast: for every town and village in the Caribbean, hour by hour for three days -- the chance of rain,
whether it will be drizzle, showers, steady or heavy rain or thunderstorms, when it is most likely to start (in local
time) and when it will be heaviest, how much may fall, the flash-flood risk, and why.

1. The newest run of each driving model (GFS, ECMWF IFS, ECMWF AIFS) that covers the next 72 hours is read onto the
   training grids (:mod:`pluvium.sources.nwp`); the forecast starts at the current hour, a few hours into those runs.
2. What is happening now is read live: rain from GOES-East (its rain-rate product, or its infrared imagery), the
   reports of every airport for the last three hours (:mod:`pluvium.sources.metar`), and lightning
   (:mod:`pluvium.sources.glm`).
3. Pluvium runs once per driving model and the chains are mixed (:mod:`pluvium.products`); the calibration fitted on
   held-out data and the online corrections learned from recent errors (:mod:`pluvium.online`) are applied.
4. Rain is routed through each town's river catchment for the flood risk (:mod:`pluvium.flood`), and everything is
   put in words (:mod:`pluvium.narrative`).
5. Results go to ``towns.csv`` (one row per place), ``parishes.csv``, ``towns.geojson`` (for maps), ``hourly.npz`` and
   ``timelines.json`` (hour by hour), ``forecast.json`` and ``report.txt``. Each run is archived so it can be checked
   against what the airports report later (``pluvium learn``).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .calibrate import Calibrator
from .dataset import Climatology, SampleBuilder, collate
from .domain import Gazetteer, Geography, PointSet
from .features import REGIME_TEXT, REGIMES
from .model import POINT_AXIS
from .narrative import any_in_window, hour_words, local_time, parish_sentence, town_sentence
from .products import TOTAL_THRESHOLDS, WINDOWS, summarize
from .train import borrowed_climatology, load_model, nearest_target, to_device
from .util import hour_of, hour_to_iso, log

MODELS = ("gfs", "ifs", "aifs")


def _concat(parts: list) -> dict:
    """Join per-chunk summaries along their point axis."""
    first = parts[0]
    out = {}
    for key, value in first.items():
        if isinstance(value, dict):
            out[key] = _concat([p[key] for p in parts])
        elif isinstance(value, np.ndarray):
            axis = 1 if key == "cause" else (1 if key == "samples24" else 0)
            out[key] = np.concatenate([p[key] for p in parts], axis)
        else:
            out[key] = value
    return out


def _slice_points(out: dict, sl: slice) -> dict:
    return {k: (v[:, sl] if POINT_AXIS.get(k, 2) == 1 else v[:, :, sl]) for k, v in out.items()}


def groups_of(towns: Gazetteer) -> np.ndarray:
    x = towns.points.extra
    n = len(towns.points)
    country = x.get("country", np.array([""] * n)).astype(str)
    unit = x.get("unit", np.array([""] * n)).astype(str)
    return np.char.add(np.char.add(country, "/"), unit)


class Forecaster:
    """
    ``Forecaster(model_dir)`` loads a trained Pluvium (weights, calibration, climatology, geography, towns, the
    online corrections) and ``run()`` makes the live forecast; ``from_fields`` makes one from driving fields you
    supply (hindcasts, tests).
    """

    def __init__(self, model_dir, geography: Geography | None = None, towns: Gazetteer | None = None, device=None,
                 climatology: Climatology | None = None, cache="~/.cache/pluvium", state_dir=None) -> None:
        d = Path(model_dir).expanduser()
        self.dir = d
        self.cache = Path(cache).expanduser()
        self.state_dir = Path(state_dir).expanduser() if state_dir else self.cache / "live"
        self.geo = geography or Geography.load(d / "geography")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, _, self.meta, _ = load_model(d, self.geo, self.device)
        self.model.eval()
        self.calibrator = Calibrator.from_json(self.meta["calibration"]) if self.meta.get("calibration") else None
        if climatology is None:
            path = d / "climatology.npz"
            climatology = Climatology.load(path) if path.exists() else Climatology.flat(len(self.geo.targets))
        self.clim = climatology
        self.towns = towns or Gazetteer(PointSet.load(d / "towns.npz"))
        g = self.meta["grids"]
        holder = SimpleNamespace(**{k: np.asarray(g[k], np.float64) for k in ("lat1", "lon1", "lat2", "lon2")})
        self.builder = SampleBuilder(self.geo, holder, None, None, self.clim, schedule=self.model.schedule)
        self._town_setup()
        self.station_meta = (self.meta.get("stations") or {}).get("list") or []
        st = self.model.stations
        if st is not None:
            self.station_clim = borrowed_climatology(self.clim, nearest_target(self.geo, st.lat, st.lon))
            near_town = nearest_target_town(self.towns, st.lat, st.lon)
            self.station_groups = groups_of(self.towns)[near_town]
        from .online import OnlineCorrector

        self.corrector = OnlineCorrector.load(self.state_dir / "corrector.json")

    def _town_setup(self) -> None:
        p = self.towns.points
        self.town_clim = borrowed_climatology(self.clim, nearest_target(self.geo, p.lat, p.lon))
        self.town_groups = groups_of(self.towns)
        self._catchments = None
        self._urban = None

    def catchments(self):
        if self._catchments is None:
            from .flood import Catchments

            p = self.towns.points
            self._catchments = Catchments(self.geo.terrain, p.lat, p.lon, self.geo.targets.lat, self.geo.targets.lon)
        return self._catchments

    # ------------------------------------------------------------------------------------- inputs --
    def driving(self, t0: int, models=MODELS) -> list[dict]:
        """The newest usable run of each model, as frames for a forecast issued at ``t0``."""
        from .sources.nwp import NWPReader, latest_cycle

        g = self.builder
        got = []
        for name in models:
            reader = NWPReader(name, g.lat1, g.lon1, g.lat2, g.lon2, self.cache / "nwp", self.model.schedule)
            init = latest_cycle(name, t0, reader)
            if init is None:
                log(f"{name}: no run covering the next 72 hours has been published yet; left out")
                continue
            try:
                frames = reader.frames(init, offset=t0 - init)
            except Exception as error:
                log(f"{name}: run {hour_to_iso(init)} not usable ({type(error).__name__}: {error}); left out")
                continue
            from .guard.inputs import check_driving

            fields, upper, notes = check_driving(frames["fields"], frames["upper"])
            notes = list(frames.get("notes", [])) + [f"{name}: {n}" for n in notes]
            if t0 - init > 18:
                notes.append(f"{name}: run {t0 - init} h old")
            got.append({"model": name, "init": init, "fields": fields, "upper": upper, "notes": notes})
        return got

    def observations(self, t0: int, rain: bool = True, airports: bool = True, lightning: bool = True) -> dict:
        """What is happening now: GOES rain around mesh points, target cells and towns; airport reports of the last
        three hours; lightning near towns in the last 15 minutes."""
        from .sources.cmorph import default_half_widths

        obs = {"mesh": None, "targets": None, "towns": None, "product": "not used", "station_codes": None,
               "lightning": None, "lightning_note": "not used"}
        mesh, towns, t = self.geo.mesh, self.towns.points, self.geo.targets
        satellite = self.geo.area.satellite if getattr(self.geo, "area", None) is not None else "goes-east"
        if rain and satellite == "none":
            obs["product"] = "no geostationary rain product read for this area yet; forecast made without rain now"
        elif rain:
            try:
                from .sources.goes import recent_rain

                obs["mesh"], obs["product"] = recent_rain(mesh.lat, mesh.lon, default_half_widths(mesh),
                                                          [t0 - 3, t0 - 2, t0 - 1], satellite)
                obs["targets"], _ = recent_rain(t.lat, t.lon, np.full(len(t), 0.04), [t0 - 1], satellite)
                obs["towns"], _ = recent_rain(towns.lat, towns.lon, np.full(len(towns), 0.04), [t0 - 1], satellite)
            except Exception as error:
                log(f"recent rain not read ({type(error).__name__}: {error}); forecasting without it")
            try:
                from .sources.mrms import at_points

                radar_t, note = at_points("rate", t.lat, t.lon, 4.0)
                if np.isfinite(radar_t).any():
                    radar_w, _ = at_points("rate", towns.lat, towns.lon, 4.0)
                    radar_m, _ = at_points("rate", mesh.lat, mesh.lon, 15.0)
                    for key, radar in (("targets", radar_t), ("towns", radar_w)):
                        if obs[key] is None:
                            obs[key] = np.full((1, radar.size), np.nan, np.float32)
                        obs[key][0] = np.where(np.isfinite(radar), radar, obs[key][0])
                    if obs["mesh"] is not None:
                        obs["mesh"][-1] = np.where(np.isfinite(radar_m), radar_m, obs["mesh"][-1])
                    obs["product"] = f"{obs['product']}; {note} where in range"
            except Exception as error:
                log(f"radar not read ({type(error).__name__}: {error})")
        if airports and self.station_meta:
            from .sources.metar import live_station_codes

            s, n, w, e = self.geo.target_box
            obs["station_codes"] = live_station_codes(self.station_meta, [t0 - 3, t0 - 2, t0 - 1],
                                                      box=(s - 1.0, n + 1.0, w - 1.0, e + 1.0))
        if lightning and satellite != "none":
            from .sources.glm import flashes_near

            obs["lightning"], obs["lightning_note"] = flashes_near(towns.lat, towns.lon, satellite=satellite)
        return obs

    # ------------------------------------------------------------------------------------ forecast --
    @torch.no_grad()
    def from_fields(self, components: list[dict], t0: int, obs: dict | None = None, chunk: int = 4000,
                    samples: int = 200, flood: bool = True) -> dict:
        """Forecast for every town (and airport) from driving fields ``[{"fields", "upper", ...}]``."""
        from .weather import decode, obs_features

        obs = obs or {}
        towns = self.towns.points
        st = self.model.stations
        mesh_mem = obs.get("mesh")
        observed = None
        if obs.get("targets") is not None and mesh_mem is not None:
            observed = np.concatenate([obs["targets"][0], mesh_mem[-1]])
        town_mem = obs["towns"][0] if obs.get("towns") is not None else None
        codes = obs.get("station_codes")
        outs, st_outs, target_outs = [], [], []
        for comp in components:
            frames = self.builder.frames_from(comp["fields"], comp["upper"], t0)
            mesh_ctx = self.builder.mesh_context(frames, t0, mesh_mem)
            ctx = self.builder.point_context(towns.lat, towns.lon, np.ones(len(towns), bool), frames, t0,
                                             self.town_clim, town_mem, town_mem is not None, observed)
            example = {"t0": np.int64(t0), "hourly": frames["hourly"], "upper": frames["upper"],
                       "tiles": frames["tiles"], "mesh_ctx": mesh_ctx}
            if st is not None and codes is not None:
                example["station_obs"] = obs_features(codes)
            batch = to_device(collate([example]), self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                latent = self.model.encode(batch)
                out = self.model.forecast_points(batch, towns, torch.from_numpy(ctx[None]).to(self.device), chunk=chunk,
                                                 latent=latent)
                if st is not None:
                    last = None
                    if codes is not None:
                        d = decode(codes[-1])
                        last = np.where(d["known"], d["wet"].astype(np.float32), np.nan)
                    sctx = self.builder.point_context(st.lat, st.lon, np.ones(len(st), bool), frames, t0,
                                                      self.station_clim, last, last is not None, observed)
                    st_outs.append({k: v.float().cpu() for k, v in self.model.forecast_points(
                        batch, st, torch.from_numpy(sctx[None]).to(self.device), latent=latent).items()})
                if flood:
                    t = self.geo.targets
                    tctx = self.builder.point_context(t.lat, t.lon, t.kind == 0, frames, t0, self.clim,
                                                      obs["targets"][0] if obs.get("targets") is not None else None,
                                                      obs.get("targets") is not None, observed)
                    target_outs.append({k: v.float().cpu() for k, v in self.model.forecast_points(
                        batch, t, torch.from_numpy(tctx[None]).to(self.device), chunk=chunk, latent=latent).items()})
            outs.append({k: v.float().cpu() for k, v in out.items()})
        weights = mixture_weights([c.get("model", "") for c in components])
        drift = None
        if self.meta.get("input_stats") and components:
            from .guard.drift import InputStats

            stats = InputStats.from_json(self.meta["input_stats"])
            first = self.builder.frames_from(components[0]["fields"], components[0]["upper"], t0)
            drift = stats.score(first["hourly"])
        parts = []
        for start in range(0, len(towns), chunk):
            sl = slice(start, min(len(towns), start + chunk))
            parts.append(summarize([_slice_points(o, sl) for o in outs], t0, samples=samples, calibrator=self.calibrator,
                                   weights=weights, corrector=self.corrector, places=self.town_groups[sl],
                                   place_lon=towns.lon[sl]))
        summary = _concat(parts)
        if drift is not None and drift["weight"] > 0:
            hours = t0 + np.arange(summary["p_wet"].shape[1])
            clim_cls = np.moveaxis(self.town_clim.classes_at(hours, towns.lon), 0, 1)
            w = drift["weight"]
            summary["classes"] = (1 - w) * summary["classes"] + w * clim_cls
            summary["p_wet"] = 1 - summary["classes"][..., 0]
        summary["drift"] = drift
        if self.meta.get("tails"):
            from .prob.extremes import GPDTail

            country = self.towns.points.extra.get("country", np.array([""] * len(towns))).astype(str)
            rp = {}
            for kind, amount in (("day", summary["totals"][24]["p90"]), ("hour", summary["bursts"].get(1, {}).get("p90"))):
                if amount is None:
                    continue
                out = np.full(len(towns), np.inf)
                for c in np.unique(country):
                    spec = self.meta["tails"].get(f"{c}|{kind}")
                    if spec:
                        tail = GPDTail(*spec)
                        m = country == c
                        out[m] = np.where(amount[m] > tail.u, tail.return_period_years(amount[m]), np.inf)
                rp[kind] = out
            summary["return_period"] = rp
        from .guard.consistency import check_summary

        summary["repairs"] = check_summary(summary)
        if st_outs:
            summary["airports"] = summarize(st_outs, t0, samples=20, calibrator=self.calibrator, weights=weights)
        if target_outs:
            cells = summarize(target_outs, t0, samples=20, calibrator=self.calibrator, weights=weights)
            catch = self.catchments()
            summary["catchment24"] = catch.catchment_mean(cells["expected_mm"][:, :24].sum(-1))
            from .hydro import curve_number, peak_timing
            from .hydro.runoff import urban_share

            hourly = catch.catchment_mean(cells["expected_mm"].T).T                     # (towns, T)
            if self._urban is None:
                self._urban = urban_share(towns.lat, towns.lon, self.towns.points.extra.get(
                    "population", np.zeros(len(towns))).astype(np.float64))
            cn = curve_number(catch.slope, self._urban)
            summary["flood_peak"] = peak_timing(hourly, cn, catch.upstream_km2, catch.slope)
        return summary

    def run(self, now=None, models=MODELS, use_memory: bool = True, out_dir=None, countries=None,
            airports: bool = True, lightning: bool = True, flood: bool = True, archive: bool = True) -> dict:
        """The live forecast from the current hour (or ``now``); writes the tables when ``out_dir`` is given."""
        t0 = hour_of(now) if now is not None else int(datetime.now(timezone.utc).timestamp() // 3600)
        log(f"forecast from {hour_to_iso(t0)} for {len(self.towns.points):,} places")
        components = self.driving(t0, models)
        if not components:
            raise RuntimeError("no driving model run is available; try again later or pass now= an earlier hour")
        obs = self.observations(t0, rain=use_memory, airports=airports, lightning=lightning)
        try:
            summary = self.from_fields(components, t0, obs, flood=flood)
        except Exception as error:                     # the network failed: issue the statistical fallback
            from .dataset import bilinear
            from .guard.fallback import fallback_summary

            log(f"model forecast failed ({type(error).__name__}: {error}); issuing the statistical fallback")
            towns = self.towns.points
            sch = self.model.schedule
            hf, _ = sch.hour_frame()
            rain = bilinear(np.clip(components[0]["fields"]["tp"], 0, None), self.builder.lat1, self.builder.lon1,
                            towns.lat, towns.lon)[hf].T
            hours = t0 + np.arange(sch.num_hours)
            clim = np.moveaxis(self.town_clim.classes_at(hours, towns.lon), 0, 1)
            summary = fallback_summary(t0, rain, clim, self.meta.get("fallback"))
            flood = False
        notes = sorted({n for c in components for n in c.get("notes", [])}) + list(summary.get("repairs", []))
        if summary.get("drift") and summary["drift"]["weight"] > 0:
            d = summary["drift"]
            notes.append(f"unusual inputs (score {d['score']:.1f}: {', '.join(d['outside'])}); forecast pulled "
                         f"{d['weight']:.0%} towards climatology")
        n_airports = 0 if obs.get("station_codes") is None else int((obs["station_codes"] >= 0).any(0).sum())
        info = {"t0": hour_to_iso(t0), "issued": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                "driving": [{"model": c["model"], "run": hour_to_iso(c["init"])} for c in components],
                "recent_rain": obs.get("product", "not used"), "airports_reporting": n_airports,
                "lightning": obs.get("lightning_note", "not used"), "notes": notes,
                "calibrated": self.calibrator is not None, "online_corrections": len(self.corrector.params),
                "model": self.meta.get("config", {}).get("name", "pluvium")}
        if archive and "airports" in summary and self.model.stations is not None:
            from .online import ForecastArchive

            a = summary["airports"]
            ForecastArchive(self.state_dir / "archive").add(t0, a["p_wet"], a["expected_mm"], self.station_groups,
                                                            self.model.stations.lon)
        if out_dir is not None:
            write_outputs(out_dir, self.towns, summary, info, self.model.schedule, countries,
                          lightning=obs.get("lightning"), terrain=self.geo.terrain)
        return {"info": info, "summary": summary}


def mixture_weights(models: list[str], ensemble_share: float = 2.0) -> np.ndarray:
    """Each deterministic model counts once; the GFS ensemble's members share ``ensemble_share`` between them (as
    much as two deterministic models), so 30 members do not drown out ECMWF."""
    members = sum(m.startswith("gefs") for m in models)
    return np.array([ensemble_share / members if m.startswith("gefs") else 1.0 for m in models], np.float64)


def nearest_target_town(towns: Gazetteer, lat, lon) -> np.ndarray:
    from scipy.spatial import cKDTree

    p = towns.points
    return cKDTree(np.stack([p.lat, p.lon], -1)).query(np.stack([np.asarray(lat), np.asarray(lon)], -1))[1]


# ------------------------------------------------------------------------------------------- tables --

def town_rows(towns: Gazetteer, s: dict, schedule, lightning: np.ndarray | None = None) -> list[dict]:
    from .flood import flood_risk

    p = towns.points
    x = p.extra
    n = len(p)
    t0 = s["t0"]
    zones = x.get("timezone", np.array(["UTC"] * n)).astype(str)
    hf, _ = schedule.hour_frame()
    future = {int(k): i for i, k in enumerate(schedule.future)}
    rows_of_hours = np.array([future[int(k)] for k in hf])
    cause_w = (s["cause"][rows_of_hours[:24]] * s["expected_mm"][:, :24].T[..., None]).sum(0)
    cause = np.where(cause_w.sum(-1) > 1e-6, cause_w.argmax(-1), -1)
    t24 = s["totals"][24]
    thunder24 = any_in_window(s["thunder"], 24) if s.get("thunder") is not None else np.zeros(n)
    heaviest = np.argmax(s["expected_mm"][:, :24], -1)
    catch = s.get("catchment24")
    fl = flood_risk(s.get("bursts", {}), t24, catch)
    rows = []
    for k in range(n):
        words = hour_words(s["classes"][k], None if s.get("thunder") is None else s["thunder"][k],
                           None if s.get("kind") is None else s["kind"][k], s["torrential"][k])
        wet_words = [w for w, pr in zip(words[:24], s["p_wet"][k, :24]) if w != "dry" and pr >= 0.2]
        main = max(set(wet_words), key=wet_words.count).replace("chance of ", "").replace("slight ", "") if wet_words else ""
        start = s["start_median_24"][k]
        start_txt = local_time(t0 + start, zones[k], p.lon[k]) if np.isfinite(start) else ""
        early, late = s["start_early_24"][k], s["start_late_24"][k]
        a_txt, b_txt = local_time(t0 + early, zones[k], p.lon[k]), local_time(t0 + late, zones[k], p.lon[k])
        if a_txt[:3] == b_txt[:3]:                                    # same day: name it once
            b_txt = b_txt[4:]
        window = f"between {a_txt} and {b_txt}" if np.isfinite(early) and late > early else ""
        row = {
            "place": str(x["name"][k]), "parish": str(x.get("unit", [""] * n)[k]), "country": str(x.get("country", [""] * n)[k]),
            "lat": round(float(p.lat[k]), 4), "lon": round(float(p.lon[k]), 4),
            "population": int(x.get("population", np.zeros(n))[k]),
            "rain_now": round(float(s["p_now"][k]), 3), "weather_now": words[0],
            **{f"any_rain_{h}h": round(float(s["any_rain"][h][k]), 3) for h in WINDOWS if h in s["any_rain"]},
            **{f"heavy_rain_{h}h": round(float(s["any_heavy"][h][k]), 3) for h in (6, 24) if h in s.get("any_heavy", {})},
            "thunder_24h": round(float(thunder24[k]), 3), "main_weather": main,
            "likely_start": start_txt if s["any_rain"][24][k] >= 0.15 else "", "start_window": window,
            "start_hours": None if not np.isfinite(start) else int(start),
            "heaviest_hour": local_time(t0 + heaviest[k], zones[k], p.lon[k]),
            "if_raining_stops_in_h": None if not np.isfinite(s["stop_median"][k]) else int(s["stop_median"][k]),
            "rain_24h_mm_median": round(float(t24["median"][k]), 1), "rain_24h_mm_p90": round(float(t24["p90"][k]), 1),
            **{f"p_24h_ge_{int(v)}mm": round(float(t24[f"p_ge_{int(v)}"][k]), 3) for v in TOTAL_THRESHOLDS},
            "wettest_3h_mm_p90": round(float(s["bursts"][3]["p90"][k]), 1) if 3 in s.get("bursts", {}) else None,
            "catchment_rain_24h_mm": None if catch is None else round(float(catch[k]), 1),
            "flood_risk": str(fl["text"][k]), "flood_reason": str(fl["reason"][k]),
            "flood_peak": (local_time(t0 + s["flood_peak"]["peak_hour"][k], zones[k], p.lon[k])
                           if s.get("flood_peak") and np.isfinite(s["flood_peak"]["peak_hour"][k]) else ""),
            "runoff_24h_mm": (round(float(s["flood_peak"]["runoff_mm"][k]), 1) if s.get("flood_peak") else None),
            "rain_24h_p90_return_years": (None if not s.get("return_period") or "day" not in s["return_period"]
                                          or not np.isfinite(s["return_period"]["day"][k])
                                          else round(float(s["return_period"]["day"][k]), 1)),
            "lightning_now": 0 if lightning is None else int(lightning[k]),
            "cause": REGIMES[cause[k]] if cause[k] >= 0 else "-",
        }
        row["summary"] = town_sentence(row)
        rows.append(row)
    return rows


def parish_rows(rows: list[dict]) -> list[dict]:
    from .flood import LEVELS

    groups: dict = {}
    for r in rows:
        groups.setdefault((r["country"], r["parish"]), []).append(r)
    out = []
    for (country, parish), rs in sorted(groups.items()):
        pop = np.array([max(r["population"], 1) for r in rs], np.float64)
        p24 = np.array([r["any_rain_24h"] for r in rs])
        top = int(np.argmax(p24))
        starts = [(r["start_hours"], r["likely_start"]) for r in rs if r["start_hours"] is not None and r["any_rain_24h"] >= 0.5]
        first = min(starts) if starts else (None, "")
        causes = [r["cause"] for r in rs if r["cause"] != "-"]
        flood = max(LEVELS.index(r["flood_risk"]) for r in rs)
        row = {"country": country, "parish": parish, "places": len(rs),
               "any_rain_24h_typical": round(float((p24 * pop).sum() / pop.sum()), 3),
               "any_rain_24h_highest": round(float(p24[top]), 3), "wettest_place": rs[top]["place"],
               "earliest_likely_start": first[1], "thunder_24h_highest": max(r["thunder_24h"] for r in rs),
               "rain_24h_mm_p90_highest": max(r["rain_24h_mm_p90"] for r in rs), "flood_highest": LEVELS[flood],
               "main_cause": max(set(causes), key=causes.count) if causes else "-"}
        row["summary"] = parish_sentence(row)
        out.append(row)
    return out


def report_text(info: dict, parishes: list[dict], countries=None) -> str:
    lines = [f"Pluvium rain forecast from {info['t0']} (issued {info['issued']})",
             "driving models: " + ", ".join(f"{d['model'].upper()} {d['run']}" for d in info["driving"]),
             f"rain already falling from: {info['recent_rain']}; airports reporting: {info.get('airports_reporting', 0)}; "
             f"lightning: {info.get('lightning', 'not used')}",
             "Research model: not an official forecast. Follow your national meteorological service.", ""]
    for r in parishes:
        if countries and r["country"] not in countries:
            continue
        cause = f" Mainly {REGIME_TEXT.get(r['main_cause'], r['main_cause'])}." if r["main_cause"] != "-" else ""
        lines.append(f"{r['country']} {r['parish'] or '(no parish)'}: {r['summary']}{cause}")
    return "\n".join(lines) + "\n"


def write_outputs(out_dir, towns: Gazetteer, summary: dict, info: dict, schedule, countries=None,
                  lightning: np.ndarray | None = None, terrain=None) -> Path:
    d = Path(out_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    rows = town_rows(towns, summary, schedule, lightning)
    with open(d / "towns.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    parishes = parish_rows(rows)
    with open(d / "parishes.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parishes[0]))
        writer.writeheader()
        writer.writerows(parishes)
    features = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                 "properties": {k: v for k, v in r.items() if k not in ("lat", "lon")}} for r in rows]
    (d / "towns.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    np.savez_compressed(d / "hourly.npz", t0=summary["t0"], p_wet=summary["p_wet"].astype(np.float16),
                        classes=summary["classes"].astype(np.float16), expected_mm=summary["expected_mm"].astype(np.float16),
                        thunder=(summary["thunder"] if summary.get("thunder") is not None
                                 else np.zeros_like(summary["p_wet"])).astype(np.float16),
                        place=towns.names.astype(str))
    (d / "forecast.json").write_text(json.dumps({**info, "parishes": parishes}, indent=1, default=str))
    (d / "report.txt").write_text(report_text(info, parishes, countries), encoding="utf-8")
    from .render import bulletins, cap_alerts

    for country in sorted({r["country"] for r in parishes}):
        if countries and country not in countries:
            continue
        for name, text in bulletins(parishes, country, info["issued"]).items():
            (d / name).write_text(text, encoding="utf-8")
    alerts = cap_alerts(parishes, rows, info["issued"] + ":00")
    if alerts:
        (d / "alerts").mkdir(exist_ok=True)
        for k, xml in enumerate(alerts):
            (d / "alerts" / f"alert_{k:03d}.xml").write_text(xml, encoding="utf-8")
    if terrain is not None:
        from .render import draw_map

        lat = np.array([r["lat"] for r in rows])
        lon = np.array([r["lon"] for r in rows])
        p24 = np.array([r["any_rain_24h"] for r in rows])
        box = (float(lat.min()) - 0.3, float(lat.max()) + 0.3, float(lon.min()) - 0.3, float(lon.max()) + 0.3)
        draw_map(d / "map_rain_24h.png", terrain, lat, lon, p24, box, f"Chance of rain in the next 24 h from {info['t0']}")
    log(f"wrote {len(rows):,} places and {len(parishes):,} parishes to {d}")
    return d


def place_timeline(towns: Gazetteer, summary: dict, query: str, country: str | None = None, hours: int = 72) -> str:
    """Hour-by-hour text for one place (the best match of ``query``)."""
    idx = towns.find(query, country)
    if idx.size == 0:
        return f"no place matching {query!r}"
    k = int(idx[0])
    p = towns.points
    zone = str(p.extra.get("timezone", np.array(["UTC"] * len(p)))[k])
    t0 = summary["t0"]
    words = hour_words(summary["classes"][k], None if summary.get("thunder") is None else summary["thunder"][k],
                       None if summary.get("kind") is None else summary["kind"][k], summary["torrential"][k])
    lines = [f"{towns.label(k)} (local time {zone})"]
    for h in range(min(hours, summary["p_wet"].shape[1])):
        pr = float(summary["p_wet"][k, h])
        mm = float(summary["expected_mm"][k, h])
        th = "" if summary.get("thunder") is None else f"  thunder {float(summary['thunder'][k, h]):3.0%}"
        lines.append(f"  {local_time(t0 + h, zone, p.lon[k]):>10}  {pr:4.0%}  {'#' * int(round(pr * 20)):<20} "
                     f"{mm:4.1f} mm  {words[h]}{th}")
    return "\n".join(lines)


__all__ = ["Forecaster", "MODELS", "groups_of", "mixture_weights", "parish_rows", "place_timeline", "report_text", "town_rows",
           "write_outputs"]
