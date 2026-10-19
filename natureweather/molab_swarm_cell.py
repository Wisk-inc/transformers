# NatureV1 0.16 -- the operational cell: the satellite path, the swarm, the parishes, the cycle record, in ONE marimo cell
# (molab, 1 GPU; runs on a CPU box too, smaller).
#
# Paste this whole file into one marimo cell and run it. What it does:
#   1. installs naturev1 0.16 with the swarm extras (GRIB decoder, terrain tiles, maps, the Hub client)
#   2. builds the map of the whole Caribbean -- every parish / province / district of every country and territory,
#      with terrain -- and the full-consensus inputs (climatology + analog database), once, cached under DATA
#   3. for every active NHC storm (or a replay, e.g. Melissa 2025): the best track + every satellite / recon / radar
#      centre fix from NHC, the GFS analysis AND the forecasts of five models -- GFS, ECMWF IFS, ECMWF AIFS,
#      GraphCast-GFS, the GEFS mean -- read from their public buckets; the NatureV1 consensus blended with the models'
#      mean track and fed the previous cycle's error; a cone sized for this forecast and a swarm sized to it; the
#      intensity consensus with its rapid-intensification branch, the chance the storm peaks at each category and the
#      calibrated peak quantiles (plan for the 90% figure); the swarm's worlds filtered minute by minute on the fixes
#      and painted onto the parishes: peak wind if hit, gusts, rain if hit and the wettest cell, when tropical-storm
#      and hurricane force arrive, the landfall report for LANDFALL_COUNTRIES
#   4. the satellite, quantitatively: the newest GOES band-13 scene shown, and read -- coldest tops, the cold share of
#      the 50-200 km annulus, ring symmetry, an eye, and their trend since the previous cycle (diagnostics beside
#      the forecast; not yet inside the probabilities)
#   5. the operations desk's record: every cycle saved under DATA/runs/<storm>/ as JSON plus the parish table as CSV;
#      the previous cycle found again automatically; what changed since it; optionally keeps watching NHC and
#      re-issues when a new 6-hourly cycle appears (WATCH_MINUTES)
#   6. optionally (TRAIN_SATELLITE) trains the satellite networks of the 0.12 satellite path on this GPU
#
# Research model, not an operational forecast: for decisions, follow your national meteorological service and the
# National Hurricane Center.
import datetime as dt
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "naturev1[swarm,hub]>=0.16.0", "ihelix>=0.6.0"],
               check=True)

from pathlib import Path

import marimo as mo
import numpy as np
import torch

# ── settings ──────────────────────────────────────────────────────────────────────────────────────────────────────
STORM = None                   # None: every active NHC storm. Or an NHC id, e.g. "al132025" (Melissa 2025)
REPLAY_TIME = None             # with STORM: replay what was known then, e.g. "2025-10-27T00" (fixes after it unused)
LANDFALL_COUNTRIES = ("Jamaica",)   # one landfall report per country: when, where first, how strong, every parish's numbers
GPU = torch.cuda.is_available()
PARTICLES = 1_000_000 if GPU else 50_000          # worlds in the swarm
IMPACT_WORLDS = 40_000 if GPU else 2_000          # worlds painted onto the map (the map is the costly part)
FULL_CONSENSUS = True          # 13-member consensus: builds a 2.5 GB climatology + analog database once (~20-30 min)
SHOW_GOES = True               # the newest GOES-19 / GOES-18 infrared image around each live storm, shown and read
FORECAST_FLOW = True           # the models' forecast steering + their tracks as members (~150 MB per model, cached)
MODELS = ("gfs", "ifs", "aifs", "graphcastgfs", "gefs")   # any that is not yet published for the cycle is skipped
PREVIOUS_CYCLE = "auto"        # the previous cycle's error feeds the new track ("auto": from the saved records; None: off)
WATCH_MINUTES = 0              # 0: issue once and stop. 20: keep watching NHC, re-issue when a new 6-hourly cycle appears
TRAIN_SATELLITE = False        # the 0.12 satellite networks on this GPU (needs ~1 TB disk for the staged GridSat)
HF_REPO, HF_TOKEN = "CorxLabs/NatureV1-500", ""   # only for TRAIN_SATELLITE pushes; paste a token, clear it after
DATA = Path(os.environ.get("NATUREV1_DATA", "/data/naturev1" if Path("/data").exists() else Path.home() / "naturev1"))
DATA.mkdir(parents=True, exist_ok=True)
RUNS = DATA / "runs"                                # the cycle records: <storm>/<time>.json + <time>_parishes.csv
DEVICE = "cuda" if GPU else "cpu"

import naturev1
from naturev1.satellite import SatelliteScene
from naturev1.swarm.bulletin import history, load_previous, write_bulletin
from naturev1.swarm.fixes import active_storms, fetch_deck, parse_bdeck
from naturev1.swarm.geography import CARIBBEAN, RegionMap
from naturev1.swarm.infrared import ir_metrics
from naturev1.swarm.landfall import format_landfall, landfall_summary
from naturev1.swarm.live import consensus_inputs, forecast_storm, goes_scene
from naturev1.swarm.plot import impact_figure, landfall_figure

print(f"naturev1 {naturev1.__version__} on {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if GPU else ""))

# ── the ground: the whole Caribbean, parish by parish, with terrain (downloaded once) ─────────────────────────────
region_file = DATA / "caribbean_region.npz"
if not region_file.exists():
    RegionMap.build(CARIBBEAN, cache=DATA / "region").save(region_file)
region = RegionMap.load(region_file)
print(f"map: {len(region.names)} administrative units, {int(region.land.sum()):,} land cells")

climatology = analogs = None
if FULL_CONSENSUS:
    climatology, analogs = consensus_inputs(DATA)


# ── helpers ───────────────────────────────────────────────────────────────────────────────────────────────────────
def newest_cycle(storm_id):
    """The newest 6-hourly best-track time NHC has posted for a storm (what a new issue would be based on)."""
    best = parse_bdeck(fetch_deck(storm_id, "b", False))
    synoptic = [f.time for f in best if int(str(f.time)[11:13]) % 6 == 0 and str(f.time)[14:16] == "00"]
    return max(synoptic) if synoptic else None


def goes_and_infrared(case, storm_id):
    """The newest GOES scene around the storm and its infrared diagnostics, with the trend since the scene kept last time."""
    scene = goes_scene(case.lat, case.lon, DATA / "goes")
    keep = DATA / "goes" / f"{storm_id}_previous.npz"
    previous, hours = None, None
    if keep.exists():
        try:
            z = np.load(keep)
            when = dt.datetime.fromisoformat(str(z["time"]))
            hours = (scene.timestamp - when).total_seconds() / 3600.0
            previous = SatelliteScene(latitude=z["lat"], longitude=z["lon"], values=z["bt"][:, None], area_km2=z["area"],
                                      channels=("C13",), timestamp=when, satellite=str(z["satellite"]))
        except Exception:
            previous, hours = None, None
    ir = ir_metrics(scene, case.lat, case.lon, previous=previous if hours and 0.5 <= hours <= 12.0 else None, previous_hours=hours)
    np.savez(keep, lat=scene.latitude, lon=scene.longitude, bt=scene.values[:, 0], area=scene.area_km2,
             time=str(scene.timestamp), satellite=str(scene.satellite))
    return scene, ir


def since_previous(storm_id, run, summary):
    """One line on what moved since the previous saved cycle: landfall probability and time, intensity, the ceiling."""
    prev = load_previous(RUNS, storm_id, run["analysis_time"])
    if prev is None:
        return ""
    rec = prev["record"]
    bits = [f"since the {str(prev['time'])[5:16]}Z cycle:"]
    lf = rec.get("landfall") or {}
    if summary and lf:
        bits.append(f"landfall probability {lf.get('p_landfall', float('nan')):.0%} -> {summary['p_landfall']:.0%}")
        if lf.get("consensus", {}).get("time") and summary.get("consensus", {}).get("time"):
            bits.append(f"consensus landfall {lf['consensus']['time'][5:16]} -> {summary['consensus']['time'][5:16]}")
    fc = rec.get("forecast", {})
    if fc.get("intensity") and len(fc["intensity"]) > 2 and fc["intensity"][2] is not None:
        bits.append(f"24 h intensity {fc['intensity'][2]:.0f} -> {run['result']['intensity'][2]:.0f} kt")
    pq_old, pq_new = fc.get("peak_quantiles"), run["result"].get("peak_quantiles")
    if pq_old and pq_new:
        bits.append(f"90% peak ceiling {pq_old['peak_kt'][2]:.0f} -> {pq_new['peak_kt'][2]:.0f} kt")
    return "  \n**" + " ".join(bits[:1]) + "** " + "; ".join(bits[1:])


def issue(storm_id, name, when):
    """One cycle for one storm: the forecast, the panels, the record. Returns (panel, analysis time)."""
    run = forecast_storm(storm_id, when=when, archive=when is not None, particles=PARTICLES, device=DEVICE, region=region,
                         climatology=climatology, analogs=analogs, impact_particles=IMPACT_WORLDS, cache=DATA, name=name,
                         forecast_flow=FORECAST_FLOW, models=MODELS, previous=PREVIOUS_CYCLE, save_runs=True)
    case, result, impact = run["case"], run["result"], run["impact"]
    scene, ir = None, None
    if SHOW_GOES and when is None:
        try:
            scene, ir = goes_and_infrared(case, storm_id)
        except Exception as error:
            print(f"GOES: {error}")
    fig = impact_figure(run, scene=scene)
    landfall_panels, main_summary = [], None
    if impact is not None:
        for country in LANDFALL_COUNTRIES:
            summary = landfall_summary(region, result, case, impact, country, analysis_time=run["analysis_time"])
            main_summary = main_summary or summary
            if summary["p_landfall"] >= 0.02:
                landfall_panels += [landfall_figure(run, summary, country), mo.plain_text(format_landfall(summary, top=14))]
            else:
                landfall_panels.append(mo.md(f"**{country}:** the centre reaches it in {summary['p_landfall']:.0%} of worlds within 5 days."))
    record = write_bulletin(run, main_summary, RUNS, storm_id)                      # the full record (replaces the bare one)
    sw = result["swarm"]
    scen = ", ".join(f"{k} {v['probability']:.0%}" for k, v in sw["scenarios"].items())
    filt = sw["filter"]
    sources = sorted({h.get("source", "best track") for h in filt})
    ess_after = f"; re-weighted by {len(run['fixes_after'])} newer fixes (ESS {run['ess_after']:.2f})" if run["ess_after"] else ""
    track = "  ".join(f"{h} h {la:.1f}N {abs(lo):.1f}{'W' if lo < 0 else 'E'}"
                      for h, (la, lo) in zip(result["leads"], result["track"]) if np.isfinite(la))
    intensity = "  ".join(f"{h} h {v:.0f} kt" for h, v in zip(result["leads"], result["intensity"]) if np.isfinite(v))
    p_ri = result.get("ri_probability")
    ri = f"  \n**Rapid intensification (+30 kt in 24 h):** {p_ri:.0%} ({result.get('intensity_regime')} weights)" if p_ri is not None else ""
    names = {64: "hurricane", 96: "major", 113: "Cat 4", 137: "Cat 5"}
    pk = result.get("peak_probability")
    if pk:
        ri += "  \n**Chance the storm peaks at or above (next 5 days):** " + "; ".join(f"{names.get(int(t), f'{t:.0f} kt')} {p:.0%}" for t, p in zip(pk["thresholds_kt"], pk["probability"]))
    pq = result.get("peak_quantiles")
    if pq:
        ri += "  \n**Peak within 5 days, calibrated quantiles:** " + "; ".join(f"{int(round(l * 100))}% {v:.0f} kt" for l, v in zip(pq["levels"], pq["peak_kt"])) + " (plan for the 90% figure)"
    if result.get("peak_if_ri_kt") is not None and p_ri is not None:
        ri += f"  \n**If it intensifies rapidly ({p_ri:.0%}):** expected peak {result['peak_if_ri_kt']:.0f} kt"
    cats = result.get("category_probability")
    if cats:
        ri += "  \n**P(at least ...):** " + "; ".join(
            f"{names.get(int(t), f'{t:.0f} kt')} " + " ".join(f"{h}h {p:.0%}" for h, p in zip(result["leads"], cats["probability"][:, j]) if np.isfinite(p))
            for j, t in enumerate(cats["thresholds_kt"]))
    if result.get("intensity_note"):
        ri += f"  \n**Note:** {result['intensity_note']}"
    envt = result.get("environment") or {}
    if "tchp_kj_cm2" in envt and np.isfinite(envt["tchp_kj_cm2"][:49]).any():
        ri += (f"  \n**Ocean under the first 48 h:** 26 C water {np.nanmean(envt['d26_m'][:49]):.0f} m deep, heat potential "
               f"{np.nanmean(envt['tchp_kj_cm2'][:49]):.0f} kJ/cm2, sea surface {np.nanmean(envt['sst_k'][:49]) - 273.15:.1f} C "
               f"(effective {np.nanmean(envt['sst_eff_k'][:49]) - 273.15:.1f} C under the core)")
    if run.get("r34_km") is not None and np.isfinite(run["r34_km"]):
        ri += f"  \n**Observed 34-kt radius:** {run['r34_km']:.0f} km (the wind field is sized to it)"
    deep = result.get("model_deepening_hpa")
    if deep is not None and np.isfinite(deep).any():
        ri += "  \n**Models' deepening (hPa, central-pressure deficit):** " + "  ".join(f"{h}h {d:+.0f}" for h, d in zip(result["leads"], deep) if np.isfinite(d))
    if ir:
        ri += f"  \n**{ir['text']}**"
    guide = run.get("guidance") or {}
    if guide.get("available"):
        used_t = (result.get("aids_track") or {}).get("members", [])
        used_i = (result.get("aids_intensity") or {}).get("members", [])
        subs = {**(result.get("aids_track") or {}).get("substituted", {}), **(result.get("aids_intensity") or {}).get("substituted", {})}
        ri += (f"  \n**NHC guidance ({len(guide['available'])} aids in the a-deck):** track blended with {', '.join(used_t) or 'none'}; "
               f"intensity with {', '.join(used_i) or 'none'}" + (f"; stand-ins {', '.join(f'{v} for {k}' for k, v in subs.items())}" if subs else ""))
        if result.get("intensity_ours") is not None:
            ri += "  \n**Our own consensus before the aids:** " + "  ".join(f"{h} h {v:.0f} kt" for h, v in zip(result["leads"], result["intensity_ours"]) if np.isfinite(v))
        at = result.get("aids_track") or {}
        if at.get("disagreement_km") is not None:
            dis = np.asarray(at["disagreement_km"], float); cone = np.asarray(result.get("cone_km", np.full(len(dis), np.nan)), float)
            far = [f"{h} h {d:.0f} km (cone {c:.0f})" for h, d, c in zip(result["leads"], dis, cone) if np.isfinite(d) and np.isfinite(c) and d > c]
            ri += ("  \n**Our track and HCCA disagree beyond the cone at " + "; ".join(far) + " -- both are in the record; the blend sits between them**") if far else \
                  "  \n**Our track and HCCA agree within the cone at every lead**"
        if at.get("adaptive_w_ours") is not None:
            pe = at.get("previous_errors_km", {})
            ri += (f"  \n**Weights earned from the previous cycle:** ours {pe.get('ours', float('nan')):.0f} km off, HCCA {pe.get('HCCA', float('nan')):.0f} km off at the time now analysed; "
                   f"our share {np.asarray(at['adaptive_w_ours'])[2]:.0%} at 24 h")
        of = guide.get("ofcl") or {}
        if of.get("intensity") is not None and of.get("track") is not None:
            ri += "  \n**NHC official forecast:** " + "  ".join(
                f"{h} h {la:.1f}N {abs(lo):.1f}{'W' if lo < 0 else 'E'} {v:.0f} kt"
                for h, (la, lo), v in zip(result["leads"], of["track"], of["intensity"]) if np.isfinite(la) and np.isfinite(v))
        if guide.get("nhc_ri") and guide["nhc_ri"].get("p30_24") is not None:
            schemes = "; ".join(f"{k} {v:.0%}" for k, v in (guide["nhc_ri"].get("schemes") or {}).items() if v is not None)
            ri += f"  \n**NHC rapid-intensification guidance:** {guide['nhc_ri']['p30_24']:.0%} ({guide['nhc_ri']['source']}; {schemes})"
            if guide.get("ri_probability_combined") is not None:
                ri += f"; equal-weight with ours: **{guide['ri_probability_combined']:.0%}**"
    used = run.get("models", {})
    mm = run.get("model_track")
    model_text = ((f"  \n**Models read:** {', '.join(m['title'] for m in used.values())}; their mean track: "
                   + "  ".join(f"{h} h {la:.1f}N {abs(lo):.1f}{'W' if lo < 0 else 'E'}" for h, (la, lo) in zip(result["leads"], mm) if np.isfinite(la)))
                  if mm is not None else "  \n**Models:** none read for this cycle (analysis-only forecast)")
    spread = result.get("model_spread_km")
    if spread is not None and np.isfinite(spread).any():
        model_text += "  \n**Models' disagreement (km):** " + "  ".join(f"{h} h {d:.0f}" for h, d in zip(result["leads"], spread) if np.isfinite(d))
    if run.get("previous_used") is not None:
        model_text += f"  \n**Previous cycle:** {str(run['previous_used'])[:16]}Z, its error fed back into this track"
    cone = result.get("cone_km")
    cone_text = ("  \n**Cone for this forecast (km, 67%):** " + "  ".join(f"{h} h {r:.0f}" for h, r in zip(result["leads"], cone))
                 + (" (climatological cone " + "/".join(f"{r:.0f}" for r in result["cone_fixed_km"]) + ")" if "cone_fixed_km" in result else "")) if cone is not None else ""
    drift = since_previous(storm_id, run, main_summary)
    record_text = f"  \n**Record:** {record['json']}" + (f", parishes {record['csv'].name}" if record.get("csv") else "") + f"; {len(history(RUNS, storm_id))} cycle(s) on file"
    panel = mo.vstack([
        mo.md(f"## {name} ({storm_id}) -- analysis {str(run['analysis_time'])[:16]}Z, {case.wind_kt:.0f} kt at {case.lat:.1f}N {abs(case.lon):.1f}W\n"
              f"**Satellite path:** {len(filt)} fixes filtered the swarm ({', '.join(sources[:6])}){ess_after}.  \n"
              f"**Swarm:** {PARTICLES:,} worlds sized to the cone; scenarios: {scen}.  \n"
              f"**Track** ({result['consensus_table']}): {track}{model_text}{cone_text}  \n**Intensity:** {intensity}{ri}{drift}{record_text}"),
        fig,
        *landfall_panels,
        mo.plain_text(impact.table(top=30)) if impact is not None else mo.md("no impact map for this storm"),
        mo.md(f"<sub>{region.attribution} Research model, not an operational forecast. Storm surge is not modelled.</sub>"),
    ])
    return panel, run["analysis_time"]


# ── the storms: once, or watched ─────────────────────────────────────────────────────────────────────────────────
panels, issued = [], {}
while True:
    if STORM is None:
        storms = [(s["id"], s["name"]) for s in active_storms() if s["id"][:2] in ("al", "ep", "cp")]
    else:
        storms = [(STORM, STORM.upper())]
    if not storms and not panels:
        panels.append(mo.md("**No active storms in the Atlantic or eastern / central Pacific right now.** "
                            "Set `STORM = \"al132025\"` and `REPLAY_TIME = \"2025-10-27T00\"` to replay Hurricane Melissa."))
    for storm_id, name in storms:
        try:
            if WATCH_MINUTES and REPLAY_TIME is None and issued.get(storm_id) is not None:
                newest = newest_cycle(storm_id)
                if newest is None or newest <= issued[storm_id]:
                    continue                                                  # nothing new from NHC for this storm
            panel, when_issued = issue(storm_id, name, REPLAY_TIME)
            if issued.get(storm_id) == when_issued:
                continue                                                      # the GFS for the newer cycle is not out yet
            issued[storm_id] = when_issued
            panels.append(panel)
            mo.output.append(panel)
        except Exception as error:                                            # one storm failing must not hide the others
            panels.append(mo.md(f"**{name} ({storm_id})**: no forecast -- {type(error).__name__}: {error}"))
            mo.output.append(panels[-1])
    if not WATCH_MINUTES or REPLAY_TIME is not None:
        break
    mo.output.append(mo.md(f"<sub>watching NHC; next check at {(dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=WATCH_MINUTES)):%H:%M} UTC</sub>"))
    time.sleep(WATCH_MINUTES * 60)

# ── optional: the satellite networks of the 0.12 satellite path, on this one GPU ─────────────────────────────────
if TRAIN_SATELLITE:
    from naturev1.b300 import follow, start_job
    from naturev1.ibtracs import fetch_ibtracs, parse_ibtracs, storm_label_table

    if HF_TOKEN.strip():
        os.environ["HF_TOKEN"] = HF_TOKEN.strip()
    SAT, CKPT, LOGS = DATA / "satellite", DATA / "checkpoints", DATA / "logs"
    for folder in (SAT, CKPT, LOGS):
        folder.mkdir(parents=True, exist_ok=True)
    ibtracs = fetch_ibtracs(DATA)
    labels = SAT / "labels.npz"
    if not labels.exists():
        np.savez(labels, **storm_label_table(parse_ibtracs(ibtracs, since=1980, until=2021)))
    from huggingface_hub import hf_hub_download
    base = DATA / "base.pt"
    if not base.exists():
        import shutil
        shutil.copy2(hf_hub_download(HF_REPO, "checkpoints/stage3/latest.pt"), base)
    start_job("gridsat", [sys.executable, "-m", "naturev1.satstage", "--root", str(SAT), "--ibtracs", str(ibtracs),
                          "--years", "1980-2021", "--workers", str(max((os.cpu_count() or 8) - 2, 2)),
                          "--resolution", "0.25"], LOGS)
    follow(LOGS, ["gridsat"])
    common = ["--checkpoint", str(base), "--root", str(SAT), "--table", str(labels), "--output", str(CKPT),
              "--hours", "2.0", "--hub-repo", HF_REPO]
    start_job("stage-a-satnet", ["-m", "naturev1.sattrain", "satnet", *common, "--workers", "8"], LOGS, gpus=[0])
    follow(LOGS, ["stage-a-satnet"])
    panels.append(mo.md(f"satellite stage A finished: {CKPT / 'stage-a-satnet'}"))

mo.vstack(panels)
