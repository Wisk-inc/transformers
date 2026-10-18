# NatureV1 0.15 -- the satellite path, the swarm sandbox and Caribbean impacts, in ONE marimo cell (molab, 1 GPU).
#
# Paste this whole file into one marimo cell and run it. What it does:
#   1. installs naturev1 0.15 with the swarm extras (GRIB decoder, terrain tiles, maps)
#   2. builds the map of the whole Caribbean -- every parish / province / district of every country and territory,
#      with terrain -- and the full-consensus inputs (climatology + analog database), once, cached under DATA
#   3. for every active NHC storm (or a replay, e.g. Melissa 2025): best track + every satellite / recon / radar
#      centre fix from NHC, the GFS analysis AND the forecasts of five models -- GFS, ECMWF IFS, ECMWF AIFS,
#      GraphCast-GFS, the GEFS mean -- read from their public buckets (the turns: a trough two days out bends the
#      track; several models: their errors partly cancel), the NatureV1 consensus blended with the models' mean
#      track, a cone sized for this forecast, the intensity consensus that switches to its rapid-intensification
#      weights when P(RI) is high, the swarm whose worlds each follow one model's forecast winds, filtered minute by
#      minute on the fixes and re-weighted by every fix since, and each world's wind and rain painted onto the
#      parishes
#   4. shows the map, the 4D worldlines, the latest GOES infrared image and the parish-by-parish impact table
#   5. optionally (TRAIN_SATELLITE) trains the satellite networks of the 0.12 satellite path on this GPU
#
# Research model, not an operational forecast: for decisions, follow your national meteorological service and the
# National Hurricane Center.

import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "naturev1[swarm]>=0.15.0", "ihelix>=0.6.0"],
               check=True)

from pathlib import Path

import marimo as mo
import numpy as np
import torch

# ── settings ──────────────────────────────────────────────────────────────────────────────────────────────────────
STORM = None                   # None: every active NHC storm. Or an NHC id, e.g. "al132025" (Melissa 2025)
REPLAY_TIME = None             # with STORM: replay what was known then, e.g. "2025-10-27T00" (fixes after it unused)
LANDFALL_COUNTRY = "Jamaica"   # the parish-by-parish landfall panel: when, where first, how strong, each parish's peak
GPU = torch.cuda.is_available()
PARTICLES = 1_000_000 if GPU else 50_000          # worlds in the swarm
IMPACT_WORLDS = 40_000 if GPU else 2_000          # worlds painted onto the map (the map is the costly part)
FULL_CONSENSUS = True          # 13-member consensus: builds a 2.5 GB climatology + analog database once (~20-30 min)
SHOW_GOES = True               # the newest GOES-19 / GOES-18 infrared image around each live storm
FORECAST_FLOW = True           # the models' forecast steering + their tracks as members (~150 MB per model, cached)
MODELS = ("gfs", "ifs", "aifs", "graphcastgfs", "gefs")   # any that is not yet published for the cycle is skipped
TRAIN_SATELLITE = False        # the 0.12 satellite networks on this GPU (needs ~1 TB disk for the staged GridSat)
HF_REPO, HF_TOKEN = "CorxLabs/NatureV1-500", ""   # only for TRAIN_SATELLITE pushes; paste a token, clear it after
DATA = Path(os.environ.get("NATUREV1_DATA", "/data/naturev1" if Path("/data").exists() else Path.home() / "naturev1"))
DATA.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if GPU else "cpu"

import naturev1
from naturev1.swarm.fixes import active_storms
from naturev1.swarm.geography import CARIBBEAN, RegionMap
from naturev1.swarm.live import consensus_inputs, forecast_storm, goes_scene
from naturev1.swarm.landfall import format_landfall, landfall_summary
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

# ── the storms ───────────────────────────────────────────────────────────────────────────────────────────────────
if STORM is None:
    storms = [(s["id"], s["name"]) for s in active_storms() if s["id"][:2] in ("al", "ep", "cp")]
else:
    storms = [(STORM, STORM.upper())]
panels = []
if not storms:
    panels.append(mo.md("**No active storms in the Atlantic or eastern / central Pacific right now.** "
                        "Set `STORM = \"al132025\"` and `REPLAY_TIME = \"2025-10-27T00\"` to replay Hurricane Melissa."))
for storm_id, name in storms:
    try:
        run = forecast_storm(storm_id, when=REPLAY_TIME, archive=REPLAY_TIME is not None, particles=PARTICLES,
                             device=DEVICE, region=region, climatology=climatology, analogs=analogs,
                             impact_particles=IMPACT_WORLDS, cache=DATA, name=name, forecast_flow=FORECAST_FLOW,
                             models=MODELS)
    except Exception as error:                       # one storm failing must not hide the others
        panels.append(mo.md(f"**{name} ({storm_id})**: no forecast -- {type(error).__name__}: {error}"))
        continue
    case, result, impact = run["case"], run["result"], run["impact"]
    scene = None
    if SHOW_GOES and REPLAY_TIME is None:
        try:
            scene = goes_scene(case.lat, case.lon, DATA / "goes")
        except Exception as error:
            print(f"GOES: {error}")
    fig = impact_figure(run, scene=scene)
    landfall_panels = []
    if impact is not None and LANDFALL_COUNTRY:
        summary = landfall_summary(region, result, case, impact, LANDFALL_COUNTRY, analysis_time=run["analysis_time"])
        if summary["p_landfall"] >= 0.02:
            landfall_panels = [landfall_figure(run, summary, LANDFALL_COUNTRY), mo.plain_text(format_landfall(summary, top=14))]
        else:
            landfall_panels = [mo.md(f"**{LANDFALL_COUNTRY}:** the centre reaches it in {summary['p_landfall']:.0%} of worlds within 5 days.")]
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
    cats = result.get("category_probability")
    if cats:
        names = {64: "hurricane", 96: "major", 113: "Cat 4", 137: "Cat 5"}
        ri += "  \n**P(at least ...):** " + "; ".join(
            f"{names.get(int(t), f'{t:.0f} kt')} " + " ".join(f"{h}h {p:.0%}" for h, p in zip(result["leads"], cats["probability"][:, j]) if np.isfinite(p))
            for j, t in enumerate(cats["thresholds_kt"]))
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
    used = run.get("models", {})
    gfs = run.get("model_track")
    gfs_text = ((f"  \n**Models read:** {', '.join(m['title'] for m in used.values())}; their mean track: "
                 + "  ".join(f"{h} h {la:.1f}N {abs(lo):.1f}{'W' if lo < 0 else 'E'}" for h, (la, lo) in zip(result["leads"], gfs) if np.isfinite(la)))
                if gfs is not None else "  \n**Models:** none read for this cycle (analysis-only forecast)")
    spread = result.get("model_spread_km")
    if spread is not None and np.isfinite(spread).any():
        gfs_text += "  \n**Models' disagreement (km):** " + "  ".join(f"{h} h {d:.0f}" for h, d in zip(result["leads"], spread) if np.isfinite(d))
    cone = result.get("cone_km")
    cone_text = ("  \n**Cone for this forecast (km, 67%):** " + "  ".join(f"{h} h {r:.0f}" for h, r in zip(result["leads"], cone))
                 + (" (climatological cone " + "/".join(f"{r:.0f}" for r in result["cone_fixed_km"]) + ")" if "cone_fixed_km" in result else "")) if cone is not None else ""
    panels.append(mo.vstack([
        mo.md(f"## {name} ({storm_id}) -- analysis {str(run['analysis_time'])[:16]}Z, {case.wind_kt:.0f} kt\n"
              f"**Satellite path:** {len(filt)} fixes filtered the swarm ({', '.join(sources[:6])}){ess_after}.  \n"
              f"**Swarm:** {PARTICLES:,} worlds; scenarios: {scen}.  \n"
              f"**Track** ({result['consensus_table']}): {track}{gfs_text}{cone_text}  \n**Intensity:** {intensity}{ri}"),
        fig,
        *landfall_panels,
        mo.plain_text(impact.table(top=30)),
        mo.md(f"<sub>{region.attribution} Research model, not an operational forecast.</sub>"),
    ]))

# ── optional: the satellite networks of the 0.12 satellite path, on this one GPU ─────────────────────────────────
if TRAIN_SATELLITE:
    from naturev1.b300 import follow, start_job
    from naturev1.ibtracs import fetch_ibtracs, parse_ibtracs, storm_label_table

    if HF_TOKEN.strip():
        os.environ["HF_TOKEN"] = HF_TOKEN.strip()
    SAT, CKPT, LOGS = DATA / "satellite", DATA / "runs", DATA / "logs"
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
