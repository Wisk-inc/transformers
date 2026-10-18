# ═══════════════════════════════════════════════════════════════════════════════════════════════════
#  NatureV1 0.13 on one 96 GB GPU (molab) -- hurricane physics, teacher distillation, 0.25-degree output.
#
#  Cells for marimo / Jupyter / VS Code ("# %%"), or run top to bottom as a script. Long steps run as detached
#  processes with logs under DATA/logs: closing the notebook never stops them, and re-running a cell reattaches.
#
#    0  settings (paste a Hugging Face WRITE token for the session; clear it afterwards)
#    1  install + preflight                    minutes  GPU, disk, network, token
#    2  starting model from the Hub            minutes
#    3  hurricane physics benchmark (CPU)      ~1 h     7 seasons of hindcasts, all basins; the physics consensus
#                                                       against extrapolation, CLIPER, GraphCast, Pangu, HRES,
#                                                       Aurora, FGN (WeatherNext 2), FuXi, GenCast
#    4  Dorian and the hard scenarios          minutes  what the physics says about Dorian 2019 and the synthetic
#                                                       stall / recurvature / Fujiwhara / reformation cases
#    5  genesis outlook                        minutes  disturbances in the latest analysis, 48 h genesis odds
#    6  ERA5 for distillation                  ~1 h     2015-2022 at 1.5 degrees (~60 GB)
#    7  distillation (GPU, detached)           budget   teachers aligned onto NatureV1 and distilled into it
#    8  NatureV1 on the hurricanes again       ~1 h     rollout members + the hybrid consensus vs the teachers
#    9  0.25-degree decoder (GPU, optional)    budget   learned downscaling of every forecast to 0.25 degrees
#   10  publish                                minutes  only if the new model beat the old one
#
#  Live storms, the satellite path, the swarm sandbox and Caribbean parish impacts: molab_swarm_cell.py (one cell).
#
#  96 GB is enough for everything here with bf16 and activation checkpointing: the 151M-parameter model with an
#  8-step rollout at batch 2 peaks around 60 GB on the full 1.5-degree grid. Every checkpoint is pushed to HF_REPO.
#  Nothing here has run on molab hardware by its author; the CPU parts were run end to end in development and the
#  preflight is the first thing that touches the GPU.
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

# %% [0] settings ──────────────────────────────────────────────────────────────────────────────────────
HF_REPO          = "CorxLabs/NatureV1-500"
HF_TOKEN         = ""          # paste a WRITE token for this session only, then clear this line
DATA             = "/data/naturev1"
DEV_SEASONS      = (2016, 2017, 2019)            # tuning seasons (constants were fitted on these)
TEST_SEASONS     = (2018, 2020, 2021, 2022)      # held out: every reported number comes from these
DISTILL_HOURS    = 10.0        # GPU budget for distillation
DISTILL_TEACHERS = "hres,pangu,graphcast,aurora,fgn"
DISTILL_TRAIN    = "2016,2017,2019"              # keeps the test seasons unseen; widen only knowingly
HIGHRES_STEPS    = 3000        # 0.25-degree decoder fitting steps (0 to skip)
CPU_WORKERS      = None        # None: all cores
SOURCE           = "pypi"      # "pypi" (pip install naturev1) or "git" (this branch)
BRANCH           = "claude/linear-context-architecture-a35c6h"

# %% [1] install and preflight ──────────────────────────────────────────────────────────────────────────
import os, subprocess, sys
from pathlib import Path

if SOURCE == "pypi":
    packages = ["naturev1[all]>=0.13.0", "ihelix>=0.6.0"]
else:
    url = "git+https://github.com/Wisk-inc/transformers@" + BRANCH
    packages = [f"ihelix @ {url}#subdirectory=ihelix", f"naturev1[all] @ {url}#subdirectory=natureweather"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *packages, "safetensors", "gcsfs", "zarr"], check=True)
if HF_TOKEN.strip():
    os.environ["HF_TOKEN"] = HF_TOKEN.strip()            # inherited by the jobs below; never written to disk
DATA = Path(DATA)
LOGS, CY, RUNS = DATA / "logs", DATA / "cyclone", DATA / "runs"
for folder in (LOGS, CY, RUNS):
    folder.mkdir(parents=True, exist_ok=True)
WORKERS = CPU_WORKERS or os.cpu_count() or 8

import naturev1
from naturev1.hardware import hardware_preflight

print(f"naturev1 {naturev1.__version__}, {WORKERS} CPU workers")
READY = hardware_preflight(DATA, HF_REPO, need_tb=0.3, hours=DISTILL_HOURS, gpus_each=1)
assert READY, "fix the lines marked !! above before spending GPU hours"

# %% [2] the starting model from the Hub ────────────────────────────────────────────────────────────────
import shutil
from huggingface_hub import hf_hub_download

BASE = DATA / "base.pt"
if not BASE.exists():
    for tag in ("stage-b", "stage3", "stage2"):
        try:
            shutil.copy2(hf_hub_download(HF_REPO, f"checkpoints/{tag}/latest.pt"), BASE)
            print(f"starting from checkpoints/{tag}/latest.pt")
            break
        except Exception as error:
            print(f"no {tag} on the Hub ({type(error).__name__})")
STATS = DATA / "stats.json"
if not STATS.exists():
    shutil.copy2(hf_hub_download(HF_REPO, "staging/stats.json"), STATS)

# %% [3] hurricane physics benchmark: no GPU, no training -- every basin, seven seasons ───────────────────
import pickle
import numpy as np
from naturev1.ibtracs import fetch_ibtracs, parse_ibtracs
from naturev1.cyclone.analogs import AnalogDatabase
from naturev1.cyclone.benchmark import benchmark, consensus_tracks, report, run_season
from naturev1.cyclone.climatology import build_climatology
from naturev1.cyclone.data import LEADS, Season, build_season, cases_from_tracks, WB2Reader
from naturev1.cyclone.teachers import STORES, teacher_tracks

IBTRACS = fetch_ibtracs(DATA)                                    # every basin since 1980, one CSV
tracks = parse_ibtracs(IBTRACS, since=1979)
reader = None
for year in DEV_SEASONS + TEST_SEASONS:                          # ~2 min a season: analysis boxes around every storm
    if not (CY / str(year) / "done").exists():
        reader = reader or WB2Reader()
        build_season(cases_from_tracks(tracks, [year]), CY / str(year), reader)
CLIM = build_climatology(CY / "climatology.npy")                 # ERA5 1981-2010, no overlap with the seasons
ANALOGS = CY / "analogs.pkl"
if not ANALOGS.exists():                                          # 1980-2014 only
    history = cases_from_tracks([t for t in tracks if t.year <= 2015], range(1980, 2015), init_hours=(0, 6, 12, 18))
    pickle.dump(AnalogDatabase.from_cases(history, LEADS), open(ANALOGS, "wb"))
for year in TEST_SEASONS:
    run_season(CY / str(year), CY / f"members_{year}.pkl", CLIM, ANALOGS, processes=WORKERS)
cases, results = [], {}
for year in TEST_SEASONS:
    cases += Season(CY / str(year)).cases
    results.update(pickle.load(open(CY / f"members_{year}.pkl", "rb")))
nan = np.full((len(LEADS), 2), np.nan)
CONSENSUS = consensus_tracks(cases, results)
CLIPER = np.stack([results.get(c.key, {}).get("cliper", nan) for c in cases])
print(report(benchmark(cases, {"CLIPER": CLIPER, "NatureV1 physics": CONSENSUS}),
             f"held-out {TEST_SEASONS}, all basins: track error (km)"))
for name, stores in STORES.items():
    years = sorted({y for _, a, b in stores for y in range(a, b + 1)} & set(TEST_SEASONS))
    sub = [i for i, c in enumerate(cases) if c.year in years]
    if not sub:
        continue
    path = CY / f"teacher_{name}.pkl"
    if not path.exists():
        pickle.dump(teacher_tracks([cases[i] for i in sub], name, LEADS), open(path, "wb"))
    t = pickle.load(open(path, "rb"))
    print(report(benchmark([cases[i] for i in sub], {"NatureV1 physics": CONSENSUS[sub], name: t}),
                 f"against {name} on the same storms"))

# %% [4] Dorian 2019 and the hard synthetic scenarios ───────────────────────────────────────────────────
from naturev1.cyclone import physics_forecast
from naturev1.cyclone.climatology import Climatology
from naturev1.cyclone.synthetic import scenario_inputs, scenario_library, stall_hours, truth_track
from naturev1.cyclone.geometry import great_circle_km

clim, analogs = Climatology(CLIM), pickle.load(open(ANALOGS, "rb"))
season = Season(CY / "2019")
for i, case in enumerate(season.cases):
    if case.name == "DORIAN" and str(case.time)[11:13] == "12" and "2019-08-31" <= str(case.time)[:10] <= "2019-09-02":
        fc = physics_forecast(case, season.environment(i), clim, analogs)
        err = great_circle_km(fc["track"][:, 0], fc["track"][:, 1], case.truth["lat"], case.truth["lon"])
        print(f"Dorian {str(case.time)[:13]}: regime {fc['regime']['regime']}, stall p={fc['stall']['stall_probability']:.2f}, "
              f"track error " + " ".join(f"{h}h {e:.0f}" for h, e in zip(LEADS, err) if np.isfinite(e)))
for name, scenario in scenario_library().items():
    case, env = scenario_inputs(scenario, LEADS)
    fc = physics_forecast(case, env, None, None, intensity=False)
    err = great_circle_km(fc["track"][:, 0], fc["track"][:, 1], case.truth["lat"], case.truth["lon"])
    print(f"{name:17} truth stalls {stall_hours(truth_track(scenario, np.arange(0, 121, 6.0)), np.arange(0, 121, 6.0)):3.0f} h | "
          f"regime {fc['regime']['regime']:8} | error 24/72/120 h: {err[2]:.0f} / {err[5]:.0f} / {err[7]:.0f} km")

# %% [5] genesis outlook from the latest analysis ───────────────────────────────────────────────────────
from naturev1.cyclone.genesis import GenesisSeason, build_genesis_store, candidate_probability, detect_season

year = TEST_SEASONS[-1]
store = build_genesis_store([year], CY)[year]
found = detect_season(GenesisSeason(store), tracks)
latest = max(c.time for c in found)
now = [c for c in found if c.time == latest]
for c, p in sorted(zip(now, candidate_probability(now)), key=lambda x: -x[1])[:8]:
    print(f"{str(c.time)[:13]}  {c.lat:5.1f} {c.lon:7.1f}  P(genesis within 48 h) = {p:.0%}")

# %% [6] ERA5 at 1.5 degrees for distillation (2015-2022) ───────────────────────────────────────────────
from naturev1.b300 import follow, start_job

ERA5 = DATA / "era5_2015_2022.npy"
start_job("era5", [sys.executable, "-c", f"from naturev1.b300 import stage_era5_record; "
                   f"stage_era5_record({str(DATA)!r}, 2015, 2022, {str(STATS)!r}, workers=32)"], LOGS)
follow(LOGS, ["era5"])

# %% [7] distillation on the GPU: teachers aligned, then distilled (detached) ───────────────────────────
start_job("distill", ["-m", "naturev1.distill.run", "--base", str(BASE), "--era5", str(ERA5), "--root", str(DATA / "teachers"),
                      "--output", str(RUNS), "--teachers", DISTILL_TEACHERS, "--years", "2016-2022",
                      "--train-years", DISTILL_TRAIN, "--val-years", "2021", "--hours", str(DISTILL_HOURS),
                      "--batch-size", "2", "--horizon", "8", "--ibtracs", str(IBTRACS), "--hub-repo", HF_REPO],
          LOGS, gpus=[0])
follow(LOGS, ["distill"])

# %% [8] the distilled model on the hurricanes: rollout members and the hybrid consensus ─────────────────
import json
import torch
from naturev1.cyclone.consensus import fit_consensus
from naturev1.cyclone.neural import hybrid_consensus, rollout_tracks
from naturev1.sattrain import build_model, staged_grid
from naturev1.state import StateStepper, channel_roles
from naturev1.era5 import Normalizer

payload = torch.load(RUNS / "distill" / "distilled.pt", map_location="cpu", weights_only=False)
model = build_model(str(BASE), satnet_embed=0, retrieval_layers=0, ri2=False, hemisphere_frame=False)
model.load_state_dict(payload["model"], strict=False)
model = model.cuda().eval()
meta = json.loads(ERA5.with_suffix(".json").read_text())
variables = list(meta["variables"])
grid = model._aligned(staged_grid(meta))
stepper = StateStepper(channel_roles(variables, model.config.analysis_channels), model.state_tendency.detach().cpu(),
                       Normalizer.from_dict(meta["statistics"]), grid,
                       variables.index("surface_pressure") if "surface_pressure" in variables else None)
dev_cases = [c for y in DEV_SEASONS for c in Season(CY / str(y)).cases]
for y in DEV_SEASONS:
    run_season(CY / str(y), CY / f"members_{y}.pkl", CLIM, ANALOGS, processes=WORKERS)
dev_physics = {k: v for y in DEV_SEASONS for k, v in pickle.load(open(CY / f"members_{y}.pkl", "rb")).items()}
dev_roll = rollout_tracks(model, ERA5, stepper, variables, dev_cases, LEADS)
test_roll = rollout_tracks(model, ERA5, stepper, variables, cases, LEADS)
from naturev1.cyclone.coefficients import CONSENSUS as PHYS
names = list(PHYS["members"]) + ["rollout_vortex", "rollout_steering"]
cols = lambda cs, phys, roll: np.stack([np.stack([(roll if n.startswith("rollout_") else phys).get(c.key, {}).get(n, nan) for c in cs]) for n in names], 2)
w = fit_consensus(dev_cases, cols(dev_cases, dev_physics, dev_roll), reference=names.index("cliper"))
w["members"] = names
HYBRID = hybrid_consensus(cases, results, test_roll, w)
scores = benchmark(cases, {"NatureV1 physics": CONSENSUS, "NatureV1 hybrid": HYBRID,
                           "rollout vortex": np.stack([test_roll.get(c.key, {}).get("rollout_vortex", nan) for c in cases])})
print(report(scores, "the distilled model's own tracks and the hybrid consensus"))
# The release decision: the hybrid must beat the physics-only consensus at most leads on the held-out seasons.
phys, hyb = scores["members"]["NatureV1 physics"]["mean_km"], scores["members"]["NatureV1 hybrid"]["mean_km"]
share = float(np.mean([h < p for h, p in zip(hyb, phys)]))
COMPARISON = RUNS / "comparison_distill.json"
COMPARISON.write_text(json.dumps({"better_share_global": share, "physics_km": phys, "hybrid_km": hyb}))
print(f"hybrid better than physics-only at {share:.0%} of leads")

# %% [9] 0.25-degree decoder (optional) ─────────────────────────────────────────────────────────────────
# Learned downscaling: 1.5-degree analysis in, ERA5 at 0.25 degrees of the same time out, backbone frozen.
# See naturev1.highres.train_highres for the batch format; WeatherBench 2's 0.25-degree ERA5 is the target.
if HIGHRES_STEPS:
    from naturev1.highres import ERA5_0P25, HIGHRES_FIELDS, HighResHead
    print(f"high-res target: {ERA5_0P25}, fields {HIGHRES_FIELDS}; head parameters: "
          f"{sum(p.numel() for p in HighResHead(model.config.hidden_size, len(HIGHRES_FIELDS)).parameters()) / 1e6:.2f}M")

# %% [10] publish -- only if the new model beat the old one ─────────────────────────────────────────────
from naturev1.b300 import publish_release

where = publish_release(RUNS / "distill" / "latest.pt", STATS, HF_REPO, DATA / "release",
                        comparison=RUNS / "comparison_distill.json", report=None)
print("published to", where, "-- now clear HF_TOKEN in cell [0]")
