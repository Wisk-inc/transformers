# ═══════════════════════════════════════════════════════════════════════════════════════════════════
#  NatureV1 0.8.5 — the whole thing in one cell. Pure Python: marimo, Colab, Jupyter or a plain script.
#
#  Paste and run. It installs what it needs, stages the data, builds the 89M-parameter model, trains it
#  in stages, scores itself against persistence and climatology, and only publishes if it earned it.
#
#  With BACKGROUND = True (the default) the run happens in a supervised process on the machine itself:
#  it keeps going if the wifi drops or the notebook closes, and it restarts itself from its last
#  checkpoint if it crashes, runs out of GPU memory, or stalls. The cell then ends with a green "training
#  in the background" note: that is it working, not an error. Run the cell again, or naturev1.follow(),
#  any time to see the run. Data already downloaded is never downloaded again -- change the years and
#  only what is new is.
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

BACKGROUND    = True      # supervised background run: survives disconnects, restarts on crash or stall
STALL_MINUTES = 30        # background: restart a run that shows no progress for this long

RUN_PRETRAIN  = True      # stage one: every channel at every lead, on ERA5 reanalysis
RUN_ROLLOUT   = True      # stage 1b: the whole atmosphere rolled forward 12 steps, CRPS ensemble
RUN_STORMS    = True      # stage two: hurricane heads on HURDAT2, backbone frozen
RUN_SCORE     = True      # WeatherBench-style scorecards against persistence and climatology
RUN_HINDCAST  = True      # forecast a held-out hurricane from its real analysis; compare with truth
RUN_PUBLISH   = False     # push to the Hub -- only if it beats persistence somewhere
RUN_FORECAST  = False     # live GOES forecast (refused while the satellite path is untrained)
RUN_WATCHER   = False     # then re-forecast every hour

STAGE_YEARS   = 5         # training years staged to local disk, ~7.5 GB each at 89 channels
VAL_YEARS     = 1         # held-out years staged too, so validation and scoring are not network-bound
EPOCHS        = 8         # passes over the staged years in stage one
BATCH         = None      # None: find the largest batch that fits once, then remember it; or a number
MAX_BATCH     = 16        # ceiling on the batch search: host RAM, not VRAM, is the limit above this
WORKERS       = 4         # data-loading processes
ROLLOUT_STEPS = 12        # longest rollout trained: 12 x 6 h = 72 h, GraphCast's curriculum
ROLLOUT_TRAIN = 4_000     # optimizer steps of rollout training (the log prints the time per step)
MEMBERS       = 2         # CRPS ensemble members; 1 = deterministic Gaussian training
STORM_FIRST_SEASON = 1979 # the satellite era; earlier best-track intensities are much less reliable
HINDCAST      = ("DORIAN", 2019)       # a held-out storm: 2019 is never trained on
STORM         = (24.6, -78.2)          # live forecast: current storm centre (lat, lon)
CITY          = (25.77, -80.19)        # live forecast: somewhere you want a local forecast
HF_REPO       = "Sigmandndnns/NatureV1-500"
HF_TOKEN      = ""        # a write token, to publish and mirror checkpoints; or set the HF_TOKEN env var
DATA_DIR      = None      # None: /content on Colab, ~/naturev1_data elsewhere; or any folder with room

# ═══ 0 ═══ install ═════════════════════════════════════════════════════════════════════════════════
# Standard library only, before numpy or torch are imported: pip may upgrade numpy while satisfying
# zarr or gcsfs, and a numpy that changes under an imported torch fails far from here.
import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")   # the cloud client logs every DataLoader fork otherwise
# Let the GPU allocator grow segments instead of fragmenting them. Training at batch 8 fills a 96 GB card
# to within ~130 MB, and without this the allocator has to flush and retry. Only takes effect if torch
# has not touched the GPU yet in this kernel -- restart the kernel for it to apply.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _halt(message, kind="danger"):
    """
    End the cell here, on purpose. Only kind="danger" is an error. marimo shows the message as a
    coloured note instead of a traceback, Jupyter and Colab show it without one, and a plain script
    exits with status 0 for "success" and 1 for anything else -- which is what the supervisor of a
    background run reads, so a run stopped part-way is resumed and a finished one is not.
    """
    marimo = sys.modules.get("marimo")
    if marimo is not None and getattr(marimo, "running_in_notebook", lambda: False)():
        try:
            note = marimo.callout(marimo.plain_text(message), kind=kind)
        except Exception:
            note = None
            print(message)
        marimo.stop(True, note)
    ipython = sys.modules.get("IPython")
    if kind != "danger" and ipython is not None and getattr(ipython, "get_ipython", lambda: None)() is not None:
        class Halted(Exception):
            def _render_traceback_(self):
                return []                         # the message printed above, and no traceback
        print(message)
        raise Halted("the cell stopped here on purpose -- the message is above")
    if kind == "success":
        print(message)
        raise SystemExit(0)
    raise SystemExit(message)


_NEEDED = {"naturev1": ("naturev1[all]>=0.8.5", (0, 8, 5)), "ihelix": ("ihelix>=0.5.1", (0, 5, 1))}


def _version(module):
    try:
        return tuple(int(p) for p in importlib.metadata.version(module).split(".")[:3])
    except Exception:
        return None


_missing = [req for mod, (req, low) in _NEEDED.items()
            if importlib.util.find_spec(mod) is None or (_version(mod) or (0,)) < low]
if _missing:
    print(f"installing: {' '.join(_missing)}")
    _loaded = {name: getattr(sys.modules.get(name), "__version__", None) for name in ("numpy", "torch")}
    _tries = ([("uv", ["uv", "pip", "install", "--python", sys.executable, "--upgrade", *_missing])]
              if shutil.which("uv") else [])
    _pip = [sys.executable, "-m", "pip", "install", "--upgrade"]
    _tries += [("pip, fresh index", [*_pip, "--no-cache-dir", "--index-url", "https://pypi.org/simple", *_missing]),
               ("pip, no cache", [*_pip, "--no-cache-dir", *_missing]),
               ("pip", [*_pip, *_missing])]
    for _label, _command in _tries:
        try:
            subprocess.check_call(_command)
            print(f"installed via {_label}")
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  {_label} failed, trying the next way")
    else:
        _halt("could not install; add these in your package manager, then re-run: " + " ".join(_missing))
    importlib.invalidate_caches()
    # A restart is needed only if an old copy is already in memory, or pip moved numpy/torch under us.
    _stale = [m for m in _NEEDED if m in sys.modules]
    for _name, _was in _loaded.items():
        try:
            if _was and importlib.metadata.version(_name) != _was:
                _stale.append(f"{_name} {_was}")
        except importlib.metadata.PackageNotFoundError:
            pass
    if _stale:
        _halt(f"installed. {', '.join(_stale)} was already loaded in this kernel: RESTART THE KERNEL, then run "
              "this cell again. (A background run that is already training is not affected.)", kind="warn")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

if BACKGROUND and not os.environ.get("NATUREV1_BACKGROUND_CHILD"):
    # Hand this same pipeline, with these settings, to a supervised process on this machine, and stop
    # here: the notebook only starts and watches it. The GPU is never touched from the notebook.
    # Imported under a private name: marimo lets a name be defined by only one cell, and this way any
    # other cell can still `import naturev1` to call naturev1.follow() or naturev1.stop().
    import naturev1 as _naturev1

    _SETTINGS = {name: globals()[name] for name in (
        "RUN_PRETRAIN", "RUN_ROLLOUT", "RUN_STORMS", "RUN_SCORE", "RUN_HINDCAST", "RUN_PUBLISH",
        "RUN_FORECAST", "RUN_WATCHER", "STAGE_YEARS", "VAL_YEARS", "EPOCHS", "BATCH", "MAX_BATCH", "WORKERS",
        "ROLLOUT_STEPS", "ROLLOUT_TRAIN", "MEMBERS", "STORM_FIRST_SEASON", "HINDCAST", "STORM", "CITY",
        "HF_REPO", "HF_TOKEN", "DATA_DIR")}
    _naturev1.launch(directory=DATA_DIR, stall_minutes=STALL_MINUTES, **_SETTINGS)
    _naturev1.follow(lines=25, directory=DATA_DIR, wait=90)   # a fresh run: wait for its first lines
    _halt("NatureV1 is training in the background, and the cell stops here on purpose -- not an error.\n"
          "To see progress: run this cell again (it will not start a second run), or in any cell\n"
          "    import naturev1; naturev1.follow()\n"
          "To end the run: naturev1.stop(). Closing the notebook or losing the wifi does not stop it.",
          kind="success")

# ═══ 1 ═══ imports, paths, hardware ════════════════════════════════════════════════════════════════
import datetime as dt
import json

import numpy as np
import torch
from ihelix import FieldGrid, Geometry, fibonacci_sphere
from naturev1 import (HEADLINE_STATE_FIELDS, CachedERA5, CheckpointManager, ERA5Window, NatureConfig, NatureV1,
                      RolloutSchedule, StateStepper, Trainer, TrainSettings, autotune_batch_size,
                      benchmark_steps, build_climatology, build_forecast, calendar_features, catalogue,
                      channel_loss_weights, check_forecast, check_grid_alignment, check_normalization,
                      check_weights, corpus_scale, device_report, era5_loader, era5_source_grid, era5_splits,
                      expand_variables, fetch_latest, format_pairing, format_plan, format_preflight,
                      lead_offsets, masked_gaussian_nll, materialise, normalize_channels,
                      open_weatherbench_levels, pair_tracks_with_reanalysis, parse_hurdat2, preflight,
                      push_to_hub, rapid_intensification, resolve_all, responds_to_input, scene_from_netcdf,
                      score_model, score_state, scoring_fields, split_by_storm, spread_skill_ratio,
                      state_forecast, storm_feature_bank, storm_scorecard, storm_state_vector,
                      train_state_rollout, train_storm_heads, training_plan, upper_air_report, watch)
from naturev1.besttrack import download
from naturev1.era5 import _target_index

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLAB_DRIVE = "/content/drive/MyDrive"
DATA_DIR = DATA_DIR or ("/content" if os.path.isdir("/content") else os.path.expanduser("~/naturev1_data"))
CKPT_DIR = (f"{COLAB_DRIVE}/naturev1_ckpt" if os.path.isdir(COLAB_DRIVE)       # Drive outlives the VM
            else os.path.join(DATA_DIR, "naturev1_ckpt"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
CACHE, VAL_CACHE, STATS = (os.path.join(DATA_DIR, name) for name in ("era5.npy", "era5_val.npy", "stats.json"))


def banner(text):
    print(f"\n{'═' * 99}\n  {text}\n{'═' * 99}")


banner("HARDWARE")
print(device_report())
print(f"data {DATA_DIR} | checkpoints {CKPT_DIR}")

# ═══ 2 ═══ data: 13 pressure levels + surface + static, resolved against this machine ══════════════
banner("DATA — ERA5 (WeatherBench 2), surface and 13 pressure levels")
ERA5, SURFACE, UPPER = open_weatherbench_levels()
LEVELS = [int(x) for x in ERA5.level.values]
SRC, LAT, LON = era5_source_grid(ERA5)
plan = resolve_all(ERA5, tuple(SURFACE) + tuple(UPPER), LEVELS, STAGE_YEARS + VAL_YEARS, SRC.num_points,
                   layers=17, batch=MAX_BATCH, precision="bf16", path=DATA_DIR,
                   staged=(CACHE, VAL_CACHE))      # data already on disk counts as room, not as used
print(plan.summary())
R = plan.resolved
CHANNELS = expand_variables(ERA5, R["variables"], R["levels"])
WIDTH = len(CHANNELS) + 7                         # room for observation-mask channels, zero-padded
print(f"\n{upper_air_report(SURFACE, UPPER, LEVELS)}")
assert not check_weights(SRC), check_weights(SRC)  # poles present, not deleted by cos(latitude)

OFFSETS, SPLITS = lead_offsets((6, 12, 18, 24, 36, 48, 72, 96, 120)), era5_splits(ERA5)
TRAIN_STEPS = max(R["years"] - VAL_YEARS, 1) * 1460
VAL_STEPS = VAL_YEARS * 1460
common = dict(history=6, lead_steps=OFFSETS, variables=tuple(R["variables"]), levels=R["levels"], channels=WIDTH)
for name, index in SPLITS.items():
    print(f"  {name:5} {str(ERA5.time.values[index[0]])[:10]} -> {str(ERA5.time.values[index[-1]])[:10]}")
print("split by TIME: weather is autocorrelated for days, so a random split leaks the answer.")

stream = ERA5Window(ERA5, indices=SPLITS["train"], stats_cache=STATS, **common)
print(f"\n{stream.normalizer.report()}")
print(f"\n{corpus_scale(ERA5, tuple(R['variables']))}\n\n{catalogue()}")

if R["streaming"]:
    print("\nnot enough disk to stage: streaming from the network (~100x slower per window)")
    train_ds, roll_ds = stream, ERA5Window(ERA5, indices=SPLITS["train"], stats_cache=STATS,
                                           state_steps=ROLLOUT_STEPS, **common)
    val_ds = ERA5Window(ERA5, indices=SPLITS["val"], augment=False, stats_cache=STATS,
                        state_steps=ROLLOUT_STEPS, **common)
else:
    # Always called, and never downloads twice: it returns at once when what is on disk already covers
    # the request, copies from disk what it can when the request grows, and resumes a download that died.
    TRAIN_SPAN, VAL_SPAN = SPLITS["train"][-TRAIN_STEPS:], SPLITS["val"][:VAL_STEPS]
    materialise(ERA5, TRAIN_SPAN, CACHE, variables=tuple(R["variables"]),
                levels=R["levels"], normalizer=stream.normalizer, workers=8)
    materialise(ERA5, VAL_SPAN, VAL_CACHE, variables=tuple(R["variables"]),
                levels=R["levels"], normalizer=stream.normalizer, workers=8)
    train_ds = CachedERA5(CACHE, history=6, lead_steps=OFFSETS, channels=WIDTH, augment=True,
                          restrict_to=TRAIN_SPAN)
    roll_ds = CachedERA5(CACHE, history=6, lead_steps=OFFSETS, channels=WIDTH, augment=True,
                         state_steps=ROLLOUT_STEPS, restrict_to=TRAIN_SPAN)
    val_ds = CachedERA5(VAL_CACHE, history=6, lead_steps=OFFSETS, channels=WIDTH, augment=False,
                        state_steps=ROLLOUT_STEPS, restrict_to=VAL_SPAN)
print(f"\ntrain {len(train_ds):,} windows | validate on {len(val_ds):,} held-out windows")

sample = train_ds[0]["analysis"]
for problem in (check_grid_alignment(SRC, sample[-1, :, train_ds.variables.index("2m_temperature")])
                + check_normalization(sample, "analysis")):
    _halt(f"DATA PROBLEM: {problem}")
print("data ok: samples land where the grid says, no spike channels")

# ═══ 3 ═══ what each channel IS — decides what a rollout does with it ══════════════════════════════
stepper = StateStepper.build(roll_ds, input_channels=WIDTH, grid=SRC, samples=64)
print(f"\nchannel roles:\n{stepper.roles.describe()}")
WEIGHTS = channel_loss_weights(train_ds.variables, WIDTH)

# ═══ 4 ═══ the model ═══════════════════════════════════════════════════════════════════════════════
banner("MODEL")
CFG = NatureConfig(
    analysis_channels=WIDTH, satellite_channels=6, environment_channels=8,
    hidden_size=512, num_layers=17, num_heads=8, num_kv_heads=4, head_dim=64, intermediate_size=2048,
    latent_points=4096, min_radius_km=120.0, max_radius_km=1600.0, history_frames=6,
    lead_times_hours=(6, 12, 18, 24, 36, 48, 72, 96, 120), track_modes=6,
    state_channels=stepper.roles.num_prognostic,        # predict and evolve the whole atmosphere
    noise_dim=32 if MEMBERS > 1 else 0)                 # per-member noise for the CRPS ensemble
model = NatureV1(CFG, fibonacci_sphere(CFG.latent_points, num_neighbours=CFG.latent_neighbours,
                                       cluster_size=CFG.latent_cluster)).to(DEVICE)
model.gradient_checkpointing_enable(True)
stepper.grid = SRC.to(DEVICE)
print(f"NatureV1 {model.num_parameters()/1e6:.2f}M parameters on {DEVICE}; state head over "
      f"{CFG.state_channels} channels; {MEMBERS}-member ensemble")
print(format_preflight(preflight(model=model, dataset=train_ds, grid=SRC, strict=True)))


def restore(*stages):
    """Load the newest checkpoint of the first stage that has one -- locally, else from the Hub."""
    for stage in stages:
        manager = CheckpointManager(f"{CKPT_DIR}/{stage}", repo_id=HF_REPO)
        manager.fetch_from_hub()
        if manager.load(model, map_location=DEVICE) is not None:
            return stage
    return None


# ═══ 5 ═══ the largest batch that fits, and what the run costs ═════════════════════════════════════
banner("BENCHMARK — largest batch that fits, and what the run costs")


def make_step(batch_size):
    items = [train_ds[i] for i in range(batch_size)]
    batch = {k: torch.stack([x[k] for x in items]).to(DEVICE) for k in ("analysis", "calendar", "field_target", "field_mask")}

    def step():
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            out = model(analysis=batch["analysis"], analysis_grid=SRC, calendar=batch["calendar"], output_grid=SRC)
            loss = masked_gaussian_nll(out["field_mean"], out["field_log_var"],
                                       torch.where(batch["field_mask"] > 0, batch["field_target"], torch.nan))
        loss.backward()
        model.zero_grad(set_to_none=True)
    return step


# The search is done once and remembered: repeating it on every restart costs minutes, pushes the card
# to the edge of its memory each time, and could pick a different batch -- a different run -- each time.
BATCH_FILE = os.path.join(CKPT_DIR, "batch.json")
if BATCH is None and os.path.exists(BATCH_FILE):
    with open(BATCH_FILE) as handle:
        REMEMBERED = json.load(handle)
    BATCH, SPEED = int(REMEMBERED["batch"]), float(REMEMBERED.get("samples_per_second", 0.0))
    print(f"batch {BATCH}, found on an earlier run (delete {BATCH_FILE} to search again)")
else:
    if BATCH is None:
        BATCH = min(autotune_batch_size(make_step, start=1, target_fraction=0.85), MAX_BATCH)
    MARK = benchmark_steps(make_step(BATCH), BATCH, warmup=1, iterations=10, gradient_checkpointing=True)
    SPEED = MARK.samples_per_second
    with open(BATCH_FILE, "w") as handle:
        json.dump({"batch": BATCH, "samples_per_second": SPEED}, handle)
    print(f"batch {BATCH} | {MARK}")
if SPEED:
    print(format_plan(training_plan(SPEED, corpus_samples=len(train_ds), epochs=EPOCHS,
                                    watts=600.0, electricity_per_kwh=0.15, cloud_per_hour=2.50)))
if DEVICE == "cuda":
    torch.cuda.empty_cache()                      # hand back what the search held before training starts
loader = era5_loader(train_ds, batch_size=BATCH, num_workers=WORKERS, analysis_grid=SRC, output_grid=SRC,
                     prefetch_factor=2)
val_loader = era5_loader(val_ds, batch_size=BATCH, num_workers=WORKERS, shuffle=False, analysis_grid=SRC,
                         output_grid=SRC, prefetch_factor=2)

# ═══ 6 ═══ stage one — every channel, every lead ═══════════════════════════════════════════════════
if RUN_PRETRAIN:
    banner("STAGE ONE — pretraining on reanalysis")
    trainer = Trainer(model, TrainSettings(
        stage="pretrain", learning_rate=3e-4, warmup_steps=1000,
        precision=R["precision"], max_steps=int(len(train_ds) * EPOCHS / BATCH),
        checkpoint_dir=f"{CKPT_DIR}/stage1", checkpoint_seconds=60, hub_repo=HF_REPO,
        ema_decay=0.0, log_every=25, val_every=500, val_batches=32), device=DEVICE)
    trainer.resume()
    trainer.fit(loader, epochs=EPOCHS, val_loader=val_loader)
    if trainer.interrupted:
        # Stopped part-way: checkpointed, and the next stage must not start from a half-trained model.
        _halt("stage one stopped and checkpointed -- run again to resume it where it left off", kind="warn")
    print("held-out falling = learning; held-out rising while training falls = memorising.")

# ═══ 7 ═══ stage 1b — the whole atmosphere, rolled forward on its own forecasts ════════════════════
if RUN_ROLLOUT:
    banner(f"ROLLOUT — {ROLLOUT_STEPS}-step full-state training, {MEMBERS}-member CRPS ensemble")
    if not RUN_PRETRAIN and restore("stage1") is None:
        print("!! no stage-one checkpoint: rolling out an untrained backbone. Set RUN_PRETRAIN = True.")
    steps = ROLLOUT_TRAIN
    roll_loader = era5_loader(roll_ds, batch_size=max(BATCH // 2, 1), num_workers=WORKERS,
                              analysis_grid=SRC, output_grid=SRC, prefetch_factor=2)
    train_state_rollout(
        model, roll_loader, stepper, max_steps=steps,
        schedule=RolloutSchedule(start_step=max(steps // 10, 1), ramp_steps=max(steps // 2, 1),
                                 max_steps=ROLLOUT_STEPS),
        channel_weights=WEIGHTS, members=MEMBERS, precision=R["precision"],
        checkpoint_dir=f"{CKPT_DIR}/rollout", hub_repo=HF_REPO, log_every=25)

# ═══ 8 ═══ stage two — hurricane heads, on the frozen backbone ═════════════════════════════════════
banner("STAGE TWO — HURDAT2 best tracks")
tracks = parse_hurdat2(download("https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt",
                                os.path.join(DATA_DIR, "hurdat2.txt")))
print(f"HURDAT2: {len(tracks):,} storms, {sum(len(t) for t in tracks):,} points, "
      f"{sum(int(t.landfall.sum()) for t in tracks):,} landfalls")
walk = rapid_intensification(tracks, threshold_kt=30.0)
print(f"rapid intensification: {walk['positives']:,} of {walk['eligible']:,} eligible = "
      f"{100 * walk['base_rate']:.2f}%  (an always-'no' classifier scores {100 - 100 * walk['base_rate']:.0f}%)")
GROUPS = split_by_storm([t for t in tracks if t.year >= STORM_FIRST_SEASON])
STORE_TIMES = ERA5.time.values.astype("datetime64[s]").astype(np.int64)


def paired(which):
    starts, targets, report = pair_tracks_with_reanalysis(GROUPS[which], STORE_TIMES, OFFSETS, history=6)
    return starts, targets, report


if RUN_STORMS:
    if not (RUN_PRETRAIN or RUN_ROLLOUT) and restore("rollout", "stage1") is None:
        _halt("no pretrained backbone found. Freezing an untrained one would fine-tune the storm "
              "heads on noise -- set RUN_PRETRAIN = True.")
    (tr_starts, tr_targets, tr_report), (va_starts, va_targets, va_report) = paired("train"), paired("validation")
    print(f"\n{format_pairing(tr_report)}")
    print(f"validation: {va_report['paired']:,} samples from {va_report['storms']} unseen storms "
          f"(seasons 2017/2019/2021)")
    print("\nreading each storm's reanalysis once and running the frozen backbone over it "
          "(cached; resumes if interrupted):")
    bank_train = storm_feature_bank(model, ERA5, stream, tr_starts, tr_targets, SRC,
                                    cache=f"{CKPT_DIR}/storm_bank_train.pt", precision=R["precision"])
    bank_val = storm_feature_bank(model, ERA5, stream, va_starts, va_targets, SRC,
                                  cache=f"{CKPT_DIR}/storm_bank_val.pt", precision=R["precision"])
    history = train_storm_heads(model, bank_train, bank_val, epochs=300,
                                patience=25, checkpoint_dir=f"{CKPT_DIR}/stage2", hub_repo=HF_REPO)
    best = min(history, key=lambda h: h["val"])
    banner("SCOREBOARD — is it generalising, or memorising?")
    print(f"best epoch {best['epoch']}: held-out {best['val']:.4f}  training {best['train']:.4f}  "
          f"gap {best['val'] - best['train']:+.4f}   (a large positive gap = memorised the training storms)")
    print(f"\nheld-out storms, seasons never trained on:\n{storm_scorecard(model, bank_val, CFG.lead_times_hours)}")

# ═══ 9 ═══ did it learn? scorecards against persistence and climatology ════════════════════════════
if not (RUN_PRETRAIN or RUN_ROLLOUT or RUN_STORMS):
    print(f"loaded: {restore('stage2', 'rollout', 'stage1') or 'nothing -- untrained'}")
print(f"\ntraining record (steps per output; 0 = never trained, never shown as a forecast):\n"
      f"  {model.trained_heads()}")

_probe_stream = iter(val_loader)


def _probe():
    batch = next(_probe_stream)
    return {"analysis": batch["analysis"].to(DEVICE), "calendar": batch["calendar"].to(DEVICE)}


responds, spread = responds_to_input(model, _probe, SRC)
print(f"\nresponds to the weather: {responds}  (spread {spread:.2e})")

card = state_card = None
if RUN_SCORE:
    banner("SCORECARDS — area-weighted, physical units, identical batches for every baseline")
    climo = build_climatology(train_ds, np.arange(len(train_ds)), WIDTH, SRC.num_points,
                              samples=200)
    batches = 32
    card = score_model(model, val_loader, SRC, LAT, len(LON), CFG.lead_times_hours,
                       fields=scoring_fields(train_ds.variables, _target_index(train_ds.variables)),
                       normalizer=train_ds.normalizer, climatology=climo, max_batches=batches, device=DEVICE)
    print("surface heads, direct multi-lead:\n" + card.table() + "\n" + card.verdict())
    state_card = score_state(model, val_loader, stepper, HEADLINE_STATE_FIELDS,
                             steps=ROLLOUT_STEPS, climatology=climo,
                             max_batches=batches, precision=R["precision"])
    print("\nfull-state rollout (Z500 in m2/s2, as WeatherBench 2 reports it):\n"
          + state_card.table() + "\n" + state_card.verdict())
    if MEMBERS > 1 and "geopotential@500" in train_ds.variables:
        batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in next(iter(val_loader)).items()}
        lead = min(4, batch["state_target"].shape[1])
        ensemble = state_forecast(model, batch, stepper, steps=lead, members=max(MEMBERS, 4),
                                  precision=R["precision"])
        z500 = train_ds.variables.index("geopotential@500")
        ratio = spread_skill_ratio(ensemble[:, -1, ..., z500], batch["state_target"][:, lead - 1, :, z500])
        print(f"\nZ500 +{6 * lead}h ensemble spread/skill {ratio:.2f}   (1.0 calibrated, <1 overconfident)")
        if ratio < 0.05:
            print("  no spread yet: the noise path starts at exactly zero and CRPS training grows it.")

# ═══ 10 ═══ hindcast — a held-out hurricane, from its real analysis ════════════════════════════════
if RUN_HINDCAST:
    banner(f"HINDCAST — {HINDCAST[0].title()} {HINDCAST[1]}, a season the model never trained on")
    storm = next((t for t in tracks if t.name.strip().upper() == HINDCAST[0] and t.year == HINDCAST[1]), None)
    starts, targets, _ = (pair_tracks_with_reanalysis([storm], STORE_TIMES, OFFSETS, history=6)
                          if storm is not None else ([], [], None))
    if not len(starts):
        print("that storm is not in the reanalysis window (1959-2021); pick another in HINDCAST")
    else:
        peak = int(np.nanargmax(storm.max_wind_kt))
        choice = min(range(len(targets)), key=lambda i: abs(targets[i].point - max(peak - 8, 0)))   # ~48 h before peak
        target = targets[choice].build()
        window = ERA5Window(ERA5, indices=starts[choice:choice + 1], augment=False, stats_cache=STATS, **common)[0]
        issued = dt.datetime.fromtimestamp(float(window["valid_time"]), tz=dt.timezone.utc)
        with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            out = model.eval()(analysis=window["analysis"][None].to(DEVICE), analysis_grid=SRC,
                               calendar=window["calendar"][None].to(DEVICE), output_grid=SRC,
                               storm_center=target["storm_center"][None].to(DEVICE),
                               storm_state=target["storm_state"][None].to(DEVICE))
        out = {k: v.float().cpu() for k, v in out.items()}
        centre = target["storm_center"].tolist()
        fc = build_forecast(out, CFG.lead_times_hours, issued, storm_center=centre,
                            normalizer=train_ds.normalizer, trained=model.trained_heads(), inputs=("analysis",))
        print(f"issued {issued:%Y-%m-%d %H:%M}Z at {centre[0]:.1f}N {-centre[1]:.1f}W, "
              f"{float(target['current_wind_kt']):.0f} kt now")
        if isinstance(fc["track_scenarios"], list):
            top = fc["track_scenarios"][0]
            print(f"most likely scenario ({top['probability']:.0%}) vs what happened:")
            for lead, point in enumerate(top["track"]):
                if target["track_valid"][lead] > 0:
                    lat = centre[0] + float(target["track_target"][lead, 0])
                    lon = centre[1] + float(target["track_target"][lead, 1])
                    error = 111.2 * np.hypot(point["latitude"] - lat, (point["longitude"] - lon) * np.cos(np.radians(lat)))
                    wind = ""
                    if isinstance(fc["intensity"], list):
                        wind = f"{fc['intensity'][lead]['max_wind_ms'] * 1.94384:5.0f} kt forecast"
                        if target["intensity_mask"][lead, 0] > 0:
                            wind += f" / {float(target['intensity_target'][lead, 0]) * 1.94384:3.0f} kt actual"
                    print(f"  +{point['lead_hours']:3d}h  {point['latitude']:5.1f},{point['longitude']:6.1f}  "
                          f"actual {lat:5.1f},{lon:6.1f}  error {error:5.0f} km   {wind}")
        else:
            print("track head untrained -- run RUN_STORMS first")
        problems = check_forecast(fc)
        print("forecast checks: " + ("pass" if not problems else "; ".join(problems)))

# ═══ 11 ═══ publish only if it earned it ═══════════════════════════════════════════════════════════
if RUN_PUBLISH:
    cards = [c for c in (state_card, card) if c is not None]
    beats = [s for c in cards for s in c.scores if s.beats_persistence]
    if not cards:
        print("\nnot publishing: nothing was scored (set RUN_SCORE = True).")
    elif not responds:
        print("\nnot publishing: the model gives the same answer whatever weather it is shown.")
    elif not beats:
        print("\nnot publishing: it loses to persistence at every field and lead.")
    else:
        url = push_to_hub(model, HF_REPO, normalizer=train_ds.normalizer, scorecard=cards[0], name="NatureV1",
                          author="Nathan", token=HF_TOKEN or None,
                          training_data=f"ERA5 (WeatherBench 2), {len(CHANNELS)} channels, {R['years']} years; "
                                        f"full-state {ROLLOUT_STEPS}-step rollout, {MEMBERS}-member CRPS; "
                                        f"HURDAT2 {STORM_FIRST_SEASON}+ for the storm heads.")
        print(f"\npublished: {url}  (beats persistence on {len(beats)} field-lead pairs)")

# ═══ 12 ═══ live forecast from GOES — only once that input path is trained ═════════════════════════
def load_scene(paths, center, half_width_deg=9.0, stride=4):
    """Geolocated samples plus true pixel footprints, cropped to a real box on the planet."""
    scene = scene_from_netcdf({c: str(p) for c, p in paths.items()}, stride=stride,
                              bounds=(center[0] - half_width_deg, center[0] + half_width_deg,
                                      center[1] - half_width_deg, center[1] + half_width_deg))
    coords = torch.tensor(np.stack([np.radians(scene.latitude), np.radians(scene.longitude) % (2 * np.pi)], -1),
                          dtype=torch.float64)
    grid = FieldGrid.from_points(coords, Geometry.globe(), num_neighbours=16, cluster_size=64,
                                 weights=torch.tensor(scene.area_km2, dtype=torch.float64))
    values = torch.tensor(normalize_channels(scene.values, scene.channels), dtype=torch.float32)
    if CFG.satellite_channels > values.shape[-1]:
        values = torch.cat([values, torch.zeros(values.shape[0], CFG.satellite_channels - values.shape[-1])], -1)
    return grid, values, scene


@torch.no_grad()
def forecast_now(storm=STORM, city=CITY):
    paths = fetch_latest(os.path.join(DATA_DIR, "goes"), ("C13", "C09"), satellite="east", product="conus")
    grid, values, scene = load_scene(paths, storm)
    print(scene)
    frames = CFG.history_frames
    out_grid = fibonacci_sphere(2000, num_neighbours=16, cluster_size=50)
    out = model.eval()(satellite=values[None, None].expand(1, frames, -1, -1).contiguous().to(DEVICE),
                       satellite_grid=grid, output_grid=out_grid, neighbours=12,
                       calendar=calendar_features(torch.full((1, frames), scene.timestamp.timestamp())).to(DEVICE),
                       storm_center=torch.tensor([storm], dtype=torch.float32, device=DEVICE))
    return build_forecast({k: v.float().cpu() for k, v in out.items()}, CFG.lead_times_hours, scene.timestamp,
                          storm_center=storm, output_grid=out_grid, point_of_interest=city,
                          normalizer=train_ds.normalizer, trained=model.trained_heads(), inputs=("satellite",))


def show(fc):
    print(f"\nissued {fc['issued']}   trustworthy: {fc['trustworthy']}")
    for key in ("warning", "note"):
        if key in fc:
            print(f"  !! {fc[key]}")
    for section in ("enso", "landfall", "track_scenarios", "eyewall", "rapid_intensification"):
        value = fc.get(section)
        if isinstance(value, dict) and value.get("available") is False:
            print(f"  {section}: not available -- {value['reason']}")
    if isinstance(fc.get("track_scenarios"), list):
        for s in fc["track_scenarios"][:4]:
            p = s["track"][5]
            print(f"  {s['probability']:>5.0%} -> +{p['lead_hours']}h {p['latitude']:6.2f},{p['longitude']:7.2f}"
                  f"  95% cone {p['cone_radius_km_95']:.0f} km")
    print("  checks: " + ("pass" if not check_forecast(fc) else "; ".join(check_forecast(fc))))


if RUN_FORECAST or RUN_WATCHER:
    if model.trained_heads()["satellite_encoder"] == 0:
        print("\nlive GOES forecast refused: no training stage has ever fed this model satellite imagery, so "
              "its satellite encoder is at initialisation and every number it produced would be noise shaped "
              "like a forecast. Use the HINDCAST section, which runs from the analysis the model was trained on.")
    elif RUN_FORECAST:
        banner("LIVE FORECAST — newest GOES scene")
        show(forecast_now())


def on_new_scene(paths):
    result = forecast_now()
    os.makedirs(os.path.join(DATA_DIR, "forecasts"), exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M")
    with open(os.path.join(DATA_DIR, "forecasts", f"forecast_{stamp}.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    show(result)


if RUN_WATCHER and model.trained_heads()["satellite_encoder"] > 0:
    banner("WATCHING — a new forecast every hour")
    watch(on_new_scene, interval_seconds=3600, directory=os.path.join(DATA_DIR, "goes"), channels=("C13", "C09"))
