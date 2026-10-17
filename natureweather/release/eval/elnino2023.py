"""
NatureV1 end to end on the 2023 Atlantic hurricanes -- a strong El Nino season no part of the model saw.

Inputs: ERA5 for 2023 rebuilt at WeatherBench 2's 1.5-degree grid from ARCO-ERA5 (validated against the
WeatherBench 2 store to rounding precision), normalised with the run's own statistics, through the full
89M-parameter backbone and the hurricane heads of the published checkpoint. One bank per storm, saved
as it goes, so a restart continues.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

S = Path(sys.argv[1])
STORMS = sys.argv[2].split(",")
HOURS = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("00",)
sys.path.insert(0, str(S))
from arco15 import ArcoTo15  # noqa: E402
from ihelix import fibonacci_sphere  # noqa: E402
from naturev1.besttrack import parse_hurdat2  # noqa: E402
from naturev1.era5 import ERA5Window, era5_source_grid, lead_offsets  # noqa: E402
from naturev1.model import NatureConfig, NatureV1  # noqa: E402
from naturev1.storms import pair_tracks_with_reanalysis, storm_feature_bank  # noqa: E402
from naturev1.upper import open_weatherbench_levels  # noqa: E402

torch.set_num_threads(4)
ck = torch.load(S / "hub/checkpoints/stage2/latest.pt", map_location="cpu", weights_only=False)
cfg = NatureConfig.from_dict(ck["config"])
model = NatureV1(cfg, fibonacci_sphere(cfg.latent_points, num_neighbours=cfg.latent_neighbours,
                                       cluster_size=cfg.latent_cluster))
model.load_state_dict(ck["model"]); model.eval()
names = json.load(open(S / "hub/staging/stats.json"))["variables"]
levels = sorted({int(v.split("@")[1]) for v in names if "@" in v})
base_vars = list(dict.fromkeys(v.split("@")[0] for v in names))
wb2, _, _ = open_weatherbench_levels()
grid, _, _ = era5_source_grid(wb2)
conv = ArcoTo15(wb2, S / "arco_cache")
offsets = lead_offsets(cfg.lead_times_hours)
tracks = {t.name: t for t in parse_hurdat2(S / "hurdat/hurdat2.txt") if t.year == 2023}
out_dir = S / "elnino2023"; out_dir.mkdir(exist_ok=True)

for name in STORMS:
    target = out_dir / f"{name}.pt"
    if target.exists():
        print(f"{name}: done already", flush=True); continue
    track = tracks[name]
    t0 = time.time()
    first = np.datetime64(str(track.time[0])[:13], "h") - np.timedelta64(36, "h")
    last = np.datetime64(str(track.time[-1])[:13], "h") + np.timedelta64(126, "h")
    times = np.arange(first, last, np.timedelta64(6, "h"))
    starts, targets, _ = pair_tracks_with_reanalysis([track], times.astype("datetime64[s]").astype(np.int64),
                                                     offsets, history=6)
    keep = [i for i, t in enumerate(targets) if str(t.track.time[t.point])[11:13] in HOURS]
    starts, targets = starts[keep], [targets[i] for i in keep]
    needed = sorted({int(s) + k for s in starts for k in range(6)})
    span = times[min(needed): max(needed) + 1]
    for i, when in enumerate(span):
        conv.frame(when)
    store = conv.dataset(span)
    shift = min(needed)
    base = ERA5Window(store, indices=np.arange(0, 1), history=6, lead_steps=offsets[:1], variables=tuple(base_vars),
                      levels=levels, channels=cfg.analysis_channels, stats_cache=str(S / "hub/staging/stats.json"),
                      augment=False)
    assert base.variables == names
    t1 = time.time()
    with torch.inference_mode():
        bank = storm_feature_bank(model, store, base, starts - shift, targets, grid, cache=None, precision="fp32",
                                  batch_size=2, workers=1, progress=False)
    bank["issued"] = [str(t.track.time[t.point])[:13] for t in targets]
    torch.save(bank, target)
    print(f"{name}: {len(targets)} forecasts, {len(span)} frames ({t1 - t0:.0f} s data, "
          f"{time.time() - t1:.0f} s model)", flush=True)
print("ALL DONE", flush=True)
