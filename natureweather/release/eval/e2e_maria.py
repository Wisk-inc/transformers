import sys, json, time, torch, numpy as np
from ihelix import fibonacci_sphere
from naturev1.model import NatureConfig, NatureV1
from naturev1.upper import open_weatherbench_levels
from naturev1.era5 import ERA5Window, era5_source_grid, lead_offsets
from naturev1.besttrack import parse_hurdat2
from naturev1.storms import (pair_tracks_with_reanalysis, storm_feature_bank, _head_outputs, track_members,
                             consensus_track, _great_circle_km)
S = sys.argv[1]; WHEN = sys.argv[2:] or ["2017-09-18T00"]
torch.set_num_threads(max(torch.get_num_threads(), 4))
ck = torch.load(f"{S}/hub/checkpoints/stage2/latest.pt", map_location="cpu", weights_only=False)
cfg = NatureConfig.from_dict(ck["config"])
model = NatureV1(cfg, fibonacci_sphere(cfg.latent_points, num_neighbours=cfg.latent_neighbours, cluster_size=cfg.latent_cluster))
model.load_state_dict(ck["model"]); model.eval()
stats = json.load(open(f"{S}/hub/staging/stats.json"))
names = stats["variables"]
levels = sorted({int(v.split("@")[1]) for v in names if "@" in v})
base_vars = list(dict.fromkeys(v.split("@")[0] for v in names))
ERA5, SURF, UP = open_weatherbench_levels()
SRC, LAT, LON = era5_source_grid(ERA5)
OFFSETS = lead_offsets(cfg.lead_times_hours)
times = ERA5.time.values
base = ERA5Window(ERA5, indices=np.arange(len(times) - 40, len(times) - 39), history=6, lead_steps=OFFSETS,
                  variables=tuple(base_vars), levels=levels, channels=cfg.analysis_channels,
                  stats_cache=f"{S}/hub/staging/stats.json", augment=False)
assert base.variables == names, "channel order differs from training"
maria = [t for t in parse_hurdat2(f"{S}/hurdat/hurdat2.txt") if t.name == "MARIA" and t.year == 2017]
starts, targets, report = pair_tracks_with_reanalysis(maria, times.astype("datetime64[s]").astype(np.int64), OFFSETS, history=6)
keep = [i for i, t in enumerate(targets) if any(str(t.track.time[t.point]).startswith(w) for w in WHEN)]
starts, targets = starts[keep], [targets[i] for i in keep]
print(f"Maria samples: {len(keep)} ({', '.join(str(t.track.time[t.point])[:13] for t in targets)})", flush=True)
t0 = time.time()
bank = storm_feature_bank(model, ERA5, base, starts, targets, SRC, cache=None, precision="fp32", workers=1, progress=False)
print(f"backbone run on CPU from raw ERA5: {time.time()-t0:.0f} s", flush=True)
torch.save(bank, f"{S}/maria_e2e_bank.pt")
