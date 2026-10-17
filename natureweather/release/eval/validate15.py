import sys, time, json, numpy as np
sys.path.insert(0, sys.argv[1])
from arco15 import ArcoTo15, SURFACE, UPPER, ACCUMULATED, LEVELS
from naturev1.upper import open_weatherbench_levels
wb2, _, _ = open_weatherbench_levels()
print("WB2 levels:", [int(x) for x in wb2.level.values])
conv = ArcoTo15(wb2, f"{sys.argv[1]}/arco_cache")
when = np.datetime64("2021-08-29T12")
t0 = time.time(); mine = conv.dataset([when]); print(f"one frame built in {time.time()-t0:.0f} s")
ref = wb2.sel(time=[when]).load()
stats = json.load(open(f"{sys.argv[1]}/hub/staging/stats.json")); std = dict(zip(stats["variables"], stats["std"]))
worst = []
for name in SURFACE + list(ACCUMULATED) + UPPER:
    a, b = mine[name], ref[name].transpose(*mine[name].dims)
    if "level" in a.dims:
        for lv in LEVELS:
            x, y = a.sel(level=lv).values, b.sel(level=lv).values
            ok = np.isfinite(x) & np.isfinite(y)
            worst.append((float(np.sqrt(np.mean((x[ok]-y[ok])**2)) / std[f"{name}@{lv}"]), f"{name}@{lv}"))
    else:
        x, y = a.values, b.values; ok = np.isfinite(x) & np.isfinite(y)
        mism = int((np.isfinite(x) != np.isfinite(y)).sum())
        worst.append((float(np.sqrt(np.mean((x[ok]-y[ok])**2)) / std[name]), f"{name} (nan mismatch {mism})"))
worst.sort(reverse=True)
print("difference from WeatherBench 2, as a fraction of each channel's natural spread (std):")
for v, n in worst[:12]: print(f"  {v:8.4f}  {n}")
print(f"  median over {len(worst)} channels: {np.median([v for v,_ in worst]):.4f}")
