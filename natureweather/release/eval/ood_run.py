"""
Red team: the full NatureV1 model on storms outside its training basin (the hurricane heads saw only the
Atlantic). Tracks from IBTrACS (USA agency fields: 1-minute winds like HURDAT2); ERA5 rebuilt at 1.5 degrees
from ARCO-ERA5 (validated against WeatherBench 2). One forecast per day at 00Z. Usage:
    python ood_run.py S EP:RICK:2021 WP:RAI:2021 SI:SEROJA:2021 ...
"""
import csv, json, sys, time
from pathlib import Path

import numpy as np
import torch

S = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
sys.path.insert(0, str(Path(__file__).parent))
from arco15 import ArcoTo15  # noqa: E402
from ihelix import fibonacci_sphere  # noqa: E402
from naturev1.besttrack import Track  # noqa: E402
from naturev1.era5 import ERA5Window, era5_source_grid, lead_offsets  # noqa: E402
from naturev1.model import NatureConfig, NatureV1  # noqa: E402
from naturev1.storms import pair_tracks_with_reanalysis, storm_feature_bank  # noqa: E402
from naturev1.upper import open_weatherbench_levels  # noqa: E402


def num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def ibtracs_track(path, name, season) -> Track:
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        next(reader)                                               # the units row
        for row in reader:
            if row["NAME"] == name and row["SEASON"] == str(season):
                rows.append(row)
    rows = [r for r in rows if r["ISO_TIME"][11:13] in ("00", "06", "12", "18") and r["ISO_TIME"][14:16] == "00"]
    if not rows:
        raise SystemExit(f"{name} {season} not found")
    sid = rows[0]["SID"]
    rows = [r for r in rows if r["SID"] == sid]
    wind = np.array([num(r["USA_WIND"]) for r in rows], dtype=np.float32)
    status = []
    for r, w in zip(rows, wind):
        s = r["USA_STATUS"].strip()
        tropical = s in ("TD", "TS", "HU", "TY", "ST", "TC") or (not s and r["NATURE"] == "TS")
        if not tropical or not np.isfinite(w):
            status.append("EX")
        else:
            status.append("TD" if w < 34 else "TS" if w < 64 else "HU")
    dist = np.array([num(r["DIST2LAND"]) for r in rows])
    record = np.array(["L" if i > 0 and dist[i] == 0 and dist[i - 1] > 0 else "" for i in range(len(rows))])
    radii = np.full((len(rows), 3, 4), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        for g, kt in enumerate((34, 50, 64)):
            for q, quad in enumerate(("NE", "SE", "SW", "NW")):
                radii[i, g, q] = num(r.get(f"USA_R{kt}_{quad}"))
    lon = np.array([num(r["LON"]) for r in rows], dtype=np.float32)
    return Track(storm_id=sid, name=name, time=np.array([np.datetime64(r["ISO_TIME"][:16].replace(" ", "T"), "s") for r in rows]),
                 latitude=np.array([num(r["LAT"]) for r in rows], dtype=np.float32),
                 longitude=((lon + 180.0) % 360.0 - 180.0).astype(np.float32),
                 max_wind_kt=wind, min_pressure_hpa=np.array([num(r["USA_PRES"]) for r in rows], dtype=np.float32),
                 status=np.array(status), rmw_nmi=np.array([num(r.get("USA_RMW")) for r in rows], dtype=np.float32),
                 wind_radii_nmi=radii, record=record)


if __name__ == "__main__":
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
    out_dir = S / "ood"; out_dir.mkdir(exist_ok=True)
    for spec in sys.argv[2:]:
        basin, name, season = spec.split(":")
        target = out_dir / f"{basin}_{name}_{season}.pt"
        if target.exists():
            print(f"{spec}: done already", flush=True); continue
        track = ibtracs_track(S / f"hurdat/ibtracs_{basin}.csv", name, int(season))
        t0 = time.time()
        first = np.datetime64(str(track.time[0])[:13], "h") - np.timedelta64(36, "h")
        last = np.datetime64(str(track.time[-1])[:13], "h") + np.timedelta64(126, "h")
        times = np.arange(first, last, np.timedelta64(6, "h"))
        starts, targets, _ = pair_tracks_with_reanalysis([track], times.astype("datetime64[s]").astype(np.int64),
                                                         offsets, history=6)
        keep = [i for i, t in enumerate(targets) if str(t.track.time[t.point])[11:13] == "00"]
        if not keep:
            print(f"{spec}: no tropical 00Z points", flush=True); continue
        starts, targets = starts[keep], [targets[i] for i in keep]
        needed = sorted({int(s) + k for s in starts for k in range(6)})
        span = times[min(needed): max(needed) + 1]
        for when in span:
            conv.frame(when)
        store = conv.dataset(span)
        base = ERA5Window(store, indices=np.arange(0, 1), history=6, lead_steps=offsets[:1],
                          variables=tuple(base_vars), levels=levels, channels=cfg.analysis_channels,
                          stats_cache=str(S / "hub/staging/stats.json"), augment=False)
        assert base.variables == names
        t1 = time.time()
        with torch.inference_mode():
            bank = storm_feature_bank(model, store, base, starts - min(needed), targets, grid, cache=None,
                                      precision="fp32", batch_size=2, workers=1, progress=False)
        bank["issued"] = [str(t.track.time[t.point])[:13] for t in targets]
        bank["peak_kt"] = float(np.nanmax(track.max_wind_kt))
        torch.save(bank, target)
        print(f"{spec}: {len(targets)} forecasts ({t1 - t0:.0f} s data, {time.time() - t1:.0f} s model)", flush=True)
    print("ALL DONE", flush=True)
