# Copyright 2026 Nathan. Apache-2.0.
"""
A fair fight between architectures on real, held-out Caribbean data.

Each contender is trained from scratch on the same issue times, with the same data, the same seed and the same
wall-clock budget; isotonic calibration is fitted on one half of the held-out period and every score is computed on
the other half. Scores are reported twice: with the reanalysis as a perfect forecast (an upper bound on live skill)
and with emulated forecast errors (closer to live). References: climatology for that place, month and hour,
persistence, and the driving model's own rain.

Run it from Python (:func:`run_bench`) or ``pluvium bench --help``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .util import hour_of, log


def thunder_climatology(stations, hours: np.ndarray) -> np.ndarray:
    """``(S, 24)`` chance of thunder per station and local hour over ``hours`` (smoothed towards the network mean)."""
    from .weather import decode

    d = decode(stations.read(hours))                                      # (T, S)
    lst = ((hours[:, None] + 0.5 + stations.lon[None, :] / 15.0) % 24).astype(int)
    hit = np.zeros((len(stations), 24))
    cnt = np.zeros((len(stations), 24))
    for s in range(len(stations)):
        np.add.at(hit[s], lst[:, s][d["known"][:, s]], d["thunder"][:, s][d["known"][:, s]])
        np.add.at(cnt[s], lst[:, s][d["known"][:, s]], 1.0)
    mean = hit.sum() / max(cnt.sum(), 1)
    return (hit + 10 * mean) / (cnt + 10)


def thunder_scores(data: dict, stations, clim: np.ndarray) -> dict | None:
    if "thunder_p" not in data:
        return None
    p, obs, known, hours = data["thunder_p"], data["thunder_obs"], data["thunder_known"], data["station_hours"]
    lst = ((hours[:, :, None] + 0.5 + stations.lon[None, None, :] / 15.0) % 24).astype(int)
    c = clim[np.arange(len(stations))[None, None, :], lst]
    m = known
    if not m.any():
        return None
    b = float(np.mean((p[m] - obs[m]) ** 2))
    bc = float(np.mean((c[m] - obs[m]) ** 2))
    return {"brier": b, "brier_clim": bc, "bss_vs_clim": 1 - b / max(bc, 1e-12), "share": float(obs[m].mean()),
            "n": int(m.sum())}


def run_bench(geo_dir, era5_dir, cmorph_dir, stations_dir, out_dir,
              contenders=("pluvium-small-v1", "pluvium-small-v2", "pluvium-small"),
              train_period=("2021-06-01T00", "2021-09-24T00"), test_period=("2021-10-01T00", "2021-11-04T00"),
              minutes: float = 30.0, seed: int = 0, workers: int = 1, threads: int | None = None,
              test_every: int = 6, lr: float = 2e-3, batch_size: int = 2, ocean_targets: int = 100,
              max_land_targets: int = 450, only: str | None = None) -> dict:
    """Train and score each contender; returns (and writes ``bench.json``) the scores."""
    from .dataset import SampleBuilder, compute_climatology
    from .domain import Geography
    from .losses import PluviumLoss
    from .sources.cmorph import CMORPHCache
    from .sources.era5 import ERA5Cache
    from .sources.stations import StationCache
    from .train import (
        TrainConfig,
        Trainer,
        borrowed_climatology,
        build_model,
        fit_calibration,
        grids_of,
        hindcast,
        nearest_target,
        station_points,
    )
    from .verify import format_report

    if threads:
        torch.set_num_threads(threads)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    geo = Geography.load(geo_dir)
    era5, cmorph, stations = ERA5Cache(era5_dir), CMORPHCache(cmorph_dir), StationCache(stations_dir)
    train_hours = np.arange(hour_of(train_period[0]), hour_of(train_period[1]))
    test_hours = np.arange(hour_of(test_period[0]), hour_of(test_period[1]))
    clim_path = out / "climatology.npz"
    if clim_path.exists():
        from .dataset import Climatology

        clim = Climatology.load(clim_path)
    else:
        clim = compute_climatology(lambda h: cmorph.targets(h), geo.targets.lon, train_hours)
        clim.save(clim_path)
    st_clim = borrowed_climatology(clim, nearest_target(geo, stations.lat, stations.lon))
    thunder_clim = thunder_climatology(stations, train_hours)
    builder = SampleBuilder(geo, era5, cmorph, stations, clim, st_clim, ocean_targets=ocean_targets,
                            max_land_targets=max_land_targets, seed=seed)
    train_starts = builder.valid_starts(train_hours)
    held = builder.valid_starts(test_hours[::test_every])
    cal_starts, test_starts = held[0::2], held[1::2]
    log(f"bench: {len(geo.mesh)} mesh points, {len(geo.targets)} target cells, {len(stations)} airports; "
        f"{train_starts.size} training issue times, {cal_starts.size} for calibration, {test_starts.size} scored")
    results = {}
    for name in contenders:
        if only and name != only:
            continue
        done = out / f"{name}.json"
        if done.exists():                                   # finished in an earlier session
            results[name] = json.loads(done.read_text())
            log(f"{name}: already scored ({results[name]['steps']} steps); kept")
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(geo, grids_of(era5), name, station_points(geo, stations))
        loss = PluviumLoss(states=model.config.states)
        v3 = model.config.version >= 3                       # version 3 trains with its own recipe
        cfg = TrainConfig(size=name, batch_size=batch_size, lr=lr, warmup_steps=50, hours=minutes / 60.0,
                          workers=workers, eval_every_minutes=1e9, save_every_minutes=4.0, eval_forecasts=8,
                          log_every=50, seed=seed, ema_decay=0.995 if v3 else 0.0, curriculum=v3)
        builder = SampleBuilder(geo, era5, cmorph, stations, clim, st_clim, ocean_targets=ocean_targets,
                                max_land_targets=max_land_targets, seed=seed, augment=v3)
        trainer = Trainer(model, loss, builder, train_starts, cal_starts[:8], out / name, cfg)  # resumes a checkpoint
        trained = trainer.run()
        train_min = trainer.elapsed / 60
        model = trainer.model
        _, cal_data = hindcast(model, builder, cal_starts, batch_size=2, emulate=True, samples=100)
        calibrator = fit_calibration(cal_data)
        row = {"parameters": model.num_parameters(), "steps": trained["step"], "train_minutes": train_min}
        for label, emulate in (("reanalysis_driven", False), ("emulated_errors", True)):
            board, data = hindcast(model, builder, test_starts, batch_size=2, emulate=emulate, calibrator=calibrator,
                                   samples=100, stations=True)
            row[label] = board.report()
            th = thunder_scores(data, stations, thunder_clim)
            if th:
                row[label]["thunder_airports"] = th
            log(f"{name} -- {label}\n" + format_report(row[label]) +
                (f"\nthunder at airports: Brier {th['brier']:.4f}  climatology {th['brier_clim']:.4f}  skill "
                 f"{th['bss_vs_clim']:+.1%}" if th else ""))
        row["truth_sources"] = loss.sources.describe()
        results[name] = row
        (out / f"{name}.json").write_text(json.dumps(row, indent=1, default=float))
    if only:                                                 # merge with contenders scored in other runs
        for other in contenders:
            path = out / f"{other}.json"
            if other not in results and path.exists():
                results[other] = json.loads(path.read_text())
    results = {k: results[k] for k in contenders if k in results}
    summary = compare(results)
    (out / "bench.json").write_text(json.dumps({"results": results, "summary": summary}, indent=1, default=float))
    (out / "BENCHMARK.md").write_text(markdown(results), encoding="utf-8")
    return {"results": results, "summary": summary}


KEYS = (("bss_vs_clim", "hourly rain chance, skill vs climatology"),
        ("bss_vs_driving", "hourly rain chance, skill vs the driving model's rain"),
        ("csi_30", "yes/no at 30%: critical success index"),
        ("any_rain_6h.bss_vs_clim", "any rain in 6 h, skill vs climatology"),
        ("any_rain_24h.bss_vs_clim", "any rain in 24 h, skill vs climatology"),
        ("start_hour_error.within_2h", "rain start within 2 h of the truth"),
        ("start_hour_error.median_h", "rain start, median error (h)"),
        ("rpss_vs_clim", "intensity classes, skill vs climatology"),
        ("heavy.bss_vs_clim", "heavy hours, skill vs climatology"),
        ("crpss_24h_vs_clim", "24 h totals, CRPS skill vs climatology"),
        ("thunder_airports.bss_vs_clim", "thunder at airports, skill vs climatology"))


def _get(row: dict, key: str):
    cur = row
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def compare(results: dict) -> list[dict]:
    rows = []
    for label in ("emulated_errors", "reanalysis_driven"):
        for key, text in KEYS:
            rows.append({"test": label, "score": text, **{name: _get(r.get(label, {}), key) for name, r in results.items()}})
    return rows


def format_compare(summary: list[dict]) -> str:
    if not summary:
        return ""
    names = [k for k in summary[0] if k not in ("test", "score")]
    lines = [f"{'test':<18} {'score':<52} " + " ".join(f"{n:>18}" for n in names)]
    for r in summary:
        vals = []
        for n in names:
            v = r[n]
            vals.append(f"{'n/a':>18}" if v is None else f"{v:>18.3f}")
        lines.append(f"{r['test']:<18} {r['score']:<52} " + " ".join(vals))
    return "\n".join(lines)


def markdown(results: dict, setup: str = "") -> str:
    """A Markdown report of a benchmark: one table per test, contenders as columns, references in the text."""
    names = list(results)
    lines = ["# Benchmark: Pluvium versions 1, 2 and 3", "", setup, ""]
    for label, title in (("emulated_errors", "Driving fields with emulated forecast errors (closer to live)"),
                         ("reanalysis_driven", "Reanalysis as a perfect forecast (upper bound on live skill)")):
        lines += [f"## {title}", "", "| Score | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for key, text in KEYS:
            vals = []
            for n in names:
                v = _get(results[n].get(label, {}), key)
                if v is None:
                    vals.append("n/a")
                elif key.endswith("median_h"):
                    vals.append(f"{v:.1f} h")
                else:
                    vals.append(f"{v:+.1%}" if "bss" in key or "rpss" in key or "crpss" in key else f"{v:.3f}")
            lines.append(f"| {text} | " + " | ".join(vals) + " |")
        r0 = results[names[0]].get(label, {})
        if r0:
            lines += ["", f"References on the same hours: climatology Brier {r0.get('brier_clim', float('nan')):.4f}, "
                          f"persistence {r0.get('brier_persist', float('nan')):.4f}, driving model's rain "
                          f"{r0.get('brier_driving', float('nan')):.4f}; {r0.get('hours_scored', 0):,} target-hours, "
                          f"{r0.get('wet_share', 0):.1%} wet.", ""]
    lines += ["Training: " + "; ".join(f"{n}: {results[n]['parameters']:,} parameters, {results[n]['steps']:,} steps in "
                                       f"{results[n]['train_minutes']:.0f} min" for n in names), ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="pluvium bench", description="Train and score architectures on held-out data.")
    p.add_argument("--geography", required=True)
    p.add_argument("--era5", required=True)
    p.add_argument("--cmorph", required=True)
    p.add_argument("--stations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--contenders", default="pluvium-small-v1,pluvium-small-v2,pluvium-small")
    p.add_argument("--minutes", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--only", default=None)
    a = p.parse_args(argv)
    got = run_bench(a.geography, a.era5, a.cmorph, a.stations, a.out, tuple(a.contenders.split(",")), minutes=a.minutes,
                    workers=a.workers, threads=a.threads, only=a.only)
    print(format_compare(got["summary"]))


if __name__ == "__main__":
    main()


__all__ = ["compare", "format_compare", "main", "markdown", "run_bench", "thunder_climatology", "thunder_scores"]
