# Copyright 2026 Nathan. Apache-2.0.
"""
The operational cycle, once an hour.

1. Learn: archived forecasts are checked against the airport reports of the hours that have passed and the online
   corrections are updated (:mod:`pluvium.online`).
2. Forecast: the live forecast is made (:class:`pluvium.forecast.Forecaster`); every output is checked and repaired
   (:mod:`pluvium.guard.consistency`). If the network fails, the statistical fallback is issued instead.
3. Publish: the outputs go to a new timestamped folder, which then atomically becomes ``latest``, so readers never see
   a half-written forecast; CAP alerts and the bulletin are written alongside; old folders beyond ``keep`` are removed.

Failures are retried with backoff; a cycle that still fails leaves the previous ``latest`` in place and says so.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from ..util import log


def _publish(staging: Path, root: Path) -> Path:
    latest = root / "latest"
    tmp = root / ".latest.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(staging.name, tmp)
    os.replace(tmp, latest)
    return latest


def run_cycle(forecaster, root, countries=None, keep: int = 48, attempts: int = 3, learn: bool = True) -> dict:
    """One cycle; returns ``{"ok", "folder", "notes"}``."""
    from ..guard.consistency import check_summary
    from ..render.alerts import cap_alerts
    from ..render.bulletin import bulletins

    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    notes = []
    if learn:
        try:
            from ..online import ForecastArchive
            from ..online import learn as learn_from
            from ..sources.metar import fetch_metars, report_hour, station_codes

            stations = forecaster.station_meta
            if stations:
                metars = fetch_metars(hours=72)
                got = learn_from(ForecastArchive(forecaster.state_dir / "archive"), forecaster.corrector,
                                 lambda hours: station_codes(stations, metars, hours), report_hour(time.time() + 300) + 1)
                forecaster.corrector.save(forecaster.state_dir / "corrector.json")
                notes.append(f"learned from {got['hours_verified']:,} airport-hours")
        except Exception as error:
            notes.append(f"learning skipped ({type(error).__name__}: {error})")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    folder = root / stamp
    for attempt in range(attempts):
        try:
            result = forecaster.run(out_dir=None, countries=countries)
            repaired = check_summary(result["summary"])
            from ..forecast import write_outputs

            write_outputs(folder, forecaster.towns, result["summary"], {**result["info"], "repairs": repaired},
                          forecaster.model.schedule, countries)
            parishes = json.loads((folder / "forecast.json").read_text())["parishes"]
            import csv

            towns = list(csv.DictReader(open(folder / "towns.csv", encoding="utf-8")))
            for t in towns:
                for k in ("lat", "lon", "heavy_rain_24h"):
                    if t.get(k) not in (None, ""):
                        t[k] = float(t[k])
            alerts = cap_alerts(parishes, towns, result["info"]["issued"] + ":00")
            (folder / "alerts").mkdir(exist_ok=True)
            for k, xml in enumerate(alerts):
                (folder / "alerts" / f"alert_{k:03d}.xml").write_text(xml, encoding="utf-8")
            for country in sorted({p["country"] for p in parishes}):
                for name, text in bulletins(parishes, country, result["info"]["issued"]).items():
                    (folder / name).write_text(text, encoding="utf-8")
            _publish(folder, root)
            notes += repaired
            break
        except Exception as error:
            notes.append(f"attempt {attempt + 1} failed: {type(error).__name__}: {error}")
            log(traceback.format_exc(limit=3))
            time.sleep(min(300, 30 * 2**attempt))
    else:
        return {"ok": False, "folder": None, "notes": notes}
    stamps = sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2] == "20" and p.name != folder.name)
    for old in stamps[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    log(f"cycle {stamp}: published; " + "; ".join(notes[:5]))
    return {"ok": True, "folder": str(folder), "notes": notes}


def run_forever(forecaster, root, countries=None, minute: int = 20) -> None:
    """Run a cycle every hour at ``minute`` past (driving runs and airport reports have arrived by then)."""
    while True:
        run_cycle(forecaster, root, countries)
        now = time.time()
        wait = (3600 - (now % 3600) + minute * 60) % 3600 or 3600
        time.sleep(wait)


__all__ = ["run_cycle", "run_forever"]
