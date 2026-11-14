# Copyright 2026 Nathan. Apache-2.0.
import json
import threading
import urllib.request

import numpy as np

from pluvium.ops.api import serve
from pluvium.render import bulletin, cap_alerts

PARISHES = [{"country": "JM", "parish": "Portland", "any_rain_24h_typical": 0.8, "any_rain_24h_highest": 0.9,
             "wettest_place": "Port Antonio", "earliest_likely_start": "Sat 2 pm", "thunder_24h_highest": 0.5,
             "rain_24h_mm_p90_highest": 60.0, "flood_highest": "high", "summary": "Showers likely."},
            {"country": "JM", "parish": "St. Elizabeth", "any_rain_24h_typical": 0.1, "any_rain_24h_highest": 0.2,
             "wettest_place": "Black River", "earliest_likely_start": "", "thunder_24h_highest": 0.0,
             "rain_24h_mm_p90_highest": 2.0, "flood_highest": "low", "summary": "Mostly dry."}]
TOWNS = [{"country": "JM", "parish": "Portland", "lat": 18.18, "lon": -76.45, "heavy_rain_24h": 0.4},
         {"country": "JM", "parish": "St. Elizabeth", "lat": 18.03, "lon": -77.85, "heavy_rain_24h": 0.0}]


def test_bulletins_and_alerts():
    en = bulletin(PARISHES, "JM", "Sat 06:00")
    jam = bulletin(PARISHES, "JM", "Sat 06:00", "jam")
    assert "Portland" in en and "Flash-flood" in en and "a go fall" in jam
    alerts = cap_alerts(PARISHES, TOWNS, "2026-11-14T06:00:00Z")
    assert len(alerts) == 3 and all("<status>Exercise</status>" in a for a in alerts)


def test_api_serves_the_latest_forecast(tmp_path):
    folder = tmp_path / "20261114T0600Z"
    folder.mkdir()
    (folder / "forecast.json").write_text(json.dumps({"t0": "x", "issued": "y", "parishes": PARISHES}))
    (folder / "towns.csv").write_text("place,country,parish\nPort Antonio,JM,Portland\n")
    np.savez(folder / "hourly.npz", p_wet=np.zeros((1, 72)), thunder=np.zeros((1, 72)), expected_mm=np.zeros((1, 72)))
    (tmp_path / "latest").symlink_to(folder.name)
    server = serve(tmp_path, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    town = json.loads(urllib.request.urlopen(f"{base}/town?name=Port%20Antonio&country=JM").read())
    assert town["town"]["parish"] == "Portland" and len(town["p_wet"]) == 72
    server.shutdown()
