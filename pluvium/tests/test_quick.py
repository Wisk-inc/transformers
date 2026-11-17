# Copyright 2026 Nathan. Apache-2.0.
"""One call to a forecast: the pieces of ``pluvium now``, and trained blends shared through a Hub repository."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_frontier_model import make_table

from pluvium import quick
from pluvium.area import Area
from pluvium.blendlive import read_state
from pluvium.frontier.model import FrontierBlend, load_blend
from pluvium.sources.baselines import AIGFS, GRAPHCAST, graphcast_template
from pluvium.sources.cmorph_daily import truth_until
from pluvium.util import hour_of

AREA = Area("testland", {"TST": "Testland"}, ("TL",), (10.0, 12.0, -80.0, -78.0), (4.0, 18.0, -86.0, -72.0), {"TL": ("en",)})


@pytest.fixture(scope="module")
def model():
    table, clim = make_table(days=40, n=40)                    # under 90 days: no storms are staged, so it is quick
    grid = (np.linspace(10.0, 12.0, 5), np.linspace(-80.0, -78.0, 8))
    grid_clim = {m: {k: np.full((5, 8), float(v[0]), np.float32) for k, v in clim.items()} for m in range(1, 13)}
    cells = (np.repeat(grid[0], 8), np.tile(grid[1], 5))
    return FrontierBlend.from_table(table, 40, ["ens", "gfs"], AREA.to_json(), grid_clim, grid, cells, max_rows=8_000,
                                    pool_rows=5_000, meta={"trained": ["2024-04-01T00", "2024-06-20T00"], "runs": 40})


class FolderHub:
    """A folder standing in for the Hub: ``upload_folder`` copies into it, :func:`url` reads from it."""

    def __init__(self, root):
        self.root, self.created = Path(root), None

    def create_repo(self, repo, repo_type="model", exist_ok=True, private=False):
        self.created = repo

    def upload_folder(self, folder_path, path_in_repo, repo_id, repo_type, commit_message):
        for path in Path(folder_path).iterdir():
            target = self.root / repo_id / path_in_repo / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())

    def url(self, repo, name, file):
        return str(self.root / repo / "blends" / name / file)


def listing(tmp_path, days_by_month):
    """Bucket listings standing in for NOAA's: one file a month naming the days published."""
    for month, days in days_by_month.items():
        y, m = month.split("-")
        rows = "".join(f"<CommonPrefixes><Prefix>data/30min/8km/{y}/{m}/{d:02d}/</Prefix></CommonPrefixes>" for d in days)
        (tmp_path / f"{month}.xml").write_text(f"<ListBucketResult>{rows}</ListBucketResult>")
    return str(tmp_path) + "/{y}-{m}.xml"


def test_training_year_ends_where_the_rain_record_ends(tmp_path, monkeypatch):
    # November 2026: NOAA has published May in full and is still filling June in; nothing later exists
    months = {f"2026-{m:02d}": [] for m in range(7, 12)}
    months.update({"2026-06": [1, 2, 3, 5, 6, 8, 9, 16, 18, 21, 26, 29], "2026-05": list(range(1, 32))})
    now = hour_of("2026-11-15T10")
    last = truth_until(now, listing=listing(tmp_path, months))
    assert last == hour_of("2026-05-31T00")                     # a month with a few days is not counted yet
    assert truth_until(now, listing="file://" + str(tmp_path / "unreachable") + "/{y}-{m}.xml") is None
    assert quick.recent_year(now, last) == ("2025-05-16T00", "2026-05-16T00")   # 15 days before the record ends
    monkeypatch.setattr("pluvium.sources.cmorph_daily.truth_until", lambda now=None: None)
    assert quick.recent_year(now) == ("2025-04-29T00", "2026-04-29T00")          # no listing: a cautious guess
    assert quick.codes_of("do, ht") == ["DO", "HT"] and quick.codes_of(["jm"]) == ["JM"]
    assert quick.default_root(["DO", "HT"]).name == "pluvium-DO-HT"
    with pytest.raises(ValueError):
        quick.codes_of(" , ")


def test_learning_catches_up_with_a_rain_record_that_arrives_late(model, tmp_path, monkeypatch):
    model.save(tmp_path / "blend")
    blend = load_blend(tmp_path / "blend")                      # a copy: learning changes the weights
    n = blend.cells[0].size

    class Reader:
        def __init__(self, cache, regions):
            pass

        def read(self, system, init, days):
            return True

        def load(self, name, system, init):
            return {}

    class Rain:
        """The record holds every day up to ``last`` but one."""

        def __init__(self, last, gap):
            self.last, self.gap = hour_of(last), hour_of(gap)

        def days(self, name, hours):
            known = hours[0] <= self.last and hours[0] != self.gap
            return [np.full((5, 8), 4.0 if known else np.nan, np.float32)]

    monkeypatch.setattr("pluvium.frontier.model.ArchiveReader", Reader)
    monkeypatch.setattr("pluvium.benchmarks.fullyear.system_at_cells",
                        lambda z, lead, cells: {"mean": np.full(n, 1.2, np.float32), "_members": None})
    first = blend.first_unlearned()
    assert first == hour_of("2024-07-05T00")                    # 15 days after training ended: runs it has not seen
    # a day is verified by runs on the training's pattern: every second day to 3 days ahead, every fourth to 14
    day = hour_of("2024-04-13T00")                              # 12 days after the first training run
    runs = blend.verifying_runs(day)
    assert [(lead, reach) for lead, _, reach in runs] == [(1, 14), (3, 3), (5, 14), (9, 14), (13, 14)]
    assert all((day - init) // 24 == lead - 1 for lead, init, _ in runs)
    assert [lead for lead, _, _ in blend.verifying_runs(day + 24)] == [2, 6, 10, 14]
    assert blend.verifying_runs(day, leads=(1, 7)) == [(1, day, 14), (7, day - 144, 14)]
    rain, now = Rain("2024-07-20T00", "2024-07-07T00"), hour_of("2025-01-10T00")
    before = blend.online.updates
    got = blend.learn(tmp_path / "cache", rain, now, max_days=4, since=first, until=rain.last)
    assert got == ["2024-07-05", "2024-07-06", "2024-07-08", "2024-07-09"]      # oldest first, the gap skipped
    assert blend.online.updates > before and blend.first_unlearned() == hour_of("2024-07-10T00")
    rest = blend.learn(tmp_path / "cache", rain, now, max_days=30, since=blend.first_unlearned(), until=rain.last)
    assert rest[0] == "2024-07-10" and rest[-1] == "2024-07-20" and len(rest) == 11
    assert blend.learn(tmp_path / "cache", rain, now, since=blend.first_unlearned(), until=rain.last) == []
    monkeypatch.setattr(Reader, "read", lambda self, system, init, days: False)     # the archives cannot be reached
    later = Rain("2024-07-25T00", "2024-07-07T00")
    assert blend.learn(tmp_path / "cache", later, now, since=blend.first_unlearned(), until=later.last) == []
    assert blend.first_unlearned() == hour_of("2024-07-21T00")  # nothing was marked as learned, so it is tried again


def test_noaa_graphcast_is_read_from_aigfs_after_the_feed_changed():
    assert graphcast_template(hour_of("2026-05-05T18")) is GRAPHCAST
    assert graphcast_template(hour_of("2026-05-06T00")) is AIGFS
    old = graphcast_template(hour_of("2025-07-01T00")).format(day="20250701", hour=0, step=24)
    new = graphcast_template(hour_of("2026-11-14T00")).format(day="20261114", hour=0, step=24)
    assert old.endswith("graphcastgfs.20250701/00/forecasts_13_levels/graphcastgfs.t00z.pgrb2.0p25.f024")
    assert new.endswith("aigfs.20261114/00/model/atmos/grib2/aigfs.t00z.sfc.f024.grib2")


def test_saved_blend_opens_without_running_code_and_an_old_file_is_refused_from_outside(model, tmp_path):
    model.save(tmp_path / "new")
    state = read_state(tmp_path / "new" / "blend.pt", trusted=False)
    assert torch.is_tensor(state["online"]["logw"]) and isinstance(state["online"]["updates"], int)
    again = load_blend(tmp_path / "new", trusted=False)
    assert np.allclose(again.online.logw, model.online.logw) and again.online.updates == model.online.updates
    # the format before 0.4 kept numpy arrays, which only the full unpickler reads
    old = dict(state, online=model.online.state())
    (tmp_path / "old").mkdir()
    for name in ("blend.json", "climatology.npz"):
        (tmp_path / "old" / name).write_bytes((tmp_path / "new" / name).read_bytes())
    torch.save(old, tmp_path / "old" / "blend.pt")
    assert load_blend(tmp_path / "old").online.updates == model.online.updates       # made here: opened
    with pytest.raises(ValueError, match="safe format"):
        load_blend(tmp_path / "old", trusted=False)                                    # from elsewhere: refused


def test_published_blend_is_downloaded_and_forecasts_the_same(model, tmp_path, monkeypatch):
    hub = FolderHub(tmp_path / "hub")
    (tmp_path / "trainer").mkdir()
    model.save(tmp_path / "trainer" / "blend")
    said = []
    where = quick.publish_blend("org/repo", tmp_path / "trainer", api=hub, say=said.append)
    assert where.endswith("org/repo/tree/main/blends/testland") and hub.created == "org/repo"
    card = (tmp_path / "hub" / "org/repo" / "blends" / "testland" / "README.md").read_text()
    assert "Testland" in card and "2024-04-01 to 2024-06-20" in card and "without running any code" in card
    # somebody else, with an empty folder for the same country
    user = tmp_path / "user"
    user.mkdir()
    AREA.save(user / "area.json")
    monkeypatch.setattr(quick, "blend_url", hub.url)
    root, got, how = quick.ready("TL", user, "org/repo", train=False, say=said.append)
    assert how == "downloaded" and (user / "blend" / "FETCHED").exists()
    assert any("downloaded from org/repo" in s for s in said)
    table, clim = make_table(days=40, n=40)
    per = {s: table["feats"][s][7] for s in ("ens", "gfs")}
    assert np.allclose(got.chances(per, clim, 2.0, 40), model.chances(per, clim, 2.0, 40), atol=1e-6)
    assert quick.ready("TL", user, "org/repo", train=False, say=said.append)[2] == "local"   # second call: nothing to do


def test_download_is_refused_for_another_box_an_unsafe_file_or_a_missing_part(model, tmp_path, monkeypatch):
    hub = FolderHub(tmp_path / "hub")
    (tmp_path / "trainer").mkdir()
    model.save(tmp_path / "trainer" / "blend")
    quick.publish_blend("org/repo", tmp_path / "trainer", api=hub, say=lambda s: None)
    folder = tmp_path / "hub" / "org/repo" / "blends" / "testland"
    said = []
    elsewhere = dict(AREA.to_json(), target_box=[20.0, 22.0, -80.0, -78.0])
    assert not quick.fetch_blend("org/repo", elsewhere, tmp_path / "a", say=said.append, url=hub.url)
    assert "different box" in said[-1] and not (tmp_path / "a" / "blend").exists()
    assert not quick.fetch_blend("org/none", AREA.to_json(), tmp_path / "a", say=said.append, url=hub.url)
    good = (folder / "blend.pt").read_bytes()
    torch.save({"nets": {}, "online": {"logw": np.zeros(3)}}, folder / "blend.pt")     # needs the unsafe unpickler
    assert not quick.fetch_blend("org/repo", AREA.to_json(), tmp_path / "b", say=said.append, url=hub.url)
    assert "could not be opened safely" in said[-1] and not (tmp_path / "b" / "blend").exists()
    (folder / "blend.pt").write_bytes(good)
    (folder / "climatology.npz").unlink()
    assert not quick.fetch_blend("org/repo", AREA.to_json(), tmp_path / "c", say=said.append, url=hub.url)
    assert "incomplete" in said[-1]
    user = tmp_path / "d"
    user.mkdir()
    AREA.save(user / "area.json")
    monkeypatch.setattr(quick, "blend_url", hub.url)
    with pytest.raises(SystemExit, match="pluvium blend train"):                       # nothing usable, told not to train
        quick.ready("TL", user, "org/repo", train=False, say=said.append)


def test_brief_shows_the_largest_places_or_the_one_asked_for():
    names = ["Hamlet", "Capital", "Port Town", "Capital Heights"]
    population = np.array([120, 900_000, 40_000, 3_000])
    day = {"date": "2026-11-15", "p_ge_1": np.array([0.2, 0.71, 0.5, 0.6]), "p_ge_25": np.array([0.0, 0.12, 0.3, 0.1]),
           "mean_mm": np.array([0.4, 6.2, 14.0, 5.0]), "alert": np.array([0, 0, 2, 0]),
           "beyond_record": np.array([False, False, True, False])}
    result = {"days": [day, dict(day, date="2026-11-16")]}
    text = quick.brief(result, names, population, top=2, days=1)
    assert text.splitlines()[0] == "Capital" and "Port Town" in text and "Hamlet" not in text
    assert "Sun 15 Nov   rain  71%   25 mm or more  12%   expected   6.2 mm" in text
    assert "[warning, beyond the training record]" in text and "16 Nov" not in text
    asked = quick.brief(result, names, population, place="capital", days=2)
    assert asked.splitlines()[0] == "Capital" and "Capital Heights" in asked and "Port Town" not in asked
    assert "no place called" in quick.brief(result, names, population, place="atlantis")


def test_parishes_are_named_one_way_summarised_and_can_be_asked_for(tmp_path):
    from pluvium.blendlive import write_table

    # GeoNames spells Jamaica's parishes three ways; people type a fourth
    assert [quick._tidy(n) for n in ("Portland Parish", "St. Elizabeth", "Clarendon", "St Ann")] == [
        "Portland", "Saint Elizabeth", "Clarendon", "Saint Ann"]
    assert quick._plain("St. Ann") == quick._plain("saint ann") == quick._plain("ST ANN") == "saint ann"
    names = ["Hamlet", "Capital", "Port Town", "Capital Heights"]
    areas = ["Saint Ann", "Kingston", "Saint Ann", "Kingston"]
    population = np.array([120, 900_000, 40_000, 3_000])
    day = {"date": "2026-11-15", "p_ge_1": np.array([0.2, 0.71, 0.5, 0.6]), "p_ge_25": np.array([0.0, 0.12, 0.3, 0.1]),
           "mean_mm": np.array([0.4, 6.2, 14.0, 5.0]), "alert": np.array([0, 0, 2, 0]),
           "beyond_record": np.array([False, False, True, False])}
    result = {"days": [day, dict(day, date="2026-11-16")]}
    text = quick.by_area(result, areas, days=1)
    lines = text.splitlines()
    assert lines[0].startswith("each line: the typical place") and lines[1] == "Kingston (2 places)"
    # the median of Kingston's two places, and the range across them
    assert lines[2] == "  Sun 15 Nov   rain  66% (60 to 71)   25 mm or more  11% (up to 12)   expected   5.6 mm (up to 6.2)"
    assert lines[3] == "Saint Ann (2 places)" and lines[4].endswith("[warning for part of it]") and "16 Nov" not in text
    one = quick.by_area(result, areas, days=2, area="St Ann")
    assert "Saint Ann (2 places)" in one and "Kingston" not in one and "16 Nov" in one
    assert quick.by_area(result, areas, area="Atlantis").startswith("no area called 'Atlantis': this country has Kingston")
    # the largest towns of one parish, each labelled with its parish
    towns = quick.brief(result, names, population, top=5, days=1, areas=areas, area="st. ann")
    assert towns.splitlines()[0] == "Port Town, Saint Ann" and "Hamlet, Saint Ann" in towns and "Capital" not in towns
    assert quick.brief(result, names, population, areas=areas, area="nowhere").startswith("no area called")
    assert "no town list" in quick.brief(result, names, population, area="Saint Ann")
    # ... and the table says which parish every row is in
    full = {"days": [dict(day, **{f"p_ge_{t}": day["p_ge_1"] for t in (10, 50)}, median_mm=day["mean_mm"],
                          p90_mm=day["mean_mm"], systems=["ens"])]}
    rows = write_table(full, names, tmp_path / "t.csv", areas=areas).read_text().splitlines()
    assert rows[0].startswith("place,area,date,p_ge_1mm") and rows[2].startswith("Capital,Kingston,2026-11-15,0.71")
    assert write_table(full, names, tmp_path / "u.csv").read_text().splitlines()[0].startswith("place,date,")


def test_first_days_say_when_in_the_day_and_each_period_can_be_listed(tmp_path):
    names = ["Hamlet", "Capital"]
    areas = ["Saint Ann", "Kingston"]
    population = np.array([120, 900_000])
    day = {"date": "2026-11-15", "p_ge_1": np.array([0.2, 0.7]), "p_ge_25": np.array([0.0, 0.1]),
           "mean_mm": np.array([0.4, 6.2])}
    result = {"days": [day, dict(day, date="2026-11-16"), dict(day, date="2026-11-17"), dict(day, date="2026-11-18")]}
    chance = np.zeros((3, 4, 2))
    chance[:, :, 1] = [0.1, 0.05, 0.3, 0.6]                              # the capital: afternoons
    chance[:, :, 0] = [0.05, 0.02, 0.1, 0.15]                            # the hamlet: rain unlikely all day
    labels = ["Sat 7 pm to 1 am", "Sun 1 am to 7 am", "Sun 7 am to 1 pm", "Sun 1 pm to 7 pm"] + [f"p{k}" for k in range(8)]
    periods = {"labels": labels, "starts": [f"2026-11-15T{6 * (k % 4):02d}" for k in range(12)], "chance": chance,
               "systems": ["ens"], "zone": "America/Jamaica", "days": 3}
    text = quick.brief(result, names, population, top=2, days=4, periods=periods)
    lines = text.splitlines()
    assert lines[1].endswith("most likely Sun 1 pm to 7 pm (60%)")       # the capital, first day
    assert "most likely" not in lines[4]                                 # the fourth day has no periods
    assert not any("most likely" in line for line in lines[6:])          # the hamlet: unlikely in every period
    listed = quick.brief(result, names, population, top=1, days=2, periods=periods, when=True).splitlines()
    assert "most likely" not in listed[1] and listed[2].startswith("      Sat 7 pm to 1 am")
    assert listed[5] == "      Sun 1 pm to 7 pm       rain  60%" and len(listed) == 1 + 2 * 5
    by = quick.by_area(result, areas, days=1, periods=periods)
    assert "Kingston (1 places)" in by and by.splitlines()[2].endswith("most likely Sun 1 pm to 7 pm (60%)")
    rows = quick.write_periods(periods, names, tmp_path / "p.csv", areas=areas).read_text().splitlines()
    assert rows[0] == "place,area,starts_utc,period,p_ge_1mm" and len(rows) == 1 + 12 * 2
    assert rows[8] == "Capital,Kingston,2026-11-15T18,Sun 1 pm to 7 pm,0.60"
    assert quick.zone_of(tmp_path) == "UTC" and quick.usual_at(tmp_path, [18.0], [-77.0], 0, 3) is None


def test_a_period_that_is_over_is_never_the_most_likely_one():
    """A day of the forecast is a UTC day, so in Jamaica its first period is the evening before. A forecast read on
    Tuesday morning said Tuesday's rain was most likely "Mon 7 pm to 1 am", which was over by then."""
    names, population = ["Montego Bay"], np.array([110_000])
    day = {"date": "2026-11-17", "p_ge_1": np.array([0.48]), "p_ge_25": np.array([0.01]), "mean_mm": np.array([2.8])}
    result = {"days": [day, dict(day, date="2026-11-18")]}
    chance = np.zeros((2, 4, 1))
    chance[0, :, 0] = [0.27, 0.10, 0.12, 0.21]                          # the evening before is the wettest period
    chance[1, :, 0] = [0.05, 0.05, 0.10, 0.28]
    day0 = hour_of("2026-11-17T00")
    labels = ["Mon 7 pm to 1 am", "Tue 1 am to 7 am", "Tue 7 am to 1 pm", "Tue 1 pm to 7 pm",
              "Tue 7 pm to 1 am", "Wed 1 am to 7 am", "Wed 7 am to 1 pm", "Wed 1 pm to 7 pm"]
    periods = {"labels": labels, "starts": [""] * 8, "hours": [day0 + 6 * k for k in range(8)], "chance": chance,
               "systems": ["ens"], "zone": "America/Jamaica", "days": 2, "issued": day0 + 14}   # 9 am in Jamaica
    lines = quick.brief(result, names, population, top=1, days=2, periods=periods).splitlines()
    assert lines[1].endswith("most likely Tue 1 pm to 7 pm (21%)") and "Mon 7 pm" not in lines[1]
    assert lines[2].endswith("most likely Wed 1 pm to 7 pm (28%)")
    listed = quick.brief(result, names, population, top=1, days=1, periods=periods, when=True).splitlines()
    assert listed[2] == "      Mon 7 pm to 1 am       over" and listed[3] == "      Tue 1 am to 7 am       over"
    assert listed[5] == "      Tue 1 pm to 7 pm       rain  21%"
    late = dict(periods, issued=day0 + 24)                               # the whole first day is over
    assert "most likely" not in quick.brief(result, names, population, top=1, days=1, periods=late)
    assert quick.brief(result, names, population, top=1, days=1, periods=dict(periods, issued=day0 - 6)).splitlines()[1] \
        .endswith("most likely Mon 7 pm to 1 am (27%)")                   # made the evening before: nothing is over


def test_usual_timing_is_read_at_places_by_month_and_fetched_for_a_blend_that_lacks_it(tmp_path):
    from pluvium.util import hour_of

    # a 2 x 2 grid: in November the last period at the first cell, in December the first period everywhere
    given = np.full((12, 4, 2, 2), 0.25, np.float32)
    given[10, :, 0, 0] = [0.0, 0.0, 0.1, 0.9]
    given[11] = np.array([0.9, 0.1, 0.0, 0.0], np.float32)[:, None, None]
    folder = tmp_path / "blend"
    folder.mkdir()
    np.savez_compressed(folder / "timing.npz", lat=np.array([18.0, 18.25]), lon=np.array([-77.5, -77.25]), given=given)
    day0 = hour_of("2026-11-29T00")                                    # two days of November, then December
    usual = quick.usual_at(tmp_path, [18.02, 18.24], [-77.49, -77.26], day0, 3)
    assert usual.shape == (3, 4, 2)
    assert np.allclose(usual[0, :, 0], [0, 0, 0.1, 0.9]) and np.allclose(usual[0, :, 1], 0.25)
    assert np.allclose(usual[2, :, 0], [0.9, 0.1, 0, 0])               # the third day is in December
    # a blend downloaded before its repository had the file gets it on the next run, and only if it is the real thing
    hub = tmp_path / "hub" / "org" / "repo" / "blends" / "testland"
    hub.mkdir(parents=True)
    url = lambda repo, name, file: str(tmp_path / "hub" / repo / "blends" / name / file)     # noqa: E731
    later = tmp_path / "later"
    (later / "blend").mkdir(parents=True)
    assert quick.top_up("org/repo", "testland", later, url=url) is False                      # nothing published yet
    np.savez_compressed(hub / "timing.npz", lat=np.zeros(2), lon=np.zeros(2), given=np.zeros((3, 3)))
    assert quick.top_up("org/repo", "testland", later, url=url) is False                      # not the right shape
    (hub / "timing.npz").write_bytes((folder / "timing.npz").read_bytes())
    assert quick.top_up("org/repo", "testland", later, url=url) is True
    assert (later / "blend" / "timing.npz").read_bytes() == (folder / "timing.npz").read_bytes()
    assert "timing.npz" in quick.BLEND_FILES and "timing.npz" not in quick.REQUIRED


def test_now_command_prints_what_it_used_and_the_forecast(model, tmp_path, monkeypatch, capsys):
    from pluvium import cli

    day = {"date": "2026-11-15", "p_ge_1": np.array([0.2, 0.71]), "p_ge_25": np.array([0.0, 0.12]),
           "mean_mm": np.array([0.4, 6.2]), "alert": np.array([0, 1]), "beyond_record": np.array([False, False])}
    got = {"root": tmp_path, "table": tmp_path / "blend_2026-11-15.csv", "names": ["Hamlet", "Capital"],
           "population": np.array([120, 900_000]), "model": model, "how": "downloaded",
           "result": {"day0": "2026-11-15", "runs": {"ens": "2026-11-15T00"}, "days": [day]}}
    seen = {}
    monkeypatch.setattr(quick, "forecast", lambda country, root, repo, **kw: seen.update(country=country, repo=repo, **kw) or got)
    cli.main(["now", "--country", "TL", "--top", "1", "--days", "1", "--learn"])
    out = capsys.readouterr().out
    assert seen["country"] == "TL" and seen["repo"] == "CorxLabs/Pluvium-500" and seen["learn"] is True and seen["train"] is True
    assert "testland: 14-day rain forecast from 2026-11-15 for 2 places, with the blend published in CorxLabs/Pluvium-500" in out
    assert "runs read: ens 2026-11-15T00Z" in out and "not available right now, and left out: gfs" in out
    assert "record storms:" in out and "Capital" in out and "Hamlet" not in out and "[watch]" in out
    assert "heavy-rain alerts: 1 places reach watch" in out and "blend_2026-11-15.csv" in out


def test_command_line_explains_a_period_without_rain_and_lists_the_upstream_models(tmp_path, monkeypatch, capsys):
    from pluvium import cli
    from pluvium.util import PluviumError

    AREA.save(tmp_path / "area.json")
    seen = {}

    def train(area, root, start, end, handover="auto"):
        seen.update(start=start, end=end)
        raise PluviumError("The satellite rain record (NOAA CMORPH) has 0 of the 97 days")

    monkeypatch.setattr(FrontierBlend, "train", classmethod(lambda cls, *a, **k: train(*a, **k)))
    monkeypatch.setattr(quick, "recent_year", lambda: ("2025-05-16T00", "2026-05-16T00"))
    with pytest.raises(SystemExit, match="pluvium: The satellite rain record"):          # a message, not a traceback
        cli.main(["blend", "train", "--root", str(tmp_path)])
    assert seen == {"start": "2025-05-16T00", "end": "2026-05-16T00"}                    # the period was chosen for us
    assert "the most recent year NOAA's rain record covers" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["blend", "train", "--root", str(tmp_path), "--start", "2026-07-10T00", "--end", "2026-10-15T00"])
    assert seen == {"start": "2026-07-10T00", "end": "2026-10-15T00"}                    # a period given is used as given
    cli.main(["upstream", "list"])
    out = capsys.readouterr().out
    assert "noaa-graphcastgfs" in out and "CC0 1.0 and CC BY 4.0" in out and "(in use today)" in out
    assert "it does not run them" in out


def test_blend_card_says_what_the_staged_storms_decided(model):
    text = quick.blend_card(model)
    assert "pluvium now --country TL" in text and "kept here" in text
    saved = json.loads(json.dumps(model.meta))
    try:
        # as training records it: the days themselves, and the score without and with the hand-over
        model.meta["handover"] = {"keep": True, "mode": "auto", "staged": {
            "storms": ["2025-10-28", "2025-10-18"], "gain": 0.4936, "calibrated": 0.2593, "handed": 0.1313, "amount": 50.0}}
        kept = quick.blend_card(model)
        model.meta["handover"] = {"keep": False, "mode": "auto", "staged": {"storms": 4, "gain": -0.015}}
        off = quick.blend_card(model)
    finally:
        model.meta.clear()
        model.meta.update(saved)
    assert "kept here" in kept and "the 2 wettest days (2025-10-28, 2025-10-18)" in kept
    assert "49.4% lower, 0.131 against 0.259. Lower is better." in kept and "[" not in kept
    assert "switched off here" in off and "the 4 wettest days were held back" in off and "1.5% higher." in off
    assert "°N" in kept and (" E." not in kept)                                 # a letter for the hemisphere, no signed "E"
