# Copyright 2026 Nathan. Apache-2.0.
"""The training program's settings: the years it trains on and holds out follow what the reanalysis archive really
holds, and the GPU memory settings are read from the environment and checked. Needs neither a network nor a GPU."""

import numpy as np
import pytest

from pluvium import program
from pluvium.program import Program, ProgramConfig, choose_years
from pluvium.sources import era5
from pluvium.util import hour_of


class Store:
    """Stands in for the hourly ERA5 store: only its time axis is asked for."""

    def __init__(self, first: str, last: str) -> None:
        self.first, self.last = hour_of(first), hour_of(last)

    def hours(self) -> np.ndarray:
        return np.arange(self.first, self.last + 1, dtype=np.int64)


def test_whole_years_are_read_from_the_archive_itself():
    # the real store on 15 November 2026: named 1959-2022, and ending on 31 December 2021
    assert era5.hourly_years(Store("2019-01-01T00", "2021-12-31T23")) == (2019, 2021)
    assert era5.hourly_years(Store("2019-01-01T00", "2022-06-30T23")) == (2019, 2021)      # a year cut short is not whole
    assert era5.hourly_years(Store("2019-03-01T00", "2022-12-31T23")) == (2020, 2022)      # nor one that starts late

    class Unreachable:
        def hours(self):
            raise OSError("no network")

    assert era5.hourly_years(Unreachable()) == (1959, 2021)                              # what the code remembers
    assert era5.HOURLY_LAST == hour_of("2021-12-31T23")


def test_years_asked_for_are_fitted_to_the_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(era5, "hourly_years", lambda store=None: (1959, 2021))

    def years(**settings):
        made = Program(ProgramConfig(root=str(tmp_path / str(len(settings))), hub_repo=None, **settings))
        made.check_years()
        return made.years, made.cfg.holdout_year

    assert ProgramConfig().holdout_year == 2021 and choose_years(24) == (2015, 2016, 2017, 2018, 2019, 2020)
    assert years() == ((2015, 2016, 2017, 2018, 2019, 2020), 2021)
    # 2022 was the default held-out year until 0.4.3: the archive does not have it, and the run stopped while
    # staging it, after every training year had been downloaded
    assert years(holdout_year=2022) == ((2015, 2016, 2017, 2018, 2019, 2020), 2021)
    assert years(train_years=(2019, 2020, 2021, 2022), holdout_year=2022) == ((2019, 2020), 2021)
    assert years(train_years=(2018, 2019), holdout_year=2020) == ((2018, 2019), 2020)        # a valid choice is kept
    with pytest.raises(ValueError, match="no training year is left"):
        years(train_years=(2021,))
    monkeypatch.setattr(era5, "hourly_years", lambda store=None: (1959, 2023))               # the archive grows
    assert years(holdout_year=2023, train_years=(2021, 2022)) == ((2021, 2022), 2023)


def test_memory_settings_come_from_the_environment_and_are_checked(tmp_path, monkeypatch):
    for name in ("GPU_HOURS", "PLUVIUM_ROOT", "HF_REPO", "HF_TOKEN", "PLUVIUM_SIZE", "PLUVIUM_COUNTRIES", "PLUVIUM_BOX",
                 "PLUVIUM_VRAM", "PLUVIUM_VRAM_SHARE", "PLUVIUM_BATCH"):
        monkeypatch.delenv(name, raising=False)
    cfg = ProgramConfig.from_env(root=str(tmp_path), hub_repo=None)
    assert (cfg.vram, cfg.vram_share, cfg.batch_size) == ("fill", 0.95, 0)                   # fill the card by default
    monkeypatch.setenv("PLUVIUM_VRAM_SHARE", "0.9")
    monkeypatch.setenv("PLUVIUM_VRAM", "fastest")
    monkeypatch.setenv("PLUVIUM_BATCH", "6")
    cfg = ProgramConfig.from_env(root=str(tmp_path), hub_repo=None)
    cfg.hub_repo = None
    assert (cfg.vram, cfg.vram_share, cfg.batch_size) == ("fastest", 0.9, 6)
    train = Program(cfg).train_cfg(1.0)
    assert (train.memory, train.memory_share, train.batch_size) == ("fastest", 0.9, 6)
    for wrong in ({"vram": "all"}, {"vram_share": 1.0}, {"vram_share": 0.1}):
        with pytest.raises(ValueError):
            Program(ProgramConfig(root=str(tmp_path), hub_repo=None, **wrong)).train_cfg(1.0)
    source = open(program.__file__).read()                                                  # run as a module, it starts
    assert callable(program.main) and source.rstrip().endswith('if __name__ == "__main__":\n    main()')


def test_staged_data_go_up_in_a_few_commits_and_only_what_the_hub_lacks(tmp_path):
    """The Hub takes at most 128 commits an hour in a repository. One commit per staged file (until 0.5.2) used that
    up with the data alone, and most of them were refused with HTTP 429."""
    import hashlib
    from types import SimpleNamespace

    from pluvium.program import FILES_PER_COMMIT, mirror_files

    files = []
    for k in range(2 * FILES_PER_COMMIT + 5):
        path = tmp_path / f"f{k}.npy"
        path.write_bytes(bytes([k % 256]) * 100)
        files.append((path, f"data/f{k}.npy"))
    same = files[0][0].read_bytes()
    old = tmp_path / "old.npy"
    old.write_bytes(b"older content")

    class Api:
        def __init__(self, fail=False):
            self.commits, self.fail = [], fail

        def get_paths_info(self, repo, paths, repo_type="model"):
            if self.fail:
                raise RuntimeError("429 Too Many Requests")
            return [SimpleNamespace(path="data/f0.npy", lfs={"sha256": hashlib.sha256(same).hexdigest()}),   # held
                    SimpleNamespace(path="data/f1.npy", lfs={"sha256": hashlib.sha256(b"older").hexdigest()})]  # stale

        def create_commit(self, repo, repo_type, commit_message, operations):
            self.commits.append([op.path_in_repo for op in operations])

    mirror = SimpleNamespace(can_push=True, repo_id="org/repo", token="t")
    api = Api()
    assert mirror_files(mirror, files + [(tmp_path / "missing.npy", "data/missing.npy")], "data", api=api) == len(files) - 1
    assert [len(c) for c in api.commits] == [FILES_PER_COMMIT, FILES_PER_COMMIT, 4]        # three commits, not 125
    pushed = [p for c in api.commits for p in c]
    assert "data/f0.npy" not in pushed and "data/f1.npy" in pushed and "data/missing.npy" not in pushed
    assert mirror_files(mirror, files, "data", api=Api(fail=True)) == 0                    # refused: said, not raised
    assert mirror_files(SimpleNamespace(can_push=False), files, "data", api=Api()) == 0    # no token: nothing to do


def test_a_year_with_only_one_of_its_two_rain_files_is_staged_again(tmp_path, monkeypatch):
    """The Hub once held ``targets_2016.npy`` without ``mesh_2016.npy`` (the uploads of an earlier run were refused
    half way). The stager asked only the targets file, took 2016 as done, and a training run on a 96 GB card stopped
    when it first needed the file that was never made."""
    import pluvium.sources.cmorph as C
    from pluvium.util import year_hours

    class Points:
        def __init__(self, n):
            self.n, self.extra = n, {"row": np.zeros(n, int), "col": np.zeros(n, int)}

        def __len__(self):
            return self.n

    targets, mesh, fetched = Points(3), Points(5), []
    for name in ("cmorph_window", "mesh_operator", "default_half_widths", "_init"):
        monkeypatch.setattr(C, name, lambda *a, **k: None)
    monkeypatch.setattr(C, "_one_hour", lambda hour: (fetched.append(hour) or hour, np.full(3, 7, np.uint16), np.full(5, 9, np.uint16)))
    first, _ = year_hours(2016)
    period = (first, first + 48)
    folder = tmp_path / "cmorph"
    C.stage_cmorph(tmp_path, targets, mesh, period=period, workers=1, verbose=False)
    assert len(fetched) == 48 and (folder / "targets_2016.npy").exists() and (folder / "mesh_2016.npy").exists()
    fetched.clear()
    C.stage_cmorph(tmp_path, targets, mesh, period=period, workers=1, verbose=False)
    assert fetched == []                                                      # a whole year is not fetched twice
    (folder / "mesh_2016.npy").unlink()                                       # half the year, as it came from the Hub
    cache = C.stage_cmorph(tmp_path, targets, mesh, period=period, workers=1, verbose=False)
    assert len(fetched) == 48 and (folder / "mesh_2016.npy").exists()
    assert cache.mesh(np.arange(first, first + 3)).shape == (3, 5) and np.isfinite(cache.mesh(np.arange(first, first + 3))).all()


def test_the_background_fetch_does_not_keep_a_finished_run_alive(tmp_path):
    import multiprocessing
    import time

    prog = Program(ProgramConfig(root=str(tmp_path), hub_repo=None))
    prog.stop_prefetch()                                                      # nothing running: nothing to do
    prog._prefetch = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(120,))
    prog._prefetch.start()
    began = time.time()
    prog.stop_prefetch()
    assert not prog._prefetch.is_alive() and time.time() - began < 30


def test_a_molab_session_is_planned_to_be_published_before_molab_stops_it(tmp_path, monkeypatch):
    """molab stops a notebook 12 hours after it starts. The first full run there trained for 11.6 of its 24 hours and
    was stopped before it was scored and published. A run there now plans to be done before the session ends."""
    import time

    from pluvium import program as P

    for name in ("PLUVIUM_SESSION_HOURS", "PLUVIUM_SESSION_START", "PLUVIUM_SESSION_WHY"):
        monkeypatch.delenv(name, raising=False)
    molab = "/tmp/uv-venv/lib/python3.13/site-packages/pluvium/program.py"            # as the tracebacks there show
    assert P.on_molab({}, places=[molab], commands=[])
    assert P.on_molab({}, places=["/usr/lib"], commands=["/usr/bin/python -m marimo edit notebook.py"])
    assert P.on_molab({"MARIMO_MANAGE_SCRIPT_METADATA": "true"}, places=["/usr/lib"], commands=[])
    assert not P.on_molab({"HOME": "/root"}, places=["/usr/lib"], commands=["bash", "python -m pluvium.program"])
    assert P.session_limit({"PLUVIUM_SESSION_HOURS": "6"}) == (6.0, "PLUVIUM_SESSION_HOURS")
    assert P.session_limit({"PLUVIUM_SESSION_HOURS": "0"})[0] is None                  # 0: no end, even on molab
    began = time.time() - 40 * 60
    assert P.session_start(12, {"PLUVIUM_SESSION_START": str(began)}) == began
    found = P.session_start(12, {})                                     # the earliest process of this session
    assert time.time() - 12 * 3600 < found <= time.time()

    monkeypatch.setattr(P, "on_molab", lambda env=None, places=None, commands=None: True)
    assert P.session_limit({}) == (12.0, "molab stops a notebook after 12 hours")
    # the notebook started 40 minutes before the program: published 11.5 hours after that, not 24 hours after
    monkeypatch.setenv("PLUVIUM_SESSION_START", str(began))
    prog = Program(ProgramConfig(root=str(tmp_path), hub_repo=None, gpu_hours=24))
    assert prog.session is not None and abs(prog.deadline - (began + 11.5 * 3600)) < 1
    assert 10.7 < prog.left() < 10.9 and prog.reserve() == 1.5
    # no end known: 24 hours from the program's start, as before
    monkeypatch.setattr(P, "on_molab", lambda env=None, places=None, commands=None: False)
    prog = Program(ProgramConfig(root=str(tmp_path), hub_repo=None, gpu_hours=24))
    assert prog.session is None and 23.9 < prog.left() <= 24 and prog.reserve() == 1.0


def test_the_supervisor_finds_the_session_once_and_every_try_gets_it(tmp_path, monkeypatch):
    import sys
    import time

    from pluvium.program import supervise

    seen = tmp_path / "env.txt"
    child = [sys.executable, "-c", "import os, sys; open(sys.argv[1], 'a').write(os.environ['PLUVIUM_SESSION_HOURS'] "
             "+ ' ' + os.environ['PLUVIUM_SESSION_START'] + '\\n'); sys.exit(0 if os.environ['PLUVIUM_ATTEMPT'] == '2' "
             "else 1)", str(seen)]
    monkeypatch.delenv("PLUVIUM_SESSION_START", raising=False)
    monkeypatch.setenv("PLUVIUM_SESSION_HOURS", "12")
    assert supervise([], 0.95, command=child) == 0
    first, second = seen.read_text().splitlines()
    assert first == second and first.split()[0] == "12" and float(first.split()[1]) <= time.time()
