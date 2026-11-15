# Copyright 2026 Nathan. Apache-2.0.
"""The training program's settings: the years it trains on and holds out follow what the reanalysis archive really
holds, and the GPU memory settings are read from the environment and checked. No network, no GPU."""

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
