# Copyright 2026 Nathan. Apache-2.0.
"""
Pressure levels through the dataset, and the fallbacks that keep a run honest when something is absent.

The bug these are named after: ERA5Window had no concept of levels, so a cell that asked for 89
channels got 11 and printed 89. Nothing raised. That is the same shape as every other failure in this
codebase -- a silent wrong answer -- and the fix is that one function decides what a channel *is* and
the same function produces it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


xr = pytest.importorskip("xarray")

from naturev1.era5 import ASINH_SCALE, LOG_SCALE, Normalizer, expand_variables, read_block  # noqa: E402
from naturev1.fallback import (  # noqa: E402
    Plan,
    resolve_batch,
    resolve_precision,
    resolve_staging,
    resolve_variables,
    with_retry,
)


LAT = np.linspace(90.0, -90.0, 13)
LON = np.linspace(0.0, 360.0, 24, endpoint=False)
LEVELS = np.array([250, 500, 850, 1000])


@pytest.fixture
def store():
    """A store with all three shapes of variable: time-varying, static, and pressure-level."""
    steps = 6
    lat_mesh, lon_mesh = np.meshgrid(LAT, LON, indexing="ij")

    def dynamic(base):
        field = base + np.arange(steps)[:, None, None] * 0.5
        return ("time", "longitude", "latitude"), field.transpose(0, 2, 1).astype(np.float32)

    def static(base):
        return ("longitude", "latitude"), base.T.astype(np.float32)

    def levelled(base):
        field = np.stack([base + level * 0.01 for level in LEVELS], -1)
        field = np.repeat(field[None], steps, 0)
        return ("time", "longitude", "latitude", "level"), field.transpose(0, 2, 1, 3).astype(np.float32)

    temperature = 300.0 - 80.0 * np.sin(np.radians(np.abs(lat_mesh)))
    return xr.Dataset(
        {
            "2m_temperature": dynamic(temperature),
            "land_sea_mask": static((lon_mesh > 180).astype(float)),
            "temperature": levelled(temperature),
            "vertical_velocity": levelled(np.zeros_like(temperature)),
        },
        coords={
            "time": np.arange("2000-01-01", "2000-01-03", np.timedelta64(8, "h"), dtype="datetime64[ns]")[:steps],
            "latitude": LAT, "longitude": LON, "level": LEVELS,
        },
    )


ALL = ("2m_temperature", "land_sea_mask", "temperature", "vertical_velocity")


# --------------------------------------------------------------------------------------------------
# One variable is not one channel
# --------------------------------------------------------------------------------------------------

def test_expand_variables_counts_levels(store):
    names = expand_variables(store, ALL, [int(x) for x in LEVELS])
    assert len(names) == 2 + 2 * len(LEVELS)
    assert names[:2] == ["2m_temperature", "land_sea_mask"]
    assert "temperature@500" in names and "vertical_velocity@1000" in names


def test_expand_variables_honours_a_level_subset(store):
    names = expand_variables(store, ("temperature",), [500, 850])
    assert names == ["temperature@500", "temperature@850"]


def test_expand_variables_skips_what_the_store_lacks(store):
    assert expand_variables(store, ("2m_temperature", "not_a_variable"), None) == ["2m_temperature"]


def test_read_block_matches_expand_variables(store):
    """
    The invariant the whole fix rests on: the thing that says what a channel is, and the thing that
    produces it, are the same function. When they drifted, a dataset came back with 11 channels while
    the cell printed 89.
    """
    levels = [int(x) for x in LEVELS]
    names = expand_variables(store, ALL, levels)
    block = read_block(store, ALL, slice(0, 3), levels)
    assert block.shape == (3, len(LAT), len(LON), len(names))


def test_static_fields_are_broadcast_not_dropped(store):
    """Land-sea mask has no time axis, and it is the only way the model can know a coastline is there."""
    names = expand_variables(store, ALL, [int(x) for x in LEVELS])
    block = read_block(store, ALL, slice(0, 3), [int(x) for x in LEVELS])
    mask = block[:, :, :, names.index("land_sea_mask")]
    assert np.allclose(mask[0], mask[1]) and np.allclose(mask[0], mask[2])

    moving = block[:, :, :, names.index("2m_temperature")]
    assert not np.allclose(moving[0], moving[1]), "a time-varying field must still vary"


def test_read_block_refuses_an_empty_variable_list(store):
    with pytest.raises(ValueError, match="none of"):
        read_block(store, ("nothing_here",), slice(0, 2))


def test_normalizer_keys_on_flat_channels(store):
    """
    Statistics per variable would be one number describing thirteen distributions: 50 hPa is 210 K and
    1000 hPa is 285 K, and a shared mean fits neither.
    """
    levels = [int(x) for x in LEVELS]
    normalizer = Normalizer.fit(store, ALL, levels=levels)
    assert normalizer.variables == expand_variables(store, ALL, levels)
    assert len(normalizer.mean) == len(normalizer.variables)

    warm = normalizer.variables.index("temperature@1000")
    cold = normalizer.variables.index("temperature@250")
    assert normalizer.mean[warm] != normalizer.mean[cold], "levels must get their own statistics"


# --------------------------------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------------------------------

def test_signed_heavy_tails_use_asinh_and_invert_exactly():
    """
    Vertical velocity sat at 15-27 sigma on every level. log1p cannot be used -- omega is signed, and
    negative omega is rising air, which is where weather comes from. Clamping it away removes the
    thing worth predicting.
    """
    normalizer = Normalizer(["vertical_velocity@500"], np.array([0.0], np.float32),
                            np.array([1.0], np.float32), np.array([0.0], np.float32))
    raw = np.array([[-5.0], [-0.1], [0.0], [0.01], [3.0]], np.float32)
    encoded = normalizer.transform(raw)

    # What asinh compresses is the dynamic RANGE, not the magnitude -- it divides by the scale first,
    # so small values expand and large ones flatten. That is the point: the tail stops dominating the
    # variance while small motions stay resolvable.
    raw_range = float(np.abs(raw).max() / np.abs(raw[raw != 0]).min())
    coded_range = float(np.abs(encoded).max() / np.abs(encoded[encoded != 0]).min())
    assert coded_range < raw_range / 10, f"range {raw_range:.0f} -> {coded_range:.0f} is not enough"
    assert np.sign(encoded[0, 0]) == -1 and np.sign(encoded[4, 0]) == 1, "sign must survive"
    for value, code in zip(raw.ravel(), encoded.ravel()):
        back = float(normalizer.decode(torch.tensor(code), "vertical_velocity@500"))
        assert back == pytest.approx(float(value), abs=1e-4)


def test_transforms_match_on_the_base_name_not_the_level():
    """Otherwise every level of every heavy-tailed field would have to be listed by hand."""
    from naturev1.era5 import _base_name

    assert _base_name("vertical_velocity@500") == "vertical_velocity"
    assert _base_name("2m_temperature") == "2m_temperature"
    assert "vertical_velocity" in ASINH_SCALE and "total_precipitation_6hr" in LOG_SCALE


# --------------------------------------------------------------------------------------------------
# Fallbacks
# --------------------------------------------------------------------------------------------------

def test_missing_variables_are_named_not_silently_dropped(store):
    """Silently dropping is how a run trains on 11 channels while the log says 89."""
    kept, levels, plan = resolve_variables(store, ("2m_temperature", "ozone", "cloud_ice"), None)
    assert kept == ["2m_temperature"]
    assert plan.choices and "ozone" in plan.choices[0].because and "cloud_ice" in plan.choices[0].because


def test_missing_levels_are_reported(store):
    kept, levels, plan = resolve_variables(store, ("temperature",), [500, 700, 850])
    assert levels == [500, 850]
    assert any("700" in c.because for c in plan.choices)


def test_no_usable_variables_raises(store):
    with pytest.raises(ValueError, match="nothing to train on"):
        resolve_variables(store, ("ozone",), None)


def test_staging_shrinks_to_the_disk_rather_than_filling_it():
    plan = Plan()
    years, streaming, plan = resolve_staging(20, channels=89, points=29_040, path="/", plan=plan)
    assert years <= 20 and isinstance(streaming, bool)
    if years < 20:
        assert plan.choices, "a reduction must be reported"


def test_staging_falls_back_to_streaming_when_nothing_fits(monkeypatch):
    """Slower by about 100x per window, but it always works -- and it says so."""
    class _Tiny:
        f_bavail, f_frsize = 1, 1024          # 1 KB free
    monkeypatch.setattr("os.statvfs", lambda path: _Tiny())

    years, streaming, plan = resolve_staging(5, channels=89, points=29_040, path="/tmp")
    assert streaming and years == 0
    assert any("streaming" in c.took for c in plan.choices)


def test_precision_falls_back_on_cpu():
    precision, plan = resolve_precision("bf16")
    if not torch.cuda.is_available():
        assert precision == "fp32"
        assert any("no CUDA" in c.because for c in plan.choices)


def test_batch_falls_back_to_one_on_cpu():
    batch, plan = resolve_batch(32, points=29_040, channels=89, layers=17)
    if not torch.cuda.is_available():
        assert batch == 1


def test_retry_gives_up_loudly_rather_than_silently():
    attempts = []

    def flaky():
        attempts.append(1)
        raise ConnectionError("dropped")

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        with_retry(flaky, attempts=3, delay=0.001, what="chunk")
    assert len(attempts) == 3


def test_retry_succeeds_after_a_transient_failure():
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise TimeoutError("slow")
        return "ok"

    assert with_retry(flaky, attempts=4, delay=0.001) == "ok"
    assert state["n"] == 3


def test_a_plan_with_no_substitutions_says_so():
    assert "no substitutions" in Plan().summary()
