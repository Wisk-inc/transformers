# Copyright 2026 Nathan. Apache-2.0.
"""
0.12: the rebuilt satellite path, end to end on synthetic data written in the real formats.

GridSat files are written as GridSat writes them (packed int16, the misleading ``valid_range``), staged
month by month, read back as storm loops and global frames, and both networks are trained for a few
steps through the same engine the 8-GPU run uses. Alongside: the hemisphere frame is an exact mirror,
the new paths are exact no-ops until trained, old checkpoints load unchanged, RI 2.0 probabilities are
ordered, cloud motion recovers a known shift, and every checkpoint is queued for the Hub.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

netCDF4 = pytest.importorskip("netCDF4")

from ihelix import fibonacci_sphere  # noqa: E402

from naturev1.besttrack import Track  # noqa: E402
from naturev1.era5 import Normalizer  # noqa: E402
from naturev1.ibtracs import storm_label_table  # noqa: E402
from naturev1.model import TRACK_MEMBERS, TRACKED_HEADS, NatureConfig, NatureV1  # noqa: E402
from naturev1.satdata import GlobalFrames, RetrievalDataset, StormPatches, StormSequenceDataset, recentre  # noqa: E402
from naturev1.satnet import apply_retrieval, cloud_motion, ri_threshold_list  # noqa: E402
from naturev1.satstage import (  # noqa: E402
    GLOBAL_CHANNELS, decode_global, nadir_view, quality_control, read_gridsat, stage_month,
)
from naturev1.state import StateStepper, channel_roles  # noqa: E402

START = np.datetime64("2005-08-10T00", "h")
STEPS = 17                                                     # two days, three-hourly


def storm_position(name: str, k: int) -> tuple[float, float]:
    if name == "north":
        return 15.0 + 0.25 * k, -50.0 - 0.4 * k                 # north-west
    return -15.0 - 0.25 * k, 150.0 - 0.4 * k                    # south-west, southern hemisphere


def brightness(lat, lon, k):
    """A warm surface and two cold, round storms -- smooth, as real cloud is at this scale."""
    field = np.full(np.broadcast(lat, lon).shape, 295.0)
    for name in ("north", "south"):
        clat, clon = storm_position(name, k)
        d2 = (lat - clat) ** 2 + (((lon - clon + 180) % 360) - 180) ** 2
        field = field - 95.0 * np.exp(-d2 / (2 * 1.5 ** 2))
    return field


def write_gridsat(path, k):
    lat = np.arange(-70.0, 70.0, 0.5) + 0.25
    lon = np.arange(-180.0, 180.0, 0.5) + 0.25
    ir = brightness(lat[:, None], lon[None, :], k)
    wv = ir - 15.0
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("time", 1)
        d.createDimension("lat", lat.size)
        d.createDimension("lon", lon.size)
        d.createDimension("Ngeo", 3)
        d.createDimension("StrLen", 50)
        d.createVariable("lat", "f4", ("lat",))[:] = lat
        d.createVariable("lon", "f4", ("lon",))[:] = lon
        for name, values in (("irwin_cdr", ir), ("irwvp", wv)):
            v = d.createVariable(name, "i2", ("time", "lat", "lon"), fill_value=np.int16(-31999))
            v.scale_factor, v.add_offset = np.float32(0.01), np.float32(200.0)
            v.valid_range = np.array([140.0, 375.0], np.float32)          # in kelvin, on an int16: the trap
            v.set_auto_maskandscale(False)
            v[0] = np.rint((values - 200.0) / 0.01).astype(np.int16)
        d.createVariable("satlon", "f4", ("time", "Ngeo"))[0] = np.array([-75.0, 140.7, 0.0], np.float32)
        names = d.createVariable("satname", "S1", ("time", "Ngeo", "StrLen"))
        for i, text in enumerate(("GOES-16", "Himawari 8", "Meteosat-11")):
            names[0, i] = netCDF4.stringtochar(np.array([text.ljust(50)], "S50"))[0]


def track(name, sid):
    times = START + np.arange(STEPS + 8) * np.timedelta64(3, "h")
    positions = [storm_position(name, k) for k in range(STEPS + 8)]
    n = len(times)
    wind = np.linspace(40, 120, n).astype(np.float32)
    return Track(storm_id=sid, name=name.upper(), time=times.astype("datetime64[s]"),
                 latitude=np.array([p[0] for p in positions], np.float32),
                 longitude=np.array([p[1] for p in positions], np.float32),
                 max_wind_kt=wind, min_pressure_hpa=(1010 - wind * 0.8).astype(np.float32),
                 status=np.array(["HU"] * n), rmw_nmi=np.full(n, 20.0, np.float32),
                 wind_radii_nmi=np.full((n, 3, 4), 50.0, np.float32), record=np.array([""] * n),
                 basin="NA" if name == "north" else "WP", interpolated=np.zeros(n, bool),
                 wind_source=np.zeros(n, np.int8), distance_to_land_km=np.full(n, 500.0, np.float32))


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    root = tmp_path_factory.mktemp("staged")
    raw = tmp_path_factory.mktemp("raw")
    for k in range(STEPS):
        stamp = (START + np.timedelta64(3 * k, "h")).astype(object)
        write_gridsat(raw / f"GRIDSAT-B1.{stamp:%Y.%m.%d.%H}.v02r01.nc", k)
    table = storm_label_table([track("north", "2005222N15310"), track("south", "2005222S15150")])
    summary = stage_month(root, 2005, 8, table, resolution=2.5, raw_dir=raw, keep_raw=True, download=False)
    return root, table, summary


def test_gridsat_is_decoded_by_hand_not_masked_away(tmp_path):
    path = tmp_path / "g.nc"
    write_gridsat(path, 0)
    fields = read_gridsat(path)
    assert np.isfinite(fields["ir"]).mean() == 1.0, "netCDF4's valid_range would have masked nearly all of it"
    assert abs(float(np.nanmax(fields["ir"])) - 295.0) < 0.02 and fields["satname"][1] == "Himawari 8"


def test_quality_control_removes_glitches_but_keeps_cloud_edges():
    ir = np.full((20, 20), 290.0, np.float32)
    ir[5, 5] = 170.0                                       # a bit error: far below all four neighbours
    ir[10:, 10:] = 210.0                                   # an anvil with a sharp corner at (10, 10)
    wv = ir - 20.0
    wv[15, 3] = ir[15, 3] + 25.0                           # water vapour warmer than the window: impossible
    ir_ok, wv_ok, report = quality_control(ir, wv)
    assert np.isnan(ir_ok[5, 5]) and report["spikes_removed"] == 1
    assert np.isfinite(ir_ok[10:, 10:]).all(), "the anvil's edge and corner are real"
    assert np.isnan(wv_ok[15, 3])


def test_view_angle_and_satellite_come_from_geometry():
    zenith, nadir = nadir_view(np.array([0.0, 0.0, 0.0]), np.array([-75.0, 140.7, 100.0]), np.array([-75.0, 140.7]))
    assert zenith[0, 0] < 1e-6 and nadir[0, 0] == 0 and nadir[0, 1] == 1
    assert 30 < zenith[0, 2] < 70 and nadir[0, 2] == 1, "100 E is seen by the satellite at 140.7 E"


def test_a_month_stages_frames_and_storm_patches(staged):
    root, table, summary = staged
    assert summary["observed_times"] == STEPS and summary["storm_patches_ok"] == 2 * STEPS < summary["storm_fixes"]
    frames = GlobalFrames(root)
    first = frames.frame(int(START.astype("datetime64[s]").astype(np.int64)))
    assert first.shape == (len(GLOBAL_CHANNELS), 56, 144) and first.dtype == np.uint8
    fields = decode_global(first)
    # A 2.5-degree cell averages the storm's 200 K core away to ~250 K; the per-cell minimum keeps it.
    assert np.nanmin(fields["ir_min"]) < 215 and np.nanmin(fields["ir_mean"]) > np.nanmin(fields["ir_min"]) + 20
    assert np.nanmax(fields["ir_coverage"]) == pytest.approx(1.0)
    # GOES, Meteosat and Himawari each own a sector; 0 marks pixels beyond every satellite's limb.
    assert {1, 2, 3} <= set(np.unique(fields["satellite_family"]).tolist()) <= {0, 1, 2, 3}
    assert frames.frame(int((START + np.timedelta64(30 * 24, "h")).astype("datetime64[s]").astype(np.int64))) is None
    # Resume: a staged month is not redone.
    again = stage_month(root, 2005, 8, table, resolution=2.5, raw_dir=root / "nothing", download=False)
    assert again["observed_times"] == STEPS


def test_recentre_moves_the_storm_by_the_right_number_of_pixels():
    patch = np.zeros((1, 320, 320), np.uint8)
    patch[0, 160, 160] = 255
    moved, inside = recentre(patch, (10.0, 20.0), (10.0 + 0.07 * 5, 20.0 - 0.07 * 3), 256)
    assert inside == 1.0
    row, col = np.unravel_index(moved[0].argmax(), moved[0].shape)
    assert (row, col) == (128 - 5, 128 + 3), "the storm sits south-east of a guess placed north-west of it"


def test_storm_loop_is_cut_around_the_first_guess_and_the_storm_moves_across_it(staged):
    root, table, _ = staged
    data = StormSequenceDataset(table, StormPatches(root), pixels=128, jitter_km=120.0, train=False)
    assert len(data) > 0
    item = data[len(data) // 2]
    assert item["codes"].shape == (4, 2, 128, 128) and bool(item["available"][0])
    from naturev1.stormsat import decode_bt

    now = decode_bt(item["codes"][0, 0].numpy())
    row, col = np.unravel_index(np.nanargmin(now), now.shape)
    north_km = (row - 63.5) * 0.07 * 111.195
    lat = float(item["latitude"])
    east_km = (col - 63.5) * 0.07 * 111.195 * math.cos(math.radians(lat))
    offset = item["centre_offset_km"].numpy()
    assert abs(north_km - offset[0]) < 60 and abs(east_km - offset[1]) < 60, "the cold core is where the label says"


def tiny_config(**over) -> NatureConfig:
    settings = dict(analysis_channels=8, state_channels=3, hidden_size=32, num_layers=2, num_heads=2,
                    num_kv_heads=1, head_dim=16, intermediate_size=64, latent_points=96, latent_neighbours=12,
                    latent_cluster=16, landmark_dim=16, index_branching=2, index_beam_width=2, index_topk=2,
                    recurrent_heads=2, recurrent_head_dim=16, history_frames=2, min_radius_km=600.0,
                    max_radius_km=2000.0, encode_radius_km=900.0)
    settings.update(over)
    return NatureConfig(**settings)


def tiny_model(**over) -> NatureV1:
    torch.manual_seed(0)
    config = tiny_config(**over)
    return NatureV1(config, fibonacci_sphere(config.latent_points, num_neighbours=config.latent_neighbours,
                                             cluster_size=config.latent_cluster))


def test_old_checkpoints_load_and_the_new_paths_change_nothing_until_trained():
    old = tiny_model()
    state = {k: v for k, v in old.state_dict().items() if k != "track_consensus"}
    state["track_consensus"] = torch.full((old.config.num_leads, 4), 0.25)            # a 0.11 checkpoint
    state["head_steps"] = old.head_steps[: TRACKED_HEADS.index("satnet")].clone()
    new = tiny_model(satnet_embed=32, satnet_widths=(8, 8, 16, 16), satnet_pixels=64, ri2=True)
    new.load_state_dict(state)
    assert new.track_consensus.shape == (old.config.num_leads, len(TRACK_MEMBERS))
    assert torch.allclose(new.track_consensus[:, :4], torch.full((old.config.num_leads, 4), 0.25))
    assert float(new.track_consensus[:, 4:].abs().sum()) == 0
    old.load_state_dict(state)
    pooled = torch.randn(3, 32)
    centre = torch.tensor([[20.0, -60.0], [-15.0, 150.0], [30.0, 10.0]])
    a = old.storm_heads(pooled, centre)
    satnet = new.satnet_outputs(torch.randint(1, 255, (3, 4, 2, 64, 64), dtype=torch.uint8),
                                torch.ones(3, 4, dtype=torch.bool), centre[:, 0])
    b = new.storm_heads(pooled, centre, storm_satnet=satnet)
    for key in a:
        assert torch.equal(a[key], b[key]), key
    assert "ri2_logits" in b and "satnet_wind_kt" in b


def test_hemisphere_frame_is_an_exact_mirror():
    model = tiny_model(hemisphere_frame=True)
    with torch.no_grad():
        for p in model.storm_head_parameters():
            p.normal_(0, 0.1)
    pooled = torch.randn(1, 32).repeat(2, 1)
    centre = torch.tensor([[20.0, -60.0], [-20.0, -60.0]])
    state = torch.tensor([[0.5, -0.3, 0.8, -1.2, 0.1, 1.0]]).repeat(2, 1)
    state[1, 2] = -state[0, 2]
    env = torch.randn(1, 8).repeat(2, 1)
    env[1, 1], env[1, 3] = -env[0, 1], -env[0, 3]
    prior = torch.randn(1, model.config.num_leads, 2).repeat(2, 1, 1)
    prior[1, :, 0] = -prior[0, :, 0]
    out = model.storm_heads(pooled, centre, state, env, prior)
    north, south = out["displacement"][0], out["displacement"][1]
    assert torch.allclose(south[..., 0], -north[..., 0], atol=1e-6) and torch.allclose(south[..., 1], north[..., 1], atol=1e-6)
    assert torch.allclose(out["correlation"][1], -out["correlation"][0], atol=1e-6)
    assert torch.allclose(out["wind_radii_nmi"][1], out["wind_radii_nmi"][0][..., [1, 0, 3, 2]], atol=1e-5)
    assert torch.allclose(out["intensity_mean"][1], out["intensity_mean"][0], atol=1e-5)
    assert torch.equal(out["track_prior"], prior), "the prior is reported as given"


def test_ri2_probabilities_never_rise_with_the_threshold():
    model = tiny_model(ri2=True)
    with torch.no_grad():
        for p in model.ri2_head.parameters():
            p.normal_(0, 0.5)
    out = model.storm_heads(torch.randn(64, 32), torch.randn(64, 2) * 20, torch.randn(64, 6), torch.randn(64, 8))
    probability = torch.sigmoid(out["ri2_logits"])
    order = [i for i, (hours, _) in enumerate(ri_threshold_list()) if hours == 24]
    assert (probability[:, order[1:]] <= probability[:, order[:-1]] + 1e-6).all()


def test_intensity_correction_shifts_strong_storms_up():
    model = tiny_model()
    pooled = torch.randn(2, 32)
    before = model.storm_heads(pooled, torch.tensor([[20.0, -60.0], [25.0, -70.0]]))["intensity_mean"][..., 0]
    model.intensity_correction[:, 0] = 5.0
    model.intensity_correction[:, 1] = 3.0
    after = model.storm_heads(pooled, torch.tensor([[20.0, -60.0], [25.0, -70.0]]))["intensity_mean"][..., 0]
    wind = before / 0.514444
    expected = (wind + 5.0 + 3.0 * (wind - 65.0) / 30.0) * 0.514444
    assert torch.allclose(after, expected, atol=1e-4)


def test_cloud_motion_recovers_a_known_shift():
    torch.manual_seed(0)
    field = torch.nn.functional.avg_pool2d(torch.randn(1, 1, 96, 192), 5, 1, 2)[0] * 30 + 250
    moved = torch.roll(field, shifts=(4, -6), dims=(1, 2))
    vectors = cloud_motion(field, moved, hours=3.0, resolution_deg=0.25, latitude=torch.zeros(96), tile=16,
                           search=8, stride=2)
    inner = vectors[0, :, 32:64, 32:160]
    metres = 0.25 * 111.195 * 1000
    assert torch.allclose(inner[1].median(), torch.tensor(4 * metres / 10800), atol=0.1), "northward"
    assert torch.allclose(inner[0].median(), torch.tensor(-6 * metres / 10800), atol=0.1), "westward"
    assert float(inner[2].median()) > 0.95


def write_era5(directory, times, rows=9, cols=16):
    variables = ["2m_temperature", "mean_sea_level_pressure", "u_component_of_wind@850",
                 "toa_incident_solar_radiation_6hr", "land_sea_mask"]
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, (len(times), rows, cols, len(variables))).astype(np.float16)
    path = directory / "era5.npy"
    np.save(path, values)
    stats = Normalizer(variables, np.zeros(5, np.float32), np.ones(5, np.float32), np.zeros(5, np.float32)).to_dict()
    lat = np.linspace(-90, 90, rows).tolist()
    lon = (np.arange(cols) * 360.0 / cols).tolist()
    path.with_suffix(".json").write_text(json.dumps({
        "variables": variables, "shape": list(values.shape), "statistics": stats, "latitude": lat, "longitude": lon,
        "time": [int(t) for t in times]}))
    return path, variables


def test_retrieval_starts_as_the_stale_state_and_trains_through_the_engine(staged, tmp_path):
    from naturev1.engine import EngineSettings
    from naturev1.sattrain import evaluate_retrieval, staged_grid, train_retrieval

    root, _, _ = staged
    seconds = int(START.astype("datetime64[s]").astype(np.int64))
    times = seconds + np.arange(-8, 9) * 21600
    path, variables = write_era5(tmp_path, times)
    meta = json.loads(path.with_suffix(".json").read_text())
    grid = staged_grid(meta)
    model = tiny_model(retrieval_layers=2, retrieval_resolution=2.5)
    roles = channel_roles(variables, 8)
    stepper = StateStepper(roles, torch.ones(8), Normalizer.from_dict(meta["statistics"]), grid, None)
    frames = GlobalFrames(root)
    data = RetrievalDataset(frames, path, None, (6, 12), 8, train=False)
    assert len(data) >= 4
    item = data[0]
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in item.items()}
    out = model.retrieve(batch["satellite"], batch["satellite_available"], batch["background"],
                         batch["stale_hours"], batch["calendar"], grid)
    assert float(out["increment"].abs().max()) == 0.0, "untrained: exactly the stale state"
    state = apply_retrieval(stepper, batch["background"], out, batch["stale_hours"], batch["valid_time"])
    assert torch.equal(state[..., roles.prognostic], batch["background"][..., roles.prognostic].float())
    assert torch.equal(state[..., 4], batch["background"][..., 4].float()), "geography does not move"

    settings = EngineSettings(run="retrieval-test", output=str(tmp_path / "runs"), hours=1.0, reserve_minutes=0.0,
                              batch_size=2, workers=0, max_steps=3, warmup_steps=1, checkpoint_minutes=1e9,
                              eval_minutes=1e9, learning_rate=1e-3)
    seasons = data.times
    report = train_retrieval(model, settings, frames, path, stepper, grid, seasons, seasons, stale_hours=(6, 12))
    assert model.trained_heads()["retrieval"] == 3
    assert (tmp_path / "runs" / "retrieval-test" / "latest.pt").exists()
    assert list((tmp_path / "runs" / "retrieval-test" / "history").glob("step_*.safetensors"))
    assert set(report["by_age"]) <= {6, 12} and math.isfinite(report["skill"])
    assert float(model.retrieval.head.weight.abs().sum()) > 0
    del evaluate_retrieval


def test_satnet_trains_through_the_engine_and_resumes(staged, tmp_path):
    from naturev1.engine import EngineSettings
    from naturev1.sattrain import train_satnet

    root, table, _ = staged
    model = tiny_model(satnet_embed=32, satnet_widths=(8, 8, 16, 16), satnet_pixels=64)
    settings = EngineSettings(run="satnet-test", output=str(tmp_path / "runs"), hours=1.0, reserve_minutes=0.0,
                              batch_size=4, workers=0, max_steps=2, warmup_steps=1, checkpoint_minutes=1e9,
                              eval_minutes=1e9)
    rows = np.arange(len(table["time"]))
    patches = StormPatches(root)
    import naturev1.satdata as satdata

    original = satdata.StormSequenceDataset.__init__

    def small(self, *args, **kwargs):
        kwargs["pixels"] = 64
        original(self, *args, **kwargs)

    satdata.StormSequenceDataset.__init__ = small
    try:
        report = train_satnet(model, settings, table, patches, rows, rows)
        assert model.trained_heads()["satnet"] == 2 and math.isfinite(report["wind_mae_kt"])
        settings.max_steps = 3
        resumed = tiny_model(satnet_embed=32, satnet_widths=(8, 8, 16, 16), satnet_pixels=64)
        train_satnet(resumed, settings, table, patches, rows, rows)
        assert resumed.trained_heads()["satnet"] == 3, "continued from step 2, not from zero"
    finally:
        satdata.StormSequenceDataset.__init__ = original


def test_every_checkpoint_is_pushed_in_order(monkeypatch, tmp_path):
    import huggingface_hub

    from naturev1.engine import CheckpointPusher

    uploads = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *args, **kwargs):
            return None

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
            uploads.append((path_in_repo, commit_message, open(path_or_fileobj, "rb").read()))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    pusher = CheckpointPusher("someone/NatureV1-test")
    target = tmp_path / "latest.pt"
    for step in range(5):
        target.write_bytes(f"step {step}".encode())
        pusher.submit([(target, "checkpoints/x/latest.pt")], f"x step {step}")
    assert pusher.wait(30)
    assert [u[1] for u in uploads] == [f"x step {s}" for s in range(5)], "none skipped, in order"
    assert [u[2] for u in uploads] == [f"step {s}".encode() for s in range(5)], "each upload is its own snapshot"
    assert not list(tmp_path.glob(".push-*")), "snapshots cleaned up"


def test_learning_rate_follows_the_clock():
    from naturev1.engine import lr_factor

    assert lr_factor(0, 0.0, 100.0, warmup=10, floor=0.1) == pytest.approx(0.1)
    assert lr_factor(50, 0.0, 100.0, warmup=10, floor=0.1) == pytest.approx(1.0)
    assert lr_factor(50, 50.0, 100.0, warmup=10, floor=0.1) == pytest.approx(0.55)
    assert lr_factor(50, 100.0, 100.0, warmup=10, floor=0.1) == pytest.approx(0.1)
    assert lr_factor(50, 500.0, 100.0, warmup=10, floor=0.1) == pytest.approx(0.1)
