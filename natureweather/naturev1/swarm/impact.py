# Copyright 2026 Nathan. Apache-2.0.
"""
From where the eye goes to which communities are hit: every world of the swarm paints its wind and rain onto the
ground, and the paintings are counted parish by parish.

For each world (a track from :class:`naturev1.swarm.core.SwarmForecast`) and each hour:

* **Intensity** -- the NatureV1 intensity forecast for the storm, times a per-world error that grows with lead (8%
  at the start, about 32% by 5 days), decaying over land the way Kaplan and DeMaria (1995) found: over each land
  hour the excess over 26.7 kt falls by 9.5%, faster over high ground, and back over water the world follows the
  forecast's changes from where it was left, not from where the forecast is.
* **Wind** -- :func:`naturev1.windfield.wind_field`: Holland profile with the Vickery-Wadhera shape, a radius of
  maximum wind from Willoughby et al. (2006) times a per-world size factor, inflow, the motion asymmetry of Lin and
  Chavas (2012), land roughness and gust factors. Sustained (1-minute) and 3-second gust, at every land cell.
* **Rain** -- the R-CLIPER rain rate (Tuleya et al. 2007), multiplied where the wind blows up the terrain: forced
  ascent ``w = V . grad(h)`` lifts the rate by up to ``1 + OROGRAPHIC_PER_MS * w`` (capped at
  ``OROGRAPHIC_CAP``), which is why a slow storm can flood the windward slopes of the Blue Mountains or the
  Cordillera Central without its eye ever coming ashore. Accumulated hour by hour.
* **Arrival** -- the first hour each cell sees tropical-storm-force sustained wind.

Then for each administrative unit and each world: the share of its land under 34, 50 and 64 kt sustained winds and
64 kt gusts, its peak wind, its area-mean rain, and rain over its steep ground. Probabilities over the swarm: a unit
counts as hit when at least ``AREA_SHARE`` of its land is affected. Storm surge is not modelled.

Cost is particles x hours x land cells near the storm; only cells within ``REACH_KM`` of each world are touched,
so a Caribbean-wide map is no slower than one island. Everything runs in PyTorch on CPU or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..windfield import (DECAY_BACKGROUND_KT, DECAY_RATE, DECAY_REDUCTION, KT, _local_frame, rcliper_rate_mm_h,
                         rmw_climatology_km, wind_field)
from .geography import RegionMap

REACH_KM = 500.0
AREA_SHARE = 0.10
OROGRAPHIC_PER_MS = 1.5
OROGRAPHIC_CAP = 2.0                 # was 4; with the radar multiplier on the base rate, 4 doubled Melissa's gauge maxima over Jamaica's hills
STEEP_SLOPE = 0.08                   # 8% -- where heavy rain becomes flash floods and landslides
WIND_KT = (34.0, 50.0, 64.0)
RAIN_MM = (100.0, 250.0)


def _rain_scale_default() -> float:
    """The R-CLIPER multiplier calibrated against gauge-corrected radar (``coefficients.RAIN_SCALE``), 1 when absent."""
    try:
        from ..cyclone import coefficients

        return float(getattr(coefficients, "RAIN_SCALE", {}).get("multiplier", 1.0))
    except Exception:
        return 1.0


#: R-CLIPER rain rates are multiplied by this (see :data:`naturev1.cyclone.coefficients.RAIN_SCALE`).
RAIN_SCALE = _rain_scale_default()


@dataclass
class ImpactSettings:
    intensity_sd0: float = 0.08
    intensity_sd_per_h: float = 0.002
    size_log_sd: float = 0.30
    reach_km: float = REACH_KM
    area_share: float = AREA_SHARE
    orographic_per_ms: float = OROGRAPHIC_PER_MS
    orographic_cap: float = OROGRAPHIC_CAP
    #: multiplier on the R-CLIPER rain rate (calibrated against radar for US landfalls; 1 = Tuleya et al. as published)
    rain_scale: float = RAIN_SCALE
    steep_slope: float = STEEP_SLOPE
    land_decay: bool = True              # False when the intensities already include it (a best track)
    cell_stride: int = 1                 # paint every n-th cell in each direction (2 on a CPU: 4x faster)
    #: the forecast's own error distribution instead of the lognormal: ``{"hours": (L,), "quantiles": (L, Q) kt}``
    #: residual quantiles (coefficients.INTENSITY_SPREAD for the storm's regime); each world draws one quantile level
    intensity_quantiles: dict | None = None
    #: the observed mean 34-kt radius (km) at the analysis: the wind field's size is anchored to it (with
    #: ``size_log_sd`` spread around it, halved) instead of the climatological B
    r34_km: float | None = None
    #: boundary-layer terrain effect per unit of slope along the wind: wind speeds up on a windward slope and is
    #: sheltered in the lee (Jackson-Hunt: a fractional speed-up of about twice the slope on a crest). 0 (default)
    #: leaves it out: it is physics, not something verified against station winds here. 1.0 is a plausible value.
    terrain_speedup_per_slope: float = 0.0


@dataclass
class ImpactResult:
    region: RegionMap
    units: list                          # one dict per affected unit, sorted by hurricane-wind probability
    cell_maps: dict                      # (H, W) grids: p34, p50, p64, p_gust64, mean_rain_mm, p_rain250, arrival_h
    #: each unit dict also carries peak_gust_kt, peak_kt_p90 (90th percentile over worlds), peak_kt_if_hit and
    #: gust_kt_if_hit (over the worlds that bring at least tropical-storm force), peak_hour, hu_arrival_h
    particles: int
    hours: np.ndarray
    meta: dict = field(default_factory=dict)

    def table(self, top: int = 20, min_probability: float = 0.01) -> str:
        """A plain-text impact report, most threatened first."""
        head = (f"{'unit':38s} {'P(TS)':>6s} {'P(50kt)':>7s} {'P(HU)':>6s} {'P(HU gust)':>10s} {'peak kt':>8s} "
                f"{'rain mm':>8s} {'P(100mm)':>8s} {'P(250mm)':>8s} {'flash':>6s} {'TS arrives':>10s}")
        lines = [head, "-" * len(head)]
        shown = [u for u in self.units if u["p34"] >= min_probability][:top]
        for u in shown:
            arrival = f"{u['arrival_h']:.0f} h" if np.isfinite(u["arrival_h"]) else "-"
            lines.append(f"{u['label'][:38]:38s} {u['p34']:6.0%} {u['p50']:7.0%} {u['p64']:6.0%} {u['p_gust64']:10.0%} "
                         f"{u['peak_kt']:8.0f} {u['rain_mm']:8.0f} {u['p_rain100']:8.0%} {u['p_rain250']:8.0%} "
                         f"{u['p_flash']:6.0%} {arrival:>10s}")
        if not shown:
            lines.append("no unit has more than a 1% chance of tropical-storm-force wind")
        lines.append(f"{self.particles:,} worlds; a unit is hit when >= {self.meta.get('area_share', AREA_SHARE):.0%} "
                     "of its land is affected; rain is the area mean; flash = P(100 mm on its steep ground); "
                     "storm surge not modelled.")
        return "\n".join(lines)


def hourly(track: np.ndarray, hours: np.ndarray, step_h: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Tracks ``(N, T, 2)`` at ``hours`` resampled every ``step_h`` hours (linear in lat and continuous lon)."""
    h = np.arange(hours[0], hours[-1] + 1e-9, step_h)
    lon = np.unwrap(np.radians(track[..., 1]), axis=1)
    out = np.empty((track.shape[0], h.size, 2))
    for k in range(2):
        src = track[..., 0] if k == 0 else np.degrees(lon)
        idx = np.clip(np.searchsorted(hours, h, side="right") - 1, 0, hours.size - 2)
        w = (h - hours[idx]) / (hours[idx + 1] - hours[idx])
        out[..., k] = src[:, idx] * (1 - w) + src[:, idx + 1] * w
    out[..., 1] = (out[..., 1] + 180.0) % 360.0 - 180.0
    return out, h


def world_intensity(region: RegionMap, track_h: np.ndarray, hours: np.ndarray, forecast_kt: np.ndarray,
                    forecast_hours: np.ndarray, settings: ImpactSettings, generator: torch.Generator) -> np.ndarray:
    """``(N, T)`` peak wind (kt) of every world: the forecast with its own growing error, decayed over land."""
    n, t = track_h.shape[:2]
    water = np.interp(hours, forecast_hours, forecast_kt)
    if settings.intensity_quantiles is not None:
        # each world keeps one quantile level of the consensus's own out-of-sample error distribution for the
        # storm's regime, interpolated in time: a calibrated spread, skewed upward in the rapid regime
        q = np.asarray(settings.intensity_quantiles["quantiles"], np.float64)              # (L, Q)
        qh = np.asarray(settings.intensity_quantiles["hours"], np.float64)
        level = torch.rand(n, generator=generator, dtype=torch.float64).numpy() * (q.shape[1] - 1)
        lo = np.floor(level).astype(int)
        frac = level - lo
        hi = np.minimum(lo + 1, q.shape[1] - 1)
        at = np.stack([np.interp(hours, np.concatenate([[0.0], qh]), np.concatenate([[0.0], q[:, j]])) for j in range(q.shape[1])])  # (Q, T)
        resid = at[lo] * (1 - frac)[:, None] + at[hi] * frac[:, None]
        world = np.maximum(water[None, :] + resid, 15.0)
    else:
        z = torch.randn(n, generator=generator, dtype=torch.float64).numpy()
        sd = settings.intensity_sd0 + settings.intensity_sd_per_h * hours
        world = water[None, :] * np.exp(sd[None, :] * z[:, None])
    land = region.unit_of(track_h[..., 0], track_h[..., 1]) >= 0
    if not settings.land_decay:
        return np.maximum(world, 15.0)
    elev = np.zeros_like(world)
    if land.any():
        i = np.clip(np.round((track_h[..., 0] - region.lat[0]) / (region.lat[1] - region.lat[0])).astype(int), 0, region.lat.size - 1)
        j = np.clip(np.round((((track_h[..., 1] + 180) % 360 - 180) - region.lon[0]) / (region.lon[1] - region.lon[0])).astype(int),
                    0, region.lon.size - 1)
        elev = np.where(land, region.elevation[i, j], 0.0)
    out = world.copy()
    current = world[:, 0].copy()
    ashore = np.zeros(n, bool)
    for k in range(1, t):
        dt = hours[k] - hours[k - 1]
        change = world[:, k] - world[:, k - 1]
        on = land[:, k]
        first = on & ~ashore
        current = np.where(first, DECAY_BACKGROUND_KT + DECAY_REDUCTION * (current - DECAY_BACKGROUND_KT), current)
        rate = DECAY_RATE * (1.0 + elev[:, k] / 1000.0)
        decayed = DECAY_BACKGROUND_KT + (current - DECAY_BACKGROUND_KT) * np.exp(-rate * dt)
        current = np.where(on, np.minimum(decayed, current), np.minimum(current + change, world[:, k]))
        ashore = on
        out[:, k] = np.maximum(current, 15.0)
    return out


def paint(region: RegionMap, tracks: np.ndarray, hours: np.ndarray, forecast_kt, forecast_hours,
          weights: np.ndarray | None = None, settings: ImpactSettings | None = None, device="cpu", chunk: int = 256,
          seed: int = 0) -> ImpactResult:
    """
    Paint every world's wind and rain onto ``region`` and count them per unit.

    Args:
        tracks: ``(N, T, 2)`` world tracks (lat, lon) at ``hours`` (from 0).
        forecast_kt / forecast_hours: the intensity forecast to centre the worlds on (kt at those hours, 0 included).
        weights: per-world weights (default equal).
    """
    st = settings or ImpactSettings()
    gen = torch.Generator().manual_seed(seed)
    track_h, h = hourly(np.asarray(tracks, np.float64), np.asarray(hours, np.float64))
    n = track_h.shape[0]
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, np.float64) / np.sum(weights)
    vmax = world_intensity(region, track_h, h, np.asarray(forecast_kt, np.float64), np.asarray(forecast_hours, np.float64), st, gen)
    size_sd = st.size_log_sd * (0.5 if st.r34_km is not None and np.isfinite(st.r34_km) else 1.0)
    size = torch.exp(size_sd * torch.randn(n, generator=gen, dtype=torch.float64)).numpy()
    motion = np.zeros_like(track_h)
    d_north = np.gradient(track_h[..., 0], h, axis=1) * 111195.0 / 3600.0
    d_east = (np.gradient(np.unwrap(np.radians(track_h[..., 1]), axis=1), h, axis=1) * 6371008.8 / 3600.0
              * np.cos(np.radians(track_h[..., 0])))
    motion[..., 0], motion[..., 1] = d_east, d_north

    cells = region.cells()
    if st.cell_stride > 1:
        # thin the map, but keep at least one cell of every unit (small islands are units too)
        thin = (cells["i"] % st.cell_stride == 0) & (cells["j"] % st.cell_stride == 0)
        _, first = np.unique(cells["unit"], return_index=True)
        thin[first] = True
        cells = {k: v[thin] for k, v in cells.items()}
    dev, dt32 = torch.device(device), torch.float32
    reach_deg = st.reach_km / 111.195
    # Only cells that some world comes within reach of, at some hour, are ever touched.
    lat_all, lon_all = track_h[..., 0].ravel(), track_h[..., 1].ravel()
    coarse = np.unique(np.stack([np.floor(lat_all), np.floor(lon_all)], -1), axis=0)
    near_any = np.zeros(cells["lat"].size, bool)
    for a, b in coarse:
        near_any |= ((np.abs(cells["lat"] - (a + 0.5)) <= reach_deg + 0.75)
                     & (np.abs(((cells["lon"] - (b + 0.5) + 180) % 360 - 180) * np.cos(np.radians(cells["lat"])))
                        <= reach_deg + 0.75))
    keep = np.nonzero(near_any)[0]
    m = keep.size
    c_lat = torch.as_tensor(cells["lat"][keep], dtype=torch.float64, device=dev)
    c_lon = torch.as_tensor(cells["lon"][keep], dtype=torch.float64, device=dev)
    c_lat32, c_lon32 = c_lat.to(dt32), c_lon.to(dt32)
    sx = torch.as_tensor(cells["slope_x"][keep], dtype=dt32, device=dev)
    sy = torch.as_tensor(cells["slope_y"][keep], dtype=dt32, device=dev)
    steep = torch.as_tensor(np.hypot(cells["slope_x"][keep], cells["slope_y"][keep]) >= st.steep_slope, device=dev)
    unit = torch.as_tensor(cells["unit"][keep], dtype=torch.long, device=dev)
    n_units = len(region.names)
    all_unit = torch.as_tensor(cells["unit"], dtype=torch.long, device=dev)          # every (sampled) cell
    unit_cells = torch.bincount(all_unit, minlength=n_units).double()
    steep_all = torch.as_tensor(np.hypot(cells["slope_x"], cells["slope_y"]) >= st.steep_slope, device=dev)
    steep_cells = torch.bincount(all_unit, weights=steep_all.double(), minlength=n_units)

    peak = torch.zeros(n, m, dtype=dt32, device=dev)
    gust = torch.zeros_like(peak)
    rain = torch.zeros_like(peak)
    arrival = torch.full_like(peak, float("inf"))
    peak_hour = torch.zeros_like(peak)                     # when each cell saw its strongest wind
    hu_arrival = torch.full_like(peak, float("inf"))       # when hurricane force first arrived
    vmax_t = torch.as_tensor(vmax, dtype=torch.float64, device=dev)
    size_t = torch.as_tensor(size, dtype=torch.float64, device=dev)
    motion_t = torch.as_tensor(motion, dtype=torch.float64, device=dev)
    track_t = torch.as_tensor(track_h, dtype=torch.float64, device=dev)
    if m:
        for k in range(h.size):
            lat_k, lon_k = track_t[:, k, 0], track_t[:, k, 1]
            # spatially compact chunks: sort the worlds by where they are this hour
            order = torch.argsort(torch.floor(lat_k / 2.0) * 1000.0 + lon_k)
            dt = (h[k + 1] - h[k]) if k + 1 < h.size else (h[k] - h[k - 1] if k > 0 else 1.0)
            for s0 in range(0, n, chunk):
                ix = order[s0:s0 + chunk]
                la, lo = lat_k[ix], lon_k[ix]
                cos_c = torch.cos(torch.deg2rad(c_lat))
                near = ((c_lat >= la.min() - reach_deg) & (c_lat <= la.max() + reach_deg)
                        & (((c_lon - lo.min() + 180) % 360 - 180) * cos_c >= -reach_deg)
                        & (((c_lon - lo.max() + 180) % 360 - 180) * cos_c <= reach_deg))
                idx = torch.nonzero(near).squeeze(1)
                if idx.numel() == 0:
                    continue
                vk = vmax_t[ix, k]
                rmw = rmw_climatology_km(vk * KT, la) * size_t[ix]
                pl, po = c_lat32[idx], c_lon32[idx]
                la32, lo32 = la.to(dt32), lo.to(dt32)
                r34 = None
                if st.r34_km is not None and np.isfinite(st.r34_km):
                    # the observed gale radius anchors the outer wind field (the profile's B is fitted to it)
                    r34 = (float(st.r34_km) * size_t[ix]).to(dt32)
                wf = wind_field(pl, po, la32, lo32, vk.to(dt32), rmw_km=rmw.to(dt32), r34_km=r34, motion_ms=motion_t[ix, k].to(dt32),
                                land=torch.ones(idx.numel(), dtype=dt32, device=dev), dtype=dt32)
                north, east = _local_frame(pl.view(1, -1), po.view(1, -1), la32.view(-1, 1), lo32.view(-1, 1))
                r = torch.sqrt(north ** 2 + east ** 2)
                far = r > st.reach_km
                speed = wf["speed_kt"].masked_fill(far, 0.0)
                g = wf["gust_kt"].masked_fill(far, 0.0)
                if st.terrain_speedup_per_slope:
                    # slope along the wind (dimensionless): positive upslope -> speed-up, negative (lee) -> sheltering
                    wind_speed = torch.sqrt(wf["u_ms"] ** 2 + wf["v_ms"] ** 2).clamp_min(1e-3)
                    along_slope = (wf["u_ms"] * sx[idx] + wf["v_ms"] * sy[idx]) / wind_speed
                    factor = (1.0 + st.terrain_speedup_per_slope * along_slope).clamp(0.7, 1.5)
                    speed = speed * factor
                    g = g * factor
                rate = rcliper_rate_mm_h(r, vk.to(dt32).view(-1, 1)).masked_fill(far, 0.0)
                ascent = (wf["u_ms"] * sx[idx] + wf["v_ms"] * sy[idx]).clamp_min(0.0)
                rate = rate * (1.0 + st.orographic_per_ms * ascent).clamp(max=st.orographic_cap) * float(st.rain_scale)
                rows, cols = ix.view(-1, 1), idx.view(1, -1)
                stronger = speed > peak[rows, cols]
                peak_hour[rows, cols] = torch.where(stronger, torch.full_like(speed, float(h[k])), peak_hour[rows, cols])
                peak[rows, cols] = torch.maximum(peak[rows, cols], speed)
                gust[rows, cols] = torch.maximum(gust[rows, cols], g)
                rain[rows, cols] = rain[rows, cols] + rate * float(dt)
                prev = arrival[rows, cols]
                arrival[rows, cols] = torch.where(torch.isinf(prev) & (speed >= WIND_KT[0]), torch.full_like(speed, float(h[k])), prev)
                prev = hu_arrival[rows, cols]
                hu_arrival[rows, cols] = torch.where(torch.isinf(prev) & (speed >= WIND_KT[2]), torch.full_like(speed, float(h[k])), prev)

    wk = torch.as_tensor(w, dtype=dt32, device=dev)
    cell_acc = {}
    for key, thr in (("p34", WIND_KT[0]), ("p50", WIND_KT[1]), ("p64", WIND_KT[2])):
        cell_acc[key] = (wk[:, None] * (peak >= thr)).sum(0)
    cell_acc["p_gust64"] = (wk[:, None] * (gust >= WIND_KT[2])).sum(0)
    cell_acc["peak"] = (wk[:, None] * peak).sum(0)
    cell_acc["gust_kt"] = (wk[:, None] * gust).sum(0)
    cell_acc["rain"] = (wk[:, None] * rain).sum(0)
    cell_acc["p_rain250"] = (wk[:, None] * (rain >= RAIN_MM[1])).sum(0)
    hit = torch.isfinite(arrival)
    cell_acc["arr_sum"] = (wk[:, None] * torch.where(hit, arrival, torch.zeros_like(arrival))).sum(0)
    cell_acc["arr_w"] = (wk[:, None] * hit).sum(0)

    def share(mask):
        out = torch.zeros(n, n_units, dtype=dt32, device=dev)
        out.index_add_(1, unit, mask.to(dt32))
        return out / unit_cells.to(dt32).clamp_min(1.0)

    unit_acc = {}
    for key, mask in (("p34", peak >= WIND_KT[0]), ("p50", peak >= WIND_KT[1]), ("p64", peak >= WIND_KT[2]),
                      ("p_gust64", gust >= WIND_KT[2])):
        unit_acc[key] = (wk[:, None] * (share(mask) >= st.area_share)).sum(0)
    upeak = torch.zeros(n, n_units, dtype=dt32, device=dev).scatter_reduce(1, unit.expand(n, -1), peak, reduce="amax",
                                                                          include_self=True)
    unit_acc["peak"] = (wk[:, None] * upeak).sum(0)
    urain = torch.zeros(n, n_units, dtype=dt32, device=dev).index_add_(1, unit, rain) / unit_cells.to(dt32).clamp_min(1.0)
    unit_acc["rain"] = (wk[:, None] * urain).sum(0)
    unit_acc["p_rain100"] = (wk[:, None] * (urain >= RAIN_MM[0])).sum(0)
    unit_acc["p_rain250"] = (wk[:, None] * (urain >= RAIN_MM[1])).sum(0)
    srain = torch.zeros(n, n_units, dtype=dt32, device=dev).index_add_(1, unit, rain * steep) \
        / steep_cells.to(dt32).clamp_min(1.0)
    unit_acc["p_flash"] = (wk[:, None] * ((srain >= RAIN_MM[0]) & (steep_cells > 0))).sum(0)
    uarr = torch.full((n, n_units), float("inf"), dtype=dt32, device=dev).scatter_reduce(
        1, unit.expand(n, -1), arrival, reduce="amin", include_self=True)
    reached = torch.isfinite(uarr)
    unit_acc["arr_sum"] = (wk[:, None] * torch.where(reached, uarr, torch.zeros_like(uarr))).sum(0)
    unit_acc["arr_w"] = (wk[:, None] * reached).sum(0)
    # the parish's peak gust, the hour of its peak wind (peak-weighted over its cells), when hurricane force arrives,
    # and the 90th percentile of its peak wind over the worlds -- the numbers a parish plans around
    ugust = torch.zeros(n, n_units, dtype=dt32, device=dev).scatter_reduce(1, unit.expand(n, -1), gust, reduce="amax", include_self=True)
    unit_acc["gust"] = (wk[:, None] * ugust).sum(0)
    hour_num = torch.zeros(n, n_units, dtype=dt32, device=dev).index_add_(1, unit, peak_hour * peak)
    hour_den = torch.zeros(n, n_units, dtype=dt32, device=dev).index_add_(1, unit, peak)
    uhour = torch.where(hour_den > 0, hour_num / hour_den.clamp_min(1e-9), torch.zeros_like(hour_num))
    windy = upeak >= WIND_KT[0]
    unit_acc["hour_sum"] = (wk[:, None] * torch.where(windy, uhour, torch.zeros_like(uhour))).sum(0)
    unit_acc["hour_w"] = (wk[:, None] * windy).sum(0)
    # the peak given that the unit is hit at all (tropical-storm force): the number a parish plans around, not
    # diluted by the worlds in which the storm passes elsewhere
    unit_acc["peak_hit"] = (wk[:, None] * torch.where(windy, upeak, torch.zeros_like(upeak))).sum(0)
    unit_acc["gust_hit"] = (wk[:, None] * torch.where(windy, ugust, torch.zeros_like(ugust))).sum(0)
    uhu = torch.full((n, n_units), float("inf"), dtype=dt32, device=dev).scatter_reduce(1, unit.expand(n, -1), hu_arrival, reduce="amin", include_self=True)
    hu_reached = torch.isfinite(uhu)
    unit_acc["hu_sum"] = (wk[:, None] * torch.where(hu_reached, uhu, torch.zeros_like(uhu))).sum(0)
    unit_acc["hu_w"] = (wk[:, None] * hu_reached).sum(0)
    order = torch.argsort(upeak, dim=0)
    sorted_peak = torch.gather(upeak, 0, order)
    cum = torch.cumsum(torch.gather(wk[:, None].expand(-1, n_units), 0, order), dim=0)
    at90 = (cum >= 0.9).to(torch.int64).argmax(dim=0)
    unit_acc["peak_p90"] = torch.gather(sorted_peak, 0, at90.view(1, -1))[0]
    # rain the same way: the area mean given the parish is hit, its 90th percentile over the worlds, and the parish's
    # wettest cell (what a rain gauge reports) given it is hit -- the swarm mean is diluted by the worlds that miss
    unit_acc["rain_hit"] = (wk[:, None] * torch.where(windy, urain, torch.zeros_like(urain))).sum(0)
    order_r = torch.argsort(urain, dim=0)
    cum_r = torch.cumsum(torch.gather(wk[:, None].expand(-1, n_units), 0, order_r), dim=0)
    unit_acc["rain_p90"] = torch.gather(torch.gather(urain, 0, order_r), 0, (cum_r >= 0.9).to(torch.int64).argmax(dim=0).view(1, -1))[0]
    urain_max = torch.zeros(n, n_units, dtype=dt32, device=dev).scatter_reduce(1, unit.expand(n, -1), rain, reduce="amax", include_self=True)
    unit_acc["rain_max_hit"] = (wk[:, None] * torch.where(windy, urain_max, torch.zeros_like(urain_max))).sum(0)
    cells = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in cells.items()}

    shape = region.unit.shape
    maps = {}
    for key in ("p34", "p50", "p64", "p_gust64", "p_rain250"):
        grid = np.full(shape, np.nan)
        grid[cells["i"], cells["j"]] = cell_acc[key].cpu().numpy()
        maps[key] = grid
    grid = np.full(shape, np.nan)
    grid[cells["i"], cells["j"]] = cell_acc["rain"].cpu().numpy()
    maps["mean_rain_mm"] = grid
    for key, name in (("peak", "mean_peak_kt"), ("gust_kt", "mean_gust_kt")):
        grid = np.full(shape, np.nan)
        grid[cells["i"], cells["j"]] = cell_acc[key].cpu().numpy()
        maps[name] = grid
    grid = np.full(shape, np.nan)
    aw = cell_acc["arr_w"].cpu().numpy()
    grid[cells["i"], cells["j"]] = np.where(aw > 0.05, cell_acc["arr_sum"].cpu().numpy() / np.maximum(aw, 1e-9), np.nan)
    maps["arrival_h"] = grid
    ua = {k: v.cpu().numpy() for k, v in unit_acc.items()}
    units = []
    for u in range(n_units):
        if ua["p34"][u] <= 0 and ua["rain"][u] < 1.0:
            continue
        units.append({"unit": u, "name": region.names[u], "country": region.countries[u], "label": region.label(u),
                      "p34": ua["p34"][u], "p50": ua["p50"][u], "p64": ua["p64"][u], "p_gust64": ua["p_gust64"][u],
                      "peak_kt": ua["peak"][u], "rain_mm": ua["rain"][u], "p_rain100": ua["p_rain100"][u],
                      "p_rain250": ua["p_rain250"][u], "p_flash": ua["p_flash"][u],
                      "arrival_h": ua["arr_sum"][u] / ua["arr_w"][u] if ua["arr_w"][u] > 0.05 else float("nan"),
                      "peak_gust_kt": ua["gust"][u], "peak_kt_p90": ua["peak_p90"][u],
                      "peak_kt_if_hit": ua["peak_hit"][u] / ua["hour_w"][u] if ua["hour_w"][u] > 0.05 else float("nan"),
                      "gust_kt_if_hit": ua["gust_hit"][u] / ua["hour_w"][u] if ua["hour_w"][u] > 0.05 else float("nan"),
                      "peak_hour": ua["hour_sum"][u] / ua["hour_w"][u] if ua["hour_w"][u] > 0.05 else float("nan"),
                      "hu_arrival_h": ua["hu_sum"][u] / ua["hu_w"][u] if ua["hu_w"][u] > 0.05 else float("nan"),
                      "rain_mm_if_hit": ua["rain_hit"][u] / ua["hour_w"][u] if ua["hour_w"][u] > 0.05 else float("nan"),
                      "rain_mm_p90": ua["rain_p90"][u],
                      "rain_max_mm_if_hit": ua["rain_max_hit"][u] / ua["hour_w"][u] if ua["hour_w"][u] > 0.05 else float("nan")})
    units.sort(key=lambda d: (-d["p64"], -d["p_gust64"], -d["p34"], -d["rain_mm"]))
    return ImpactResult(region, units, maps, n, h, {"area_share": st.area_share})


def impact_from_forecast(region: RegionMap, result: dict, case, particles: int | None = None, device="cpu",
                         settings: ImpactSettings | None = None, seed: int = 0, r34_km: float | None = None) -> ImpactResult:
    """
    The impact report for a :func:`naturev1.cyclone.forecast.physics_forecast` result that carries a swarm
    (``swarm=N``): its worlds, re-centred on the consensus track, carry the NatureV1 intensity forecast, spread by the
    forecast's own calibrated error distribution for the storm's regime when the result has one
    (``category_probability``), and the wind field sized by the observed 34-kt radius ``r34_km`` when given.
    """
    import dataclasses

    st = settings or ImpactSettings()
    if st.intensity_quantiles is None and result.get("intensity_regime") is not None:
        try:
            from ..cyclone import coefficients

            table = coefficients.INTENSITY_SPREAD
            st = dataclasses.replace(st, intensity_quantiles={"hours": np.asarray(table["leads"], float),
                                                              "quantiles": np.asarray(table["quantiles"][result["intensity_regime"]], float)})
        except Exception:
            pass
    if r34_km is not None and np.isfinite(r34_km) and st.r34_km is None:
        st = dataclasses.replace(st, r34_km=float(r34_km))
    settings = st
    fc = result["swarm"]["forecast"]
    tracks, hours = fc.track, fc.track_hours
    weights = fc.weights
    if particles is not None and particles < tracks.shape[0]:
        pick = np.random.default_rng(seed).choice(tracks.shape[0], particles, replace=False, p=weights / weights.sum())
        tracks, weights = tracks[pick], np.full(particles, 1.0 / particles)
    leads = np.asarray(result["leads"], float)
    kt = np.asarray(result.get("intensity", np.full(leads.size, case.wind_kt)), float)
    kt = np.where(np.isfinite(kt), kt, np.nan)
    good = np.isfinite(kt)
    f_hours = np.concatenate([[0.0], leads[good]])
    f_kt = np.concatenate([[case.wind_kt], kt[good]])
    return paint(region, tracks, hours, f_kt, f_hours, weights, settings, device, seed=seed)


__all__ = ["AREA_SHARE", "RAIN_SCALE", "ImpactResult", "ImpactSettings", "hourly", "impact_from_forecast", "paint", "world_intensity"]
