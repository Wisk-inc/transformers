# Copyright 2026 Nathan. Apache-2.0.
"""
The wind a storm puts on the ground: a full asymmetric surface wind field, and the hazard products built on it.

The heads forecast a handful of numbers -- peak wind, the radius of maximum wind, how far gale and hurricane
force extend per quadrant. What people need is the wind *at a place*: will my town see hurricane-force wind,
when do gales arrive, how much rain. This module turns the forecast into exactly that, with the parametric
physics every operational wind-hazard system uses:

* **The radial profile** of Holland (1980), ``V(r) = Vmax [(Rmax/r)^B exp(1 - (Rmax/r)^B)]^(1/2)``, with B
  from Vickery and Wadhera (2008) or -- when the gale radius is known -- fitted so the profile passes through
  34 kt at that radius, quadrant by quadrant. Radius of maximum wind from Willoughby, Darling and Rahn (2006)
  when the eyewall head has no answer.
* **Direction**: cyclonic, turned inward by the inflow angle of Phadke et al. (2003) -- 10 degrees inside the
  eyewall rising to 25 beyond it.
* **Asymmetry** from the storm's motion (Lin and Chavas 2012): 55% of the translation vector, turned 20 degrees
  cyclonically, fading with the vortex -- the reason the right-front quadrant is the dangerous one. Not added
  on top of quadrant radii, which already contain it.
* **Land**: surface roughness takes ~20% off the sustained wind; gusts are a larger multiple of it over land;
  after landfall the peak wind decays as Kaplan and DeMaria (1995): ``V = Vb + (R V0 - Vb) exp(-a t)``.
* **Hazard products**, NHC-style, by Monte Carlo over the forecast's own distribution: coherent track
  realisations drawn from the scenario mixture, intensities from the intensity distribution, sizes from the
  radii -- each turned into a wind field every hour. Out come the probability of 34, 50 and 64 kt winds by
  each lead at any set of points, the earliest-reasonable and most-likely arrival time of gales, and
  rainfall totals from the R-CLIPER rain-rate climatology (Tuleya et al. 2007).
"""

from __future__ import annotations

import math

import torch

KT = 0.514444
NMI_KM = 1.852
KM_PER_DEGREE = 111.195
#: Kaplan and DeMaria (1995) inland decay: background wind (kt), landfall reduction factor, e-folding rate (1/h).
DECAY_BACKGROUND_KT, DECAY_REDUCTION, DECAY_RATE = 26.7, 0.9, 0.095
#: Sustained-wind reduction over land (roughness), and 3-second gust over 1-minute sustained, water / land.
LAND_REDUCTION = 0.8
GUST_FACTOR_WATER, GUST_FACTOR_LAND = 1.2, 1.35
#: Lin and Chavas (2012): share of the translation vector added, and its cyclonic rotation (degrees).
MOTION_SHARE, MOTION_TURN_DEG = 0.55, 20.0
#: R-CLIPER (Tuleya et al. 2007) coefficients: T0, Tm (mm/h), rm, re (km), each a + b U.
RCLIPER = {"t0": (-1.10, 3.96), "tm": (-1.60, 4.80), "rm": (64.5, -13.0), "re": (150.0, -16.0)}
THRESHOLDS_KT = (34.0, 50.0, 64.0)


def rmw_climatology_km(vmax_ms: torch.Tensor, latitude: torch.Tensor) -> torch.Tensor:
    """Willoughby, Darling and Rahn (2006): Rmax = 46.4 exp(-0.0155 Vmax + 0.0169 |lat|) km."""
    return 46.4 * torch.exp(-0.0155 * vmax_ms + 0.0169 * latitude.abs())


def holland_b(rmw_km: torch.Tensor, latitude: torch.Tensor) -> torch.Tensor:
    """Vickery and Wadhera (2008): B = 1.881 - 0.00557 Rmax - 0.01295 |lat|, kept within [0.8, 2.5]."""
    return (1.881 - 0.00557 * rmw_km - 0.01295 * latitude.abs()).clamp(0.8, 2.5)


def holland_profile(r_km: torch.Tensor, vmax: torch.Tensor, rmw_km: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Surface wind at radius ``r`` (same units as ``vmax``). Broadcasts."""
    x = (rmw_km / r_km.clamp_min(1e-3)) ** b
    return vmax * torch.sqrt((x * torch.exp(1.0 - x)).clamp_min(0.0))


def fit_b(vmax: torch.Tensor, rmw_km: torch.Tensor, radius_km: torch.Tensor, speed: torch.Tensor,
          iterations: int = 40) -> torch.Tensor:
    """The Holland B putting ``speed`` at ``radius_km`` (beyond the eyewall), by bisection on [0.3, 3.5]."""
    low = torch.full_like(vmax, 0.3)
    high = torch.full_like(vmax, 3.5)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        stronger = holland_profile(radius_km, vmax, rmw_km, middle) > speed
        low = torch.where(stronger, middle, low)                  # larger B makes the outer wind weaker
        high = torch.where(stronger, high, middle)
    return 0.5 * (low + high)


def radius_of(speed: torch.Tensor, vmax: torch.Tensor, rmw_km: torch.Tensor, b: torch.Tensor,
              iterations: int = 50) -> torch.Tensor:
    """The radius beyond the eyewall where the profile falls to ``speed``; 0 where the storm never reaches it."""
    low = rmw_km.clone()
    high = torch.full_like(rmw_km, 2000.0)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        faster = holland_profile(middle, vmax, rmw_km, b) > speed
        low = torch.where(faster, middle, low)
        high = torch.where(faster, high, middle)
    return torch.where(vmax > speed, 0.5 * (low + high), torch.zeros_like(rmw_km))


def inflow_angle_deg(r_km: torch.Tensor, rmw_km: torch.Tensor) -> torch.Tensor:
    """Phadke et al. (2003): 10 degrees inside Rmax, 10 + 75 (r/Rmax - 1) to 1.2 Rmax, 25 beyond."""
    ratio = r_km / rmw_km.clamp_min(1.0)
    return torch.where(ratio < 1.0, torch.full_like(ratio, 10.0),
                       torch.where(ratio < 1.2, 10.0 + 75.0 * (ratio - 1.0), torch.full_like(ratio, 25.0)))


def land_decay_kt(landfall_kt: torch.Tensor, hours_over_land: torch.Tensor) -> torch.Tensor:
    """Kaplan and DeMaria (1995) inland decay of the peak wind, from its value at landfall."""
    decayed = DECAY_BACKGROUND_KT + (DECAY_REDUCTION * landfall_kt - DECAY_BACKGROUND_KT) * torch.exp(
        -DECAY_RATE * hours_over_land)
    return torch.where(hours_over_land > 0, torch.minimum(decayed, landfall_kt), landfall_kt)


def _local_frame(points_lat, points_lon, centre_lat, centre_lon):
    """(north_km, east_km) from the centre to every point -- a local tangent plane, good to a few hundred km."""
    north = (points_lat - centre_lat) * KM_PER_DEGREE
    east = ((points_lon - centre_lon + 180.0) % 360.0 - 180.0) * KM_PER_DEGREE * torch.cos(
        torch.deg2rad(0.5 * (points_lat + centre_lat)))
    return north, east


def wind_field(points_lat: torch.Tensor, points_lon: torch.Tensor, centre_lat: torch.Tensor, centre_lon: torch.Tensor,
               vmax_kt: torch.Tensor, rmw_km: torch.Tensor | None = None, r34_km: torch.Tensor | None = None,
               motion_ms: torch.Tensor | None = None, land: torch.Tensor | None = None,
               dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    """
    The 1-minute sustained surface wind of a storm at a set of points.

    Args:
        points_lat, points_lon: ``(P,)`` degrees.
        centre_lat, centre_lon, vmax_kt: ``(S,)`` -- one or many storms (or realisations) at once.
        rmw_km: ``(S,)``; climatology when None.
        r34_km: ``(S,)`` mean gale radius, or ``(S, 4)`` per quadrant NE/SE/SW/NW (total wind, motion included);
            B from Vickery and Wadhera when None.
        motion_ms: ``(S, 2)`` storm motion (east, north) m/s -- adds the motion asymmetry unless quadrant
            radii were given.
        land: ``(P,)`` land fraction of each point -- roughness and gust factor.

        dtype: arithmetic precision (float32 for large swarms painted onto maps).

    Returns:
        ``{"speed_kt": (S, P), "u_ms", "v_ms", "gust_kt"}``.
    """
    lat_p, lon_p = points_lat.to(dtype).view(1, -1), points_lon.to(dtype).view(1, -1)
    clat, clon = centre_lat.to(dtype).view(-1, 1), centre_lon.to(dtype).view(-1, 1)
    vmax = vmax_kt.to(dtype).view(-1, 1).clamp_min(0.0)
    if rmw_km is None:
        rmw = rmw_climatology_km(vmax * KT, clat)
    else:
        rmw = rmw_km.to(dtype).view(-1, 1).clamp(5.0, 300.0)
    north, east = _local_frame(lat_p, lon_p, clat, clon)
    r = torch.sqrt(north ** 2 + east ** 2).clamp_min(0.5)
    azimuth = torch.atan2(east, north)                                     # from north, clockwise
    hemisphere = torch.where(clat >= 0, 1.0, -1.0)
    quadrant_radii = r34_km is not None and r34_km.dim() == 2 and r34_km.shape[-1] == 4
    if quadrant_radii:
        radii = r34_km.to(dtype)
        target = torch.full_like(radii, 34.0)
        bq = fit_b(vmax.expand_as(radii), rmw.expand_as(radii), radii.clamp_min(1.0), target)
        bq = torch.where((radii > rmw) & (vmax > 34.0), bq, holland_b(rmw, clat).expand_as(radii))
        # Smoothly between quadrant centres (45, 135, 225, 315 degrees), periodically.
        centres = torch.tensor([45.0, 135.0, 225.0, 315.0], dtype=dtype) * math.pi / 180.0
        position = (azimuth % (2 * math.pi)) - centres[0]
        index = torch.floor(position / (math.pi / 2)).long() % 4
        frac = (position % (math.pi / 2)) / (math.pi / 2)
        b = (bq.gather(1, index) * (1 - frac) + bq.gather(1, (index + 1) % 4) * frac)
    elif r34_km is not None:
        radius = r34_km.to(dtype).view(-1, 1)
        fitted = fit_b(vmax, rmw, radius.clamp_min(1.0), torch.full_like(vmax, 34.0))
        b = torch.where((radius > rmw) & (vmax > 34.0), fitted, holland_b(rmw, clat))
    else:
        b = holland_b(rmw, clat)
    speed = holland_profile(r, vmax, rmw, b) * KT                          # m/s
    turn = torch.deg2rad(inflow_angle_deg(r, rmw))
    # Cyclonic tangential direction is anticlockwise in the north; the inflow turns it toward the centre.
    radial_e, radial_n = east / r, north / r
    tangential_e, tangential_n = -radial_n * hemisphere, radial_e * hemisphere
    u = speed * (tangential_e * torch.cos(turn) - radial_e * torch.sin(turn))
    v = speed * (tangential_n * torch.cos(turn) - radial_n * torch.sin(turn))
    if motion_ms is not None and not quadrant_radii:
        motion = motion_ms.to(dtype)
        angle = torch.deg2rad(torch.tensor(MOTION_TURN_DEG, dtype=dtype)) * hemisphere
        mu = motion[:, :1] * torch.cos(angle) - motion[:, 1:] * torch.sin(angle)
        mv = motion[:, :1] * torch.sin(angle) + motion[:, 1:] * torch.cos(angle)
        fade = (speed / (vmax * KT).clamp_min(1e-3)).clamp(0.0, 1.0)
        u = u + MOTION_SHARE * mu * fade
        v = v + MOTION_SHARE * mv * fade
    total = torch.sqrt(u ** 2 + v ** 2) / KT
    gust = GUST_FACTOR_WATER
    if land is not None:
        share = land.to(dtype).view(1, -1).clamp(0.0, 1.0)
        reduction = 1.0 - (1.0 - LAND_REDUCTION) * share
        total, u, v = total * reduction, u * reduction, v * reduction
        gust = GUST_FACTOR_WATER + (GUST_FACTOR_LAND - GUST_FACTOR_WATER) * share
    return {"speed_kt": total, "u_ms": u, "v_ms": v, "gust_kt": total * gust}


def rcliper_rate_mm_h(r_km: torch.Tensor, vmax_kt: torch.Tensor, tail: float = 1.0) -> torch.Tensor:
    """R-CLIPER rain rate (Tuleya et al. 2007): linear to a peak at ``rm``, then exponential decay with scale ``re``;
    ``tail`` widens that decay scale (1.5 against gauge-corrected radar: the published profile is too concentrated)."""
    u = 1.0 + (vmax_kt - 35.0) / 33.0
    t0 = (RCLIPER["t0"][0] + RCLIPER["t0"][1] * u).clamp_min(0.0)
    tm = (RCLIPER["tm"][0] + RCLIPER["tm"][1] * u).clamp_min(0.0)
    rm = (RCLIPER["rm"][0] + RCLIPER["rm"][1] * u).clamp_min(10.0)
    re = (RCLIPER["re"][0] + RCLIPER["re"][1] * u).clamp_min(20.0) * float(tail)
    inner = t0 + (tm - t0) * (r_km / rm)
    outer = tm * torch.exp(-(r_km - rm) / re)
    return torch.where(r_km < rm, inner, outer).clamp_min(0.0)


def sample_storm_paths(outputs: dict, batch_index: int, lead_times_hours, samples: int = 500,
                       generator: torch.Generator | None = None, trained: dict | None = None) -> dict[str, torch.Tensor]:
    """
    Coherent realisations of the forecast: ``(samples, L)`` displacements, winds, gale radii and eyewall radii.

    A track realisation picks a scenario by its probability and then one standard-normal draw that is used at
    every lead, so each realisation is a smooth track that stays on one side of its scenario's mean -- not a
    zigzag of independent draws. Intensity is drawn the same way from the intensity distribution. Radii come
    from the wind-radii and eyewall heads when they were trained, from climatology otherwise.
    """
    probabilities = outputs["mode_logits"][batch_index].float().softmax(-1)
    mean = outputs["displacement"][batch_index].float()                          # (M, L, 2)
    scale = outputs["log_scale"][batch_index].float().exp()                     # (M, L, 2)
    rho = outputs["correlation"][batch_index].float()                           # (M, L)
    mode = torch.multinomial(probabilities, samples, replacement=True, generator=generator)
    z = torch.randn(samples, 2, generator=generator)
    sx, sy, r = scale[mode, :, 0], scale[mode, :, 1], rho[mode]
    zx = z[:, :1]
    zy = r * z[:, :1] + torch.sqrt((1 - r ** 2).clamp_min(1e-6)) * z[:, 1:]
    displacement = torch.stack([mean[mode, :, 0] + sx * zx, mean[mode, :, 1] + sy * zy], -1)
    wind_mean = outputs["intensity_mean"][batch_index, :, 0].float() / KT
    wind_sd = (0.5 * outputs["intensity_log_var"][batch_index, :, 0].float()).exp() / KT
    winds = (wind_mean.view(1, -1) + wind_sd.view(1, -1) * torch.randn(samples, 1, generator=generator)).clamp_min(15.0)
    use_radii = trained is None or trained.get("wind_radii", 0) > 0
    use_rmw = trained is None or trained.get("eyewall_rmw", 0) > 0
    r34 = None
    if use_radii and "wind_radii_nmi" in outputs:
        r34 = outputs["wind_radii_nmi"][batch_index, :, 0].float().clamp_min(0.0) * NMI_KM   # (L, 4) km
        r34 = r34.unsqueeze(0).expand(samples, -1, -1)
    rmw = None
    if use_rmw and "eyewall_rmw_nmi" in outputs:
        rmw = (outputs["eyewall_rmw_nmi"][batch_index].float() * NMI_KM).view(1, -1).expand(samples, -1)
    return {"displacement": displacement, "wind_kt": winds, "r34_km": r34, "rmw_km": rmw}


@torch.no_grad()
def wind_hazard(outputs: dict, batch_index: int, origin: tuple[float, float], lead_times_hours, points_lat,
                points_lon, samples: int = 400, step_hours: float = 1.0, land_fn=None, seed: int = 0,
                trained: dict | None = None) -> dict:
    """
    Wind-speed probabilities and gale arrival times at points, NHC-style, by Monte Carlo over the forecast.

    Every realisation (:func:`sample_storm_paths`) is interpolated to ``step_hours`` and turned into a wind
    field at every step; over land the peak wind decays from its landfall value (Kaplan and DeMaria).

    Args:
        origin: the storm's position now. points_lat, points_lon: ``(P,)``.
        land_fn: ``(lat, lon) -> land fraction`` for arrays of positions; None treats everything as water.

    Returns:
        ``{"probability": (P, L, 3)`` cumulative chance of >= 34/50/64 kt by each lead, ``"arrival_hours":
        (P, 2)`` earliest-reasonable (10th percentile) and most-likely (median) hour gales begin, NaN when
        under 10% / 50% of realisations bring them, ``"peak_kt": (P,)`` median of each realisation's peak,
        ``"rain_mm": (P, 2)`` median and 90th-percentile storm-total rain (R-CLIPER)}``.
    """
    generator = torch.Generator().manual_seed(seed)
    paths = sample_storm_paths(outputs, batch_index, lead_times_hours, samples, generator, trained)
    leads = torch.tensor([0.0] + [float(h) for h in lead_times_hours])
    lat0, lon0 = float(origin[0]), float(origin[1])
    disp = torch.cat([torch.zeros(samples, 1, 2), paths["displacement"].double()], 1)              # (S, L+1, 2)
    wind_now = paths["wind_kt"][:, :1].double()
    winds = torch.cat([wind_now, paths["wind_kt"].double()], 1)
    r34 = None if paths["r34_km"] is None else torch.cat([paths["r34_km"][:, :1], paths["r34_km"]], 1).double()
    rmw = None if paths["rmw_km"] is None else torch.cat([paths["rmw_km"][:, :1], paths["rmw_km"]], 1).double()
    points_lat = torch.as_tensor(points_lat, dtype=torch.float64).view(-1)
    points_lon = torch.as_tensor(points_lon, dtype=torch.float64).view(-1)
    land_points = None if land_fn is None else torch.as_tensor(land_fn(points_lat, points_lon), dtype=torch.float64)
    hours = torch.arange(0.0, float(leads[-1]) + 1e-9, step_hours, dtype=torch.float64)
    peak = torch.zeros(samples, points_lat.numel(), dtype=torch.float64)
    first = torch.full((samples, points_lat.numel()), float("nan"), dtype=torch.float64)
    reach = torch.zeros(samples, points_lat.numel(), len(THRESHOLDS_KT), len(lead_times_hours), dtype=torch.bool)
    rain = torch.zeros(samples, points_lat.numel(), dtype=torch.float64)
    over_land_hours = torch.zeros(samples, dtype=torch.float64)
    landfall_wind = winds[:, 0].clone()
    previous = None
    for hour in hours.tolist():
        k = int(torch.searchsorted(leads, torch.tensor(hour, dtype=leads.dtype), right=True).clamp(1, len(leads) - 1)) - 1
        span = float(leads[k + 1] - leads[k])
        frac = (hour - float(leads[k])) / span if span > 0 else 0.0
        d = disp[:, k] * (1 - frac) + disp[:, k + 1] * frac
        lat, lon = lat0 + d[:, 0], (lon0 + d[:, 1] + 180.0) % 360.0 - 180.0
        vmax = winds[:, k] * (1 - frac) + winds[:, k + 1] * frac
        if land_fn is not None:
            on_land = torch.as_tensor(land_fn(lat, lon), dtype=torch.float64) > 0.5
            landfall_wind = torch.where(on_land & (over_land_hours == 0), vmax, landfall_wind)
            over_land_hours = torch.where(on_land, over_land_hours + step_hours, torch.zeros_like(over_land_hours))
            vmax = torch.where(on_land, land_decay_kt(landfall_wind, over_land_hours), vmax)
        motion = None
        if previous is not None:
            dn = (lat - previous[0]) * KM_PER_DEGREE * 1000.0
            de = (((lon - previous[1] + 180) % 360) - 180) * KM_PER_DEGREE * 1000.0 * torch.cos(torch.deg2rad(lat))
            motion = torch.stack([de, dn], -1) / (step_hours * 3600.0)
        previous = (lat, lon)
        radius = None
        if r34 is not None:
            radius = r34[:, k] * (1 - frac) + r34[:, k + 1] * frac
        eyewall = None if rmw is None else rmw[:, k] * (1 - frac) + rmw[:, k + 1] * frac
        field = wind_field(points_lat, points_lon, lat, lon, vmax, eyewall, radius, motion, land_points)["speed_kt"]
        peak = torch.maximum(peak, field)
        gale = field >= THRESHOLDS_KT[0]
        first = torch.where(torch.isnan(first) & gale, torch.full_like(first, hour), first)
        for t_index, threshold in enumerate(THRESHOLDS_KT):
            hit = field >= threshold
            for l_index, lead in enumerate(lead_times_hours):
                if hour <= float(lead):
                    reach[:, :, t_index, l_index] |= hit
        north, east = _local_frame(points_lat.view(1, -1), points_lon.view(1, -1), lat.view(-1, 1), lon.view(-1, 1))
        rain = rain + rcliper_rate_mm_h(torch.sqrt(north ** 2 + east ** 2), vmax.view(-1, 1)) * step_hours
    probability = reach.double().mean(0).permute(0, 2, 1)                                       # (P, L, 3)
    arrived = torch.isfinite(first)
    arrival = torch.full((points_lat.numel(), 2), float("nan"), dtype=torch.float64)
    for p in range(points_lat.numel()):
        times = first[arrived[:, p], p]
        share = times.numel() / samples
        if share >= 0.1:
            arrival[p, 0] = torch.quantile(times, min(0.1 / share, 1.0))
        if share >= 0.5:
            arrival[p, 1] = torch.quantile(times, min(0.5 / share, 1.0))
    return {"probability": probability.float(), "arrival_hours": arrival.float(),
            "peak_kt": peak.median(0).values.float(),
            "rain_mm": torch.stack([rain.median(0).values, torch.quantile(rain, 0.9, dim=0)], -1).float(),
            "samples": samples}


@torch.no_grad()
def wind_probability_map(outputs: dict, batch_index: int, origin: tuple[float, float], lead_times_hours,
                         half_width_deg: float = 12.0, step_deg: float = 0.25, samples: int = 200, land_fn=None,
                         trained: dict | None = None, seed: int = 0) -> dict:
    """
    NHC-style wind-speed probability maps on a regular box around the storm: the chance of 34, 50 and 64 kt by
    every lead at every 0.25-degree point, plus gale arrival and storm-total rain -- the same Monte Carlo as
    :func:`wind_hazard`, on a grid. Returns ``{"latitude": (H,), "longitude": (W,), "probability": (H, W, L, 3),
    "arrival_hours": (H, W, 2), "rain_mm": (H, W, 2)}``.
    """
    lat0, lon0 = float(origin[0]), float(origin[1])
    lats = torch.arange(lat0 - half_width_deg, lat0 + half_width_deg + 1e-9, step_deg, dtype=torch.float64)
    lons = torch.arange(lon0 - half_width_deg, lon0 + half_width_deg + 1e-9, step_deg, dtype=torch.float64)
    lats = lats.clamp(-89.9, 89.9)
    grid_lat, grid_lon = torch.meshgrid(lats, lons, indexing="ij")
    result = wind_hazard(outputs, batch_index, origin, lead_times_hours, grid_lat.reshape(-1),
                         ((grid_lon.reshape(-1) + 180.0) % 360.0) - 180.0, samples=samples, land_fn=land_fn,
                         seed=seed, trained=trained)
    shape = (lats.numel(), lons.numel())
    return {"latitude": lats.float(), "longitude": (((lons + 180.0) % 360.0) - 180.0).float(),
            "probability": result["probability"].view(*shape, len(lead_times_hours), 3),
            "arrival_hours": result["arrival_hours"].view(*shape, 2), "rain_mm": result["rain_mm"].view(*shape, 2),
            "samples": samples}


def wind_radii_from_profile(vmax_kt: torch.Tensor, rmw_km: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric 34/50/64 kt radii (km) implied by a Holland profile: ``(..., 3)``, 0 where never reached."""
    return torch.stack([radius_of(torch.full_like(vmax_kt, t), vmax_kt, rmw_km, b) for t in THRESHOLDS_KT], -1)


__all__ = ["THRESHOLDS_KT", "fit_b", "holland_b", "holland_profile", "inflow_angle_deg", "land_decay_kt",
           "radius_of", "rcliper_rate_mm_h", "rmw_climatology_km", "sample_storm_paths", "wind_field", "wind_hazard",
           "wind_probability_map",
           "wind_radii_from_profile"]
