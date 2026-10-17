# Copyright 2026 Nathan. Apache-2.0.
"""
The storm's environment, read off the atmosphere the model already holds: steering, shear, moisture, ocean.

A hurricane is carried by the flow around it. Average the wind through the depth of the troposphere over
a few hundred kilometres around the centre and you have, to within a couple of metres per second, where
the storm will be tomorrow; that deep-layer mean is what every track model since the 1970s has been
built on, and the National Hurricane Center's statistical baselines still use it. Whether it strengthens
is decided by four more numbers: the wind shear between 200 and 850 hPa (which tears the core apart),
the humidity of the middle troposphere (which feeds it or starves it), the ocean temperature (its fuel,
with 26.5 C the usual threshold), and what the storm is already doing. Those are the SHIPS predictors.

NatureV1's backbone pools the atmosphere around the storm into one vector and leaves the heads to work
all of this out from 22,000 labels. That is asking a lot of a small head. Here the same physics is
computed explicitly from the same state and handed to the heads as inputs, and the steering flow is also
integrated into a track of its own -- a physics-only member that needs no training at all. The learned
track head then predicts the *residual* from that steering track, which is a far easier target: beta
drift, the trough interaction, the things steering alone gets wrong.

Everything here works on the model's own state tensor, so it can be read from the analysis (what the
atmosphere is doing now) or from the model's rollout (what it will be doing), with no second data source.
"""

from __future__ import annotations

import math

import torch

from .model import STORM_ENVIRONMENT_FEATURES


EARTH_RADIUS_KM = 6371.0
METRES_PER_DEGREE = 111_195.0
#: Pressure levels of the deep-layer mean steering flow, weighted by the thickness each one stands for.
STEERING_LEVELS = (850, 700, 500, 400, 300, 250, 200)
#: The operational shear definition: the 200 hPa wind minus the 850 hPa wind.
SHEAR_LEVELS = (200, 850)
#: Mid-level moisture, the first available of these.
HUMIDITY_LEVELS = (700, 600, 500)
#: 26.5 C, the sea-surface temperature below which tropical cyclones struggle to form or hold together.
OCEAN_THRESHOLD_K = 299.65
#: Beta drift: the poleward-and-westward push every tropical cyclone gets from the change of the Coriolis
#: parameter with latitude. One to two metres per second, towards the north-west in the northern
#: hemisphere and the south-west in the southern. ``(u, v)`` in m/s for the northern hemisphere; the
#: meridional part flips sign south of the equator.
BETA_DRIFT_MS = (-1.0, 1.0)


def grid_latlon(grid) -> tuple[torch.Tensor, torch.Tensor]:
    """Latitude and longitude of every grid point, degrees, longitude in ``[-180, 180)``."""
    coords = grid.coords[..., :2].to(torch.float64)
    latitude = torch.rad2deg(coords[:, 0])
    longitude = (torch.rad2deg(coords[:, 1]) + 180.0) % 360.0 - 180.0
    return latitude, longitude


def great_circle_km(lat1, lon1, lat2, lon2) -> torch.Tensor:
    """Great-circle distance between two sets of points in degrees; broadcasts."""
    lat1, lon1, lat2, lon2 = (torch.deg2rad(x.to(torch.float64)) for x in (lat1, lon1, lat2, lon2))
    inner = (torch.sin((lat2 - lat1) / 2) ** 2
             + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * torch.asin(inner.clamp(0.0, 1.0).sqrt())


def wrap_longitude(longitude):
    return (longitude + 180.0) % 360.0 - 180.0


def advect(centre: torch.Tensor, u: torch.Tensor, v: torch.Tensor, hours: float) -> torch.Tensor:
    """
    Move ``(B, 2)`` positions in degrees by a wind ``(u, v)`` in m/s for ``hours``.

    The eastward step is divided by the cosine of the latitude at the middle of the step, so a storm at
    30N moving at 5 m/s covers the right number of degrees of longitude (a degree there is 96 km).
    """
    centre = centre.to(torch.float64)
    seconds = hours * 3600.0
    dlat = v.to(torch.float64) * seconds / METRES_PER_DEGREE
    mid = torch.deg2rad(centre[:, 0] + dlat / 2)
    dlon = u.to(torch.float64) * seconds / (METRES_PER_DEGREE * torch.cos(mid).clamp_min(0.05))
    latitude = (centre[:, 0] + dlat).clamp(-89.9, 89.9)
    return torch.stack([latitude, wrap_longitude(centre[:, 1] + dlon)], -1)


class _Reader:
    """Physical fields from the normalized state, area-averaged around each storm."""

    def __init__(self, state: torch.Tensor, variables, normalizer, grid) -> None:
        if state.dim() != 3:
            raise ValueError(f"state must be (B, P, C), got {tuple(state.shape)}")
        self.state, self.variables, self.normalizer = state, list(variables), normalizer
        self.latitude, self.longitude = (x.to(state.device) for x in grid_latlon(grid))
        self.area = grid.weights.to(state.device, torch.float64)
        self._decoded: dict[str, torch.Tensor] = {}

    def has(self, name: str) -> bool:
        return name in self.variables

    def field(self, name: str) -> torch.Tensor:
        """``(B, P)`` physical values of one channel."""
        if name not in self._decoded:
            values = self.state[..., self.variables.index(name)].to(torch.float64)
            self._decoded[name] = torch.as_tensor(self.normalizer.decode(values, name), dtype=torch.float64)
        return self._decoded[name]

    def levels(self, variable: str, wanted) -> list[int]:
        return [level for level in wanted if self.has(f"{variable}@{level}")]

    def weights(self, centre: torch.Tensor, inner_km: float, outer_km: float) -> torch.Tensor:
        """
        ``(B, P)`` normalized area weights over the ring ``inner_km <= d <= outer_km`` around each centre.

        A ring rather than a disc for the dynamical fields: the storm's own circulation at the centre is
        not its steering, and leaving out the inner 200 km is the cheap version of removing the vortex.
        A grid too coarse to put four points in the ring falls back to the disc, then to the nearest
        point, so the answer degrades gracefully rather than becoming NaN.
        """
        centre = centre.to(self.state.device, torch.float64)
        distance = great_circle_km(centre[:, :1], centre[:, 1:], self.latitude[None], self.longitude[None])
        ring = (distance >= inner_km) & (distance <= outer_km)
        disc = distance <= outer_km
        enough = ring.sum(-1, keepdim=True) >= 4
        chosen = torch.where(enough, ring, disc)
        nearest = torch.zeros_like(chosen).scatter_(1, distance.argmin(-1, keepdim=True), True)
        chosen = torch.where(chosen.any(-1, keepdim=True), chosen, nearest)
        weights = chosen.to(torch.float64) * self.area[None]
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)

    def mean(self, name: str, weights: torch.Tensor) -> torch.Tensor:
        return (torch.nan_to_num(self.field(name)) * weights).sum(-1)


def _thickness_weights(levels: list[int]) -> list[float]:
    """Trapezoid weights: each level stands for the half-layers on either side of it."""
    if len(levels) == 1:
        return [1.0]
    pressures = sorted(levels, reverse=True)
    weights = []
    for index, level in enumerate(pressures):
        above = pressures[index + 1] if index + 1 < len(pressures) else level
        below = pressures[index - 1] if index > 0 else level
        weights.append((below - above) / 2.0)
    total = sum(weights)
    ordered = dict(zip(pressures, weights))
    return [ordered[level] / total for level in levels]


def _steering_wind(reader: _Reader, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Deep-layer mean ``(u, v)`` over the ring, m/s, and whether any upper-level wind was available."""
    levels = [level for level in STEERING_LEVELS
              if reader.has(f"u_component_of_wind@{level}") and reader.has(f"v_component_of_wind@{level}")]
    if not levels:
        zeros = torch.zeros(weights.shape[0], dtype=torch.float64, device=weights.device)
        return zeros, zeros, False
    u = v = 0.0
    for level, weight in zip(levels, _thickness_weights(levels)):
        u = u + weight * reader.mean(f"u_component_of_wind@{level}", weights)
        v = v + weight * reader.mean(f"v_component_of_wind@{level}", weights)
    return u, v, True


def _closest_level(reader: _Reader, variable: str, target: int, window: int) -> int | None:
    levels = reader.levels(variable, range(target - window, target + window + 1))
    return min(levels, key=lambda level: abs(level - target)) if levels else None


@torch.no_grad()
def storm_environment(state: torch.Tensor, variables, normalizer, grid, storm_center: torch.Tensor,
                      inner_km: float = 200.0, outer_km: float = 800.0, core_km: float = 500.0,
                      chunk: int = 256) -> torch.Tensor:
    """
    The SHIPS-style predictors around each storm, scaled to order one, in
    :data:`naturev1.STORM_ENVIRONMENT_FEATURES` order: deep-layer steering ``(u, v)`` and 200-850 hPa
    shear ``(u, v, speed)`` over the 200-800 km ring, each in tens of m/s; mid-level specific humidity
    over the inner 500 km in units of 5 g/kg; sea-surface temperature (2 m temperature where the ocean is
    not observed) over the inner 500 km, as its distance from 26.5 C in units of 3 K; and an ``observed``
    flag that is 0 when the state carries no upper-level winds at all.

    Args:
        state: ``(B, P, C)`` normalized values on ``grid`` -- the last frame of an analysis window, or a
            rollout state.
        variables: the flat channel names of the state (``dataset.variables``).
        normalizer: the dataset's normalizer, to put channels back into physical units.
        storm_center: ``(B, 2)`` degrees.

    Returns:
        ``(B, len(STORM_ENVIRONMENT_FEATURES))`` float32, on the state's device.
    """
    rows = []
    for start in range(0, state.shape[0], chunk):
        centre = storm_center[start : start + chunk].to(state.device, torch.float64)
        sub = _Reader(state[start : start + chunk], variables, normalizer, grid)
        ring = sub.weights(centre, inner_km, outer_km)
        core = sub.weights(centre, 0.0, core_km)
        steer_u, steer_v, observed = _steering_wind(sub, ring)

        shear_u = shear_v = torch.zeros_like(steer_u)
        top = _closest_level(sub, "u_component_of_wind", SHEAR_LEVELS[0], 100)
        bottom = _closest_level(sub, "u_component_of_wind", SHEAR_LEVELS[1], 150)
        if top is not None and bottom is not None and top != bottom and sub.has(f"v_component_of_wind@{top}") \
                and sub.has(f"v_component_of_wind@{bottom}"):
            shear_u = sub.mean(f"u_component_of_wind@{top}", ring) - sub.mean(f"u_component_of_wind@{bottom}", ring)
            shear_v = sub.mean(f"v_component_of_wind@{top}", ring) - sub.mean(f"v_component_of_wind@{bottom}", ring)
        shear_speed = torch.hypot(shear_u, shear_v)

        humidity = torch.zeros_like(steer_u)
        level = next((lv for lv in HUMIDITY_LEVELS if sub.has(f"specific_humidity@{lv}")), None)
        if level is not None:
            humidity = sub.mean(f"specific_humidity@{level}", core) * 1000.0            # kg/kg -> g/kg

        temperature = torch.full_like(steer_u, OCEAN_THRESHOLD_K)
        if sub.has("2m_temperature"):
            temperature = sub.mean("2m_temperature", core)
        if sub.has("sea_surface_temperature"):
            sst = torch.nan_to_num(sub.field("sea_surface_temperature"))
            index = sub.variables.index("sea_surface_temperature")
            masked = list(getattr(normalizer, "masked", []))
            if index in masked and len(sub.variables) + masked.index(index) < state.shape[-1]:
                seen = sub.state[..., len(sub.variables) + masked.index(index)].to(torch.float64) > 0.5
            else:
                seen = sst > 200.0
            ocean = (seen.to(torch.float64) * core)
            share = ocean.sum(-1)
            over_sea = (sst * ocean).sum(-1) / share.clamp_min(1e-30)
            temperature = torch.where(share > 0.3, over_sea, temperature)

        rows.append(torch.stack([
            steer_u / 10.0, steer_v / 10.0, shear_u / 10.0, shear_v / 10.0, shear_speed / 10.0,
            humidity / 5.0, (temperature - OCEAN_THRESHOLD_K) / 3.0,
            torch.full_like(steer_u, 1.0 if observed else 0.0),
        ], -1))
    return torch.cat(rows).to(torch.float32)


def describe_environment(features: torch.Tensor) -> str:
    """One storm's environment in the units a forecaster reads."""
    f = [float(x) for x in features.flatten().tolist()]
    if len(f) != len(STORM_ENVIRONMENT_FEATURES):
        raise ValueError(f"expected {len(STORM_ENVIRONMENT_FEATURES)} features, got {len(f)}")
    if f[7] < 0.5:
        return "environment: no upper-level winds in the state; steering and shear unavailable"
    steer_kt = math.hypot(f[0], f[1]) * 10 / 0.514444
    heading = (math.degrees(math.atan2(f[0], f[1])) + 360.0) % 360.0
    return (f"steering {steer_kt:.0f} kt towards {heading:03.0f}deg | shear {f[4] * 10 / 0.514444:.0f} kt | "
            f"700 hPa humidity {f[5] * 5:.1f} g/kg | ocean {f[6] * 3 + OCEAN_THRESHOLD_K - 273.15:.1f} C"
            f"{' (below 26.5 C)' if f[6] < 0 else ''}")


@torch.no_grad()
def steering_track(states: torch.Tensor, variables, normalizer, grid, storm_center: torch.Tensor,
                   lead_times_hours, step_hours: float = 6.0, beta_drift_ms=BETA_DRIFT_MS,
                   inner_km: float = 200.0, outer_km: float = 800.0) -> torch.Tensor:
    """
    Carry each storm with the deep-layer steering flow, and return where it gets to at every lead.

    Args:
        states: ``(B, P, C)`` one state, used at every step (the flow as it is now, held steady), or
            ``(S, B, P, C)`` a sequence -- the analysis followed by a rollout -- where step ``k`` reads
            ``states[min(k, S - 1)]``, so the storm is steered by the atmosphere *at that time*.
        storm_center: ``(B, 2)`` degrees.
        lead_times_hours: the leads to report; need not be multiples of ``step_hours``.
        beta_drift_ms: the climatological beta drift added to the steering; ``None`` for pure advection.

    Returns:
        ``(B, L, 2)`` displacement from the start in degrees ``(d_lat, d_lon)``, the short way round --
        the same convention as the best-track targets, so it drops straight in as a track prior.
    """
    sequence = states if states.dim() == 4 else states.unsqueeze(0)
    leads = [float(h) for h in lead_times_hours]
    steps = int(math.ceil(max(leads) / step_hours)) if leads else 0
    centre = storm_center.to(sequence.device, torch.float64)
    origin = centre.clone()
    positions = [centre]
    for step in range(steps):
        reader = _Reader(sequence[min(step, sequence.shape[0] - 1)], variables, normalizer, grid)
        ring = reader.weights(centre, inner_km, outer_km)
        u, v, observed = _steering_wind(reader, ring)
        if observed and beta_drift_ms is not None:
            u = u + beta_drift_ms[0]
            v = v + beta_drift_ms[1] * torch.where(centre[:, 0] < 0, -1.0, 1.0)      # poleward, either way
        centre = advect(centre, u, v, step_hours) if observed else centre
        positions.append(centre)
    path = torch.stack(positions, 1)                                                    # (B, steps+1, 2)

    out = []
    for hours in leads:
        exact = hours / step_hours
        low = min(int(math.floor(exact)), steps)
        high = min(low + 1, steps)
        frac = exact - low if high != low else 0.0
        lat = path[:, low, 0] * (1 - frac) + path[:, high, 0] * frac
        dlon_low = wrap_longitude(path[:, low, 1] - origin[:, 1])
        dlon_high = wrap_longitude(path[:, high, 1] - origin[:, 1])
        out.append(torch.stack([lat - origin[:, 0], dlon_low * (1 - frac) + dlon_high * frac], -1))
    return torch.stack(out, 1).to(torch.float32)


def extrapolation_track(storm_state: torch.Tensor, lead_times_hours) -> torch.Tensor:
    """
    Straight-line extrapolation of the last twelve hours of motion: ``(B, L, 2)`` degrees.

    Reads the motion columns of :func:`naturev1.storm_state_vector`. The hardest baseline at 12-24 h,
    and the one a consensus leans on least past two days.
    """
    motion = storm_state[:, 2:4].to(torch.float32)
    hours = torch.tensor([float(h) for h in lead_times_hours], dtype=torch.float32, device=motion.device)
    return motion.unsqueeze(1) * (hours / 12.0).view(1, -1, 1)


def displacement_km(displacement: torch.Tensor, latitude: torch.Tensor) -> torch.Tensor:
    """``(..., 2)`` degrees into ``(..., 2)`` kilometres, east-west scaled by the latitude's cosine."""
    scale = torch.stack([torch.ones_like(latitude), torch.cos(torch.deg2rad(latitude)).clamp_min(0.05)], -1)
    while scale.dim() < displacement.dim():
        scale = scale.unsqueeze(-2)
    return displacement * scale * (METRES_PER_DEGREE / 1000.0)




# --------------------------------------------------------------------------------------------------
# 0.12: steering that depends on the storm, and beta drift that depends on its size
# --------------------------------------------------------------------------------------------------

#: Top of the steering layer by intensity (Velden and Leslie 1991): a shallow depression is carried by the
#: lower troposphere, a major hurricane by the whole depth of it. (wind threshold kt, top hPa)
STEERING_TOPS = ((50.0, 500), (90.0, 300), (1e9, 200))


def steering_top_hpa(wind_kt: torch.Tensor) -> torch.Tensor:
    """``(B,)`` the top of the steering layer for each storm's current wind."""
    top = torch.full_like(wind_kt, 200.0)
    for threshold, level in reversed(STEERING_TOPS):
        top = torch.where(wind_kt < threshold, torch.full_like(top, float(level)), top)
    return top


def beta_drift_ms(latitude: torch.Tensor, size_km: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Beta drift ``(u, v)`` m/s: north-westward in the north, south-westward in the south.

    Its speed grows with the storm's outer circulation -- the beta gyres a large storm spins up are stronger
    (Fiorino and Elsberry 1989) -- from 1 m/s for a 150 km gale radius to 2.5 m/s at 450 km and beyond, and
    scales with the square root of the planetary-vorticity gradient, which weakens toward the poles.
    """
    lat = latitude.to(torch.float64)
    size = torch.full_like(lat, 200.0) if size_km is None else size_km.to(lat.device, torch.float64)
    speed = 1.0 + 1.5 * ((size - 150.0) / 300.0).clamp(0.0, 1.0)
    speed = speed * (torch.cos(torch.deg2rad(lat)) / math.cos(math.radians(15.0))).clamp_min(0.2).sqrt()
    heading = torch.deg2rad(torch.where(lat >= 0, torch.full_like(lat, 320.0), torch.full_like(lat, 220.0)))
    return speed * torch.sin(heading), speed * torch.cos(heading)


def _layered_wind(reader: _Reader, weights: torch.Tensor, top: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Thickness-weighted mean wind from the lowest steering level up to each storm's own layer top."""
    levels = [lv for lv in STEERING_LEVELS
              if reader.has(f"u_component_of_wind@{lv}") and reader.has(f"v_component_of_wind@{lv}")]
    if not levels:
        zeros = torch.zeros(weights.shape[0], dtype=torch.float64, device=weights.device)
        return zeros, zeros, False
    u = torch.stack([reader.mean(f"u_component_of_wind@{lv}", weights) for lv in levels], -1)
    v = torch.stack([reader.mean(f"v_component_of_wind@{lv}", weights) for lv in levels], -1)
    total_u = torch.zeros_like(u[:, 0])
    total_v = torch.zeros_like(v[:, 0])
    for _, layer_top in STEERING_TOPS:
        chosen = [lv for lv in levels if lv >= layer_top] or levels
        thickness = torch.tensor(_thickness_weights(chosen), dtype=torch.float64, device=u.device)
        index = [levels.index(lv) for lv in chosen]
        mask = (top == float(layer_top)).to(torch.float64)
        total_u = total_u + mask * (u[:, index] * thickness).sum(-1)
        total_v = total_v + mask * (v[:, index] * thickness).sum(-1)
    return total_u, total_v, True


@torch.no_grad()
def physics_steering_track(states: torch.Tensor, variables, normalizer, grid, storm_center: torch.Tensor,
                           lead_times_hours, wind_kt: torch.Tensor | None = None, size_km: torch.Tensor | None = None,
                           step_hours: float = 6.0, inner_km: float = 200.0, outer_km: float = 800.0) -> torch.Tensor:
    """
    :func:`steering_track` with the physics it leaves out: the steering layer deepens with intensity
    (``wind_kt``, ``(B,)`` or ``(B, steps)`` along the forecast), and beta drift grows with size (``size_km``,
    the mean gale radius). Same output convention: ``(B, L, 2)`` degrees from the start.
    """
    sequence = states if states.dim() == 4 else states.unsqueeze(0)
    leads = [float(h) for h in lead_times_hours]
    steps = int(math.ceil(max(leads) / step_hours)) if leads else 0
    centre = storm_center.to(sequence.device, torch.float64)
    origin = centre.clone()
    batch = centre.shape[0]
    winds = (torch.full((batch, steps), 60.0, dtype=torch.float64, device=sequence.device) if wind_kt is None
             else wind_kt.to(sequence.device, torch.float64))
    if winds.dim() == 1:
        winds = winds.view(-1, 1).expand(-1, max(steps, 1))
    positions = [centre]
    for step in range(steps):
        reader = _Reader(sequence[min(step, sequence.shape[0] - 1)], variables, normalizer, grid)
        ring = reader.weights(centre, inner_km, outer_km)
        top = steering_top_hpa(winds[:, min(step, winds.shape[1] - 1)])
        u, v, observed = _layered_wind(reader, ring, top)
        if observed:
            bu, bv = beta_drift_ms(centre[:, 0], size_km)
            centre = advect(centre, u + bu, v + bv, step_hours)
        positions.append(centre)
    path = torch.stack(positions, 1)
    out = []
    for hours in leads:
        exact = hours / step_hours
        low = min(int(math.floor(exact)), steps)
        high = min(low + 1, steps)
        frac = exact - low if high != low else 0.0
        lat = path[:, low, 0] * (1 - frac) + path[:, high, 0] * frac
        dlon = wrap_longitude(path[:, low, 1] - origin[:, 1]) * (1 - frac) + wrap_longitude(path[:, high, 1] - origin[:, 1]) * frac
        out.append(torch.stack([lat - origin[:, 0], dlon], -1))
    return torch.stack(out, 1).to(torch.float32)


__all__ = ["BETA_DRIFT_MS", "STEERING_TOPS", "advect", "beta_drift_ms", "describe_environment", "displacement_km",
           "extrapolation_track", "grid_latlon", "great_circle_km", "physics_steering_track", "steering_top_hpa",
           "steering_track", "storm_environment"]
