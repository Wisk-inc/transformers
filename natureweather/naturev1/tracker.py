# Copyright 2026 Nathan. Apache-2.0.
"""
Following the storm in the model's own atmosphere: a vortex tracker, and the rollout track members.

The steering member carries the storm with the flow *as it is now*. Two things the red team found say that
is not enough. Storms that turned poleward from low latitudes (Otis, Rai) were held at the latitude they
started: a turn is a change in the flow, and the flow held steady never turns. And a track from steering
alone cannot know that the storm itself deepens or fills.

Every operational track centre answers this the same way: run the global model forward and follow the
storm through it. Here that is two members, both from the full-state rollout the backbone was trained for:

* ``rollout_steering`` -- the deep-layer steering, re-read from the *forecast* atmosphere at every step
  (:func:`naturev1.steering_track` given the rollout sequence), plus beta drift;
* ``rollout_vortex`` -- the low itself, followed through the forecast sea-level pressure: at each step,
  the pressure minimum within reach of where the storm was heading, refined to a fraction of a grid
  cell. When the low fills or leaves the search area, the member ends there (NaN), rather than jumping
  to a different system.

A 1.5-degree model resolves a hurricane's pressure field coarsely, so the vortex member is noisy for
small storms and good for large ones; the consensus weights decide, per lead, how much to trust it.

:func:`find_storms` uses the same machinery to propose storm centres where there is no advisory to give
one -- the satellite storm network then refines them (:mod:`naturev1.satnet`).
"""

from __future__ import annotations

import math

import torch

from .steering import great_circle_km, steering_track, wrap_longitude

EARTH_RADIUS_KM = 6371.0


def _regular(grid) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Latitude and longitude axes of a regular grid laid out latitude-major, in degrees."""
    coords = grid.coords[:, :2].to(torch.float64)
    lat = torch.rad2deg(coords[:, 0])
    lon = (torch.rad2deg(coords[:, 1]) + 360.0) % 360.0
    cols = int((lat == lat[0]).sum())
    rows = lat.numel() // cols
    if rows * cols != lat.numel():
        raise ValueError("the vortex tracker needs a regular latitude-longitude grid")
    return lat.view(rows, cols)[:, 0], lon.view(rows, cols)[0], rows, cols


def _field(state: torch.Tensor, variables, normalizer, name: str) -> torch.Tensor:
    return torch.as_tensor(normalizer.decode(state[..., list(variables).index(name)].double(), name))


def vortex_centre(state: torch.Tensor, variables, normalizer, grid, first_guess: torch.Tensor,
                  radius_km: float = 400.0, annulus_km: tuple[float, float] = (400.0, 800.0),
                  min_depth_hpa: float = 0.5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The sea-level-pressure low nearest each first guess.

    Args:
        state: ``(B, P, C)`` normalized state on a regular grid.
        first_guess: ``(B, 2)`` degrees.

    Returns:
        ``(centre (B, 2) degrees, depth (B,) hPa below the surrounding annulus, found (B,) bool)``. A low
        is found when the minimum is inside the search disc rather than on its rim, and at least
        ``min_depth_hpa`` deeper than its surroundings.
    """
    lat_axis, lon_axis, rows, cols = _regular(grid)
    lat_axis, lon_axis = lat_axis.to(state.device), lon_axis.to(state.device)
    pressure = _field(state, variables, normalizer, "mean_sea_level_pressure").to(state.device).view(-1, rows, cols) / 100.0
    lat = lat_axis.view(1, -1, 1).expand(1, rows, cols)
    lon = lon_axis.view(1, 1, -1).expand(1, rows, cols)
    guess = first_guess.to(state.device, torch.float64)
    distance = great_circle_km(guess[:, 0].view(-1, 1, 1), guess[:, 1].view(-1, 1, 1), lat, lon)
    inside = distance <= radius_km
    masked = torch.where(inside, pressure, torch.full_like(pressure, math.inf))
    flat = masked.flatten(1).argmin(-1)
    i, j = flat // cols, flat % cols
    batch = torch.arange(pressure.shape[0], device=state.device)
    lowest = pressure[batch, i, j]

    def neighbour(di, dj):
        return pressure[batch, (i + di).clamp(0, rows - 1), (j + dj) % cols]

    # Sub-grid refinement: the vertex of the parabola through the minimum and its two neighbours, per axis.
    def vertex(minus, plus):
        curvature = minus + plus - 2.0 * lowest
        return torch.where(curvature > 1e-9, 0.5 * (minus - plus) / curvature.clamp_min(1e-9),
                           torch.zeros_like(lowest)).clamp(-0.5, 0.5)

    lat_step = float(lat_axis[1] - lat_axis[0]) if rows > 1 else 1.0
    lon_step = float(lon_axis[1] - lon_axis[0]) if cols > 1 else 1.0
    centre_lat = lat_axis[i] + vertex(neighbour(-1, 0), neighbour(1, 0)) * lat_step
    centre_lon = wrap_longitude(lon_axis[j] + vertex(neighbour(0, -1), neighbour(0, 1)) * lon_step)
    ring = (distance >= annulus_km[0]) & (distance <= annulus_km[1])
    around = (pressure * ring).flatten(1).sum(-1) / ring.flatten(1).sum(-1).clamp_min(1)
    depth = around - lowest
    # A minimum on the rim of the search disc is a slope, not a low: it must be the lowest point of its own
    # 3 x 3 neighbourhood in the unmasked field.
    around_min = torch.stack([neighbour(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]).min(0).values
    found = torch.isfinite(lowest) & (depth >= min_depth_hpa) & (lowest <= around_min)
    return torch.stack([centre_lat, centre_lon], -1), depth, found


@torch.no_grad()
def track_vortex(states: torch.Tensor, variables, normalizer, grid, storm_center: torch.Tensor, lead_times_hours,
                 step_hours: float = 6.0, radius_km: float = 350.0) -> torch.Tensor:
    """
    Follow each storm's low through a sequence of states. ``(S, B, P, C)`` -> ``(B, L, 2)`` displacements in
    degrees, NaN once the low is lost. ``states[k]`` is the state ``(k + 1) * step_hours`` after now.

    The search for step k+1 is centred where the storm was heading -- its position at step k plus the
    motion over the last step -- so a fast storm is not lost and a slow one is not confused with a
    neighbour.
    """
    centre = storm_center.to(states.device, torch.float64)
    origin, velocity = centre.clone(), torch.zeros_like(centre)
    alive = torch.isfinite(centre).all(-1)
    path = [centre.clone()]
    for step in range(states.shape[0]):
        guess = torch.stack([centre[:, 0] + velocity[:, 0], wrap_longitude(centre[:, 1] + velocity[:, 1])], -1)
        found_at, _, found = vortex_centre(states[step], variables, normalizer, grid, torch.nan_to_num(guess),
                                           radius_km=radius_km)
        alive = alive & found
        moved = torch.stack([found_at[:, 0] - centre[:, 0], wrap_longitude(found_at[:, 1] - centre[:, 1])], -1)
        velocity = torch.where(alive.view(-1, 1), moved, velocity)
        centre = torch.where(alive.view(-1, 1), found_at.to(centre.dtype), torch.full_like(centre, float("nan")))
        path.append(centre.clone())
    path = torch.stack(path, 1)                                                     # (B, steps + 1, 2)
    out = []
    for hours in lead_times_hours:
        exact = float(hours) / step_hours
        low = min(int(math.floor(exact)), path.shape[1] - 1)
        high = min(low + 1, path.shape[1] - 1)
        frac = exact - low if high != low else 0.0
        if frac == 0.0:
            # Exactly on a step: read it alone -- interpolating towards a lost (NaN) next step would
            # poison a position that is perfectly known.
            high = low
        lat = path[:, low, 0] * (1 - frac) + path[:, high, 0] * frac
        dlon = (wrap_longitude(path[:, low, 1] - origin[:, 1]) * (1 - frac)
                + wrap_longitude(path[:, high, 1] - origin[:, 1]) * frac)
        out.append(torch.stack([lat - origin[:, 0], dlon], -1))
    return torch.stack(out, 1).to(torch.float32)


@torch.no_grad()
def rollout_member_tracks(model, batch: dict, stepper, storm_centers: torch.Tensor, lead_times_hours,
                          variables, precision: str = "bf16") -> dict:
    """
    Run the full-state rollout once for an initial time and derive both rollout members for every storm
    active at it.

    Args:
        batch: one initial condition (``analysis (1, T, P, C)``, ``calendar``, ``valid_time``), as for
            :func:`naturev1.state_forecast`.
        storm_centers: ``(S, 2)`` degrees -- every storm at that time; one rollout serves them all.

    Returns:
        ``{"rollout_steering": (S, L, 2), "rollout_vortex": (S, L, 2)}`` displacements in degrees.
    """
    from .state import state_forecast

    steps = int(math.ceil(max(lead_times_hours) / stepper.step_hours))
    forecast = state_forecast(model, batch, stepper, steps=steps, members=1, precision=precision)[0]   # (steps, 1, P, C)
    count = storm_centers.shape[0]
    now = batch["analysis"][:, -1]
    sequence = torch.cat([now.unsqueeze(0), forecast], 0).expand(-1, count, -1, -1).float()
    steering = steering_track(sequence, variables, stepper.normalizer, stepper.grid, storm_centers, lead_times_hours,
                              step_hours=stepper.step_hours)
    vortex = track_vortex(forecast.expand(-1, count, -1, -1).float(), variables, stepper.normalizer, stepper.grid,
                          storm_centers, lead_times_hours, stepper.step_hours)
    return {"rollout_steering": steering, "rollout_vortex": vortex}


@torch.no_grad()
def find_storms(state: torch.Tensor, variables, normalizer, grid, max_latitude: float = 40.0, min_latitude: float = 4.0,
                min_depth_hpa: float = 2.0, separation_km: float = 800.0) -> list[dict]:
    """
    Candidate tropical cyclones in one state ``(P, C)``: closed sea-level-pressure lows in the tropics,
    deeper than ``min_depth_hpa`` against the 400-800 km ring around them, with cyclonic 850 hPa
    vorticity where the state has the winds -- the classic objective detection criteria, at the
    resolution the model has. Each candidate is a first guess for the satellite storm network to refine,
    not a confirmed storm. Sorted deepest first.
    """
    lat_axis, lon_axis, rows, cols = _regular(grid)
    pressure = _field(state.unsqueeze(0), variables, normalizer, "mean_sea_level_pressure").view(rows, cols) / 100.0
    padded = torch.nn.functional.pad(pressure.view(1, 1, rows, cols), (1, 1, 0, 0), mode="circular")
    padded = torch.nn.functional.pad(padded, (0, 0, 1, 1), mode="replicate")[0, 0]
    window = padded.unfold(0, 3, 1).unfold(1, 3, 1).reshape(rows, cols, 9)
    is_min = pressure <= window.min(-1).values
    band = (lat_axis.abs().view(-1, 1) >= min_latitude) & (lat_axis.abs().view(-1, 1) <= max_latitude)
    candidates = torch.nonzero(is_min & band)
    vorticity = None
    names = list(variables)
    if "u_component_of_wind@850" in names and "v_component_of_wind@850" in names:
        u = _field(state.unsqueeze(0), names, normalizer, "u_component_of_wind@850").view(rows, cols)
        v = _field(state.unsqueeze(0), names, normalizer, "v_component_of_wind@850").view(rows, cols)
        dy = math.radians(float(lat_axis[1] - lat_axis[0])) * EARTH_RADIUS_KM * 1000.0
        dx = (math.radians(float(lon_axis[1] - lon_axis[0])) * EARTH_RADIUS_KM * 1000.0
              * torch.cos(torch.deg2rad(lat_axis)).clamp_min(0.05).view(-1, 1))
        dv_dx = (torch.roll(v, -1, 1) - torch.roll(v, 1, 1)) / (2 * dx)
        du_dy = torch.zeros_like(u)
        du_dy[1:-1] = (u[2:] - u[:-2]) / (2 * dy)
        vorticity = dv_dx - du_dy
    found = []
    for i, j in candidates.tolist():
        guess = torch.tensor([[float(lat_axis[i]), float(lon_axis[j])]], dtype=torch.float64)
        centre, depth, ok = vortex_centre(state.unsqueeze(0), names, normalizer, grid, guess, radius_km=300.0,
                                          min_depth_hpa=min_depth_hpa)
        if not bool(ok[0]):
            continue
        if vorticity is not None:
            hemisphere = 1.0 if float(lat_axis[i]) >= 0 else -1.0
            if float(vorticity[i, j]) * hemisphere < 2e-5:
                continue
        found.append({"latitude": float(centre[0, 0]), "longitude": float(centre[0, 1]), "depth_hpa": float(depth[0]),
                      "pressure_hpa": float(pressure[i, j])})
    found.sort(key=lambda c: -c["depth_hpa"])
    kept = []
    for candidate in found:
        if all(float(great_circle_km(torch.tensor(candidate["latitude"]), torch.tensor(candidate["longitude"]),
                                     torch.tensor(k["latitude"]), torch.tensor(k["longitude"]))) > separation_km
               for k in kept):
            kept.append(candidate)
    return kept


__all__ = ["find_storms", "rollout_member_tracks", "track_vortex", "vortex_centre"]
