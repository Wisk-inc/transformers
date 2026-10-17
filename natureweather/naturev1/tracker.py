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

from .steering import great_circle_km, wrap_longitude

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


def _local_extreme(field: torch.Tensor, lat_axis, lon_axis, rows: int, cols: int, guess: torch.Tensor,
                   radius_km: float, largest: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """The grid point of the largest (or smallest) value within ``radius_km`` of each guess: ``(centre, value)``."""
    lat = lat_axis.view(1, -1, 1).expand(1, rows, cols).to(field.device)
    lon = lon_axis.view(1, 1, -1).expand(1, rows, cols).to(field.device)
    distance = great_circle_km(guess[:, 0].view(-1, 1, 1), guess[:, 1].view(-1, 1, 1), lat, lon)
    fill = -math.inf if largest else math.inf
    masked = torch.where(distance <= radius_km, field.view(-1, rows, cols), torch.full_like(field.view(-1, rows, cols), fill))
    flat = masked.flatten(1).argmax(-1) if largest else masked.flatten(1).argmin(-1)
    i, j = flat // cols, flat % cols
    value = masked.flatten(1).gather(1, flat.view(-1, 1))[:, 0]
    return torch.stack([lat_axis.to(field.device)[i], wrap_longitude(lon_axis.to(field.device)[j])], -1), value


def multi_fix_centre(state: torch.Tensor, variables, normalizer, grid, first_guess: torch.Tensor,
                     radius_km: float = 350.0, agree_km: float = 200.0) -> dict[str, torch.Tensor]:
    """
    The storm centre from several fields at once, as operational trackers fix it: the sea-level-pressure
    minimum, the 850 hPa cyclonic vorticity maximum and the 850 hPa height minimum. Fixes within ``agree_km``
    of the pressure fix are averaged (a single field's grid noise averages out); their scatter is reported
    as the fix's own uncertainty.

    Returns ``{"centre": (B, 2), "found": (B,), "spread_km": (B,), "fixes": (B,)}``.
    """
    lat_axis, lon_axis, rows, cols = _regular(grid)
    guess = first_guess.to(state.device, torch.float64)
    pressure_fix, _, found = vortex_centre(state, variables, normalizer, grid, guess, radius_km=radius_km)
    fixes = [pressure_fix]
    names = list(variables)
    if "u_component_of_wind@850" in names and "v_component_of_wind@850" in names:
        from .thermo import GridFields

        fields = GridFields(state, names, normalizer, grid)
        vorticity = fields.vorticity(850) * torch.where(guess[:, :1] >= 0, 1.0, -1.0)
        fixes.append(_local_extreme(vorticity, lat_axis, lon_axis, rows, cols, guess, radius_km, largest=True)[0])
    if "geopotential@850" in names:
        height = _field(state, names, normalizer, "geopotential@850").to(state.device)
        fixes.append(_local_extreme(height, lat_axis, lon_axis, rows, cols, guess, radius_km, largest=False)[0])
    stacked = torch.stack(fixes, 1)                                                   # (B, F, 2)
    near = great_circle_km(stacked[..., 0], stacked[..., 1], pressure_fix[:, :1], pressure_fix[:, 1:]) <= agree_km
    near[:, 0] = True
    weight = near.to(torch.float64)
    reference = pressure_fix[:, 1:]
    east = wrap_longitude(stacked[..., 1] - reference)
    centre_lat = (stacked[..., 0] * weight).sum(1) / weight.sum(1)
    centre_lon = wrap_longitude(reference[:, 0] + (east * weight).sum(1) / weight.sum(1))
    centre = torch.stack([centre_lat, centre_lon], -1)
    spread = (great_circle_km(stacked[..., 0], stacked[..., 1], centre[:, :1], centre[:, 1:]) * weight).sum(1) / weight.sum(1)
    found = found | (weight.sum(1) >= 2)
    return {"centre": centre, "found": found, "spread_km": spread, "fixes": weight.sum(1)}


@torch.no_grad()
def track_storm_in_rollout(states: torch.Tensor, variables, normalizer, grid, storm_center: torch.Tensor,
                           lead_times_hours, step_hours: float = 6.0, radius_km: float = 350.0) -> dict[str, torch.Tensor]:
    """
    Follow each storm through ``(S, B, P, C)`` forecast states with the multi-field fix, and read its intensity
    and structure as it goes: the lowest sea-level pressure and strongest 10 m wind within 300 km, and Hart's
    phase space (warm or cold core, thermal asymmetry). Ends (NaN) when the low is lost.

    Returns, at every lead: ``displacement (B, L, 2)`` degrees, ``pressure_hpa``, ``wind_kt`` (as the model's
    1.5-degree state resolves it -- a proxy, corrected statistically downstream), ``phase (B, L, 3)``.
    """
    from .thermo import GridFields, cyclone_phase

    centre = storm_center.to(states.device, torch.float64)
    origin, velocity = centre.clone(), torch.zeros_like(centre)
    alive = torch.isfinite(centre).all(-1)
    names = list(variables)
    path, pressure, wind, phase = [centre.clone()], [], [], []
    for step in range(states.shape[0]):
        guess = torch.stack([centre[:, 0] + velocity[:, 0], wrap_longitude(centre[:, 1] + velocity[:, 1])], -1)
        fix = multi_fix_centre(states[step], names, normalizer, grid, torch.nan_to_num(guess), radius_km=radius_km)
        alive = alive & fix["found"]
        moved = torch.stack([fix["centre"][:, 0] - centre[:, 0], wrap_longitude(fix["centre"][:, 1] - centre[:, 1])], -1)
        velocity = torch.where(alive.view(-1, 1), moved, velocity)
        centre = torch.where(alive.view(-1, 1), fix["centre"], torch.full_like(centre, float("nan")))
        path.append(centre.clone())
        fields = GridFields(states[step], names, normalizer, grid)
        distance = fields.distance_km(torch.nan_to_num(centre))
        near = distance <= 300.0
        msl = fields.field("mean_sea_level_pressure") / 100.0 if fields.has("mean_sea_level_pressure") else None
        speed = (torch.hypot(fields.field("10m_u_component_of_wind"), fields.field("10m_v_component_of_wind")) / 0.514444
                 if fields.has("10m_u_component_of_wind") and fields.has("10m_v_component_of_wind") else None)
        nan = torch.full_like(centre[:, 0], float("nan"))
        low = torch.where(near, msl, torch.full_like(msl, math.inf)).min(-1).values if msl is not None else nan
        peak = torch.where(near, speed, torch.full_like(speed, -math.inf)).max(-1).values if speed is not None else nan
        pressure.append(torch.where(alive, low, nan))
        wind.append(torch.where(alive, peak, nan))
        ph = cyclone_phase(fields, torch.nan_to_num(centre), velocity)
        phase.append(torch.where(alive.view(-1, 1), torch.stack([ph["phase_b"], ph["phase_vtl"], ph["phase_vtu"]], -1),
                                 torch.full((centre.shape[0], 3), float("nan"), dtype=torch.float64, device=centre.device)))
    path = torch.stack(path, 1)
    series = {"pressure_hpa": torch.stack(pressure, 1), "wind_kt": torch.stack(wind, 1), "phase": torch.stack(phase, 1)}
    displacement, values = [], {k: [] for k in series}
    for hours in lead_times_hours:
        exact = float(hours) / step_hours
        low_i = min(int(math.floor(exact)), path.shape[1] - 1)
        high_i = min(low_i + 1, path.shape[1] - 1)
        frac = exact - low_i if high_i != low_i else 0.0
        if frac == 0.0:
            high_i = low_i
        lat = path[:, low_i, 0] * (1 - frac) + path[:, high_i, 0] * frac
        dlon = (wrap_longitude(path[:, low_i, 1] - origin[:, 1]) * (1 - frac)
                + wrap_longitude(path[:, high_i, 1] - origin[:, 1]) * frac)
        displacement.append(torch.stack([lat - origin[:, 0], dlon], -1))
        step_index = max(min(int(round(exact)) - 1, states.shape[0] - 1), 0)
        for key, value in series.items():
            values[key].append(value[:, step_index])
    out = {"displacement": torch.stack(displacement, 1).float()}
    out.update({key: torch.stack(v, 1).float() for key, v in values.items()})
    return out


@torch.no_grad()
def rollout_member_tracks(model, batch: dict, stepper, storm_centers: torch.Tensor, lead_times_hours,
                          variables, precision: str = "bf16", members: int = 1, wind_kt: torch.Tensor | None = None,
                          size_km: torch.Tensor | None = None) -> dict:
    """
    Run the full-state rollout once for an initial time and derive every rollout member for every storm
    active at it.

    Args:
        batch: one initial condition (``analysis (1, T, P, C)``, ``calendar``, ``valid_time``), as for
            :func:`naturev1.state_forecast`.
        storm_centers: ``(S, 2)`` degrees -- every storm at that time; one rollout serves them all.
        members: rollout ensemble size. With more than one (and a model trained with ensemble noise), each
            member is tracked and their mean is the ``rollout_ensemble`` member, their scatter its spread.
        wind_kt, size_km: ``(S,)`` the storms' current wind and gale radius, for the intensity-dependent
            steering layer and size-dependent beta drift.

    Returns:
        ``{"rollout_steering", "rollout_vortex", "rollout_ensemble": (S, L, 2) degrees,
        "rollout_ensemble_spread_km": (S, L), "rollout_pressure_hpa", "rollout_wind_kt": (S, L),
        "rollout_phase": (S, L, 3)}``.
    """
    from .state import state_forecast
    from .steering import displacement_km, physics_steering_track

    steps = int(math.ceil(max(lead_times_hours) / stepper.step_hours))
    forecasts = state_forecast(model, batch, stepper, steps=steps, members=max(members, 1), precision=precision)
    count = storm_centers.shape[0]
    now = batch["analysis"][:, -1]
    first = forecasts[0]                                                              # (steps, 1, P, C)
    sequence = torch.cat([now.unsqueeze(0), first], 0).expand(-1, count, -1, -1).float()
    steering = physics_steering_track(sequence, variables, stepper.normalizer, stepper.grid, storm_centers,
                                      lead_times_hours, wind_kt=wind_kt, size_km=size_km, step_hours=stepper.step_hours)
    tracked = [track_storm_in_rollout(f.expand(-1, count, -1, -1).float(), variables, stepper.normalizer, stepper.grid,
                                      storm_centers, lead_times_hours, stepper.step_hours) for f in forecasts]
    main = tracked[0]
    out = {"rollout_steering": steering, "rollout_vortex": main["displacement"],
           "rollout_pressure_hpa": main["pressure_hpa"], "rollout_wind_kt": main["wind_kt"], "rollout_phase": main["phase"]}
    stacked = torch.stack([t["displacement"] for t in tracked])                       # (members, S, L, 2)
    have = torch.isfinite(stacked).all(-1)
    count_have = have.sum(0)
    mean = torch.nan_to_num(stacked).sum(0) / count_have.clamp_min(1).unsqueeze(-1)
    mean = torch.where((count_have >= max(1, (len(tracked) + 1) // 2)).unsqueeze(-1), mean, torch.full_like(mean, float("nan")))
    latitude = storm_centers[:, 0].to(mean.device).float()
    spread = (displacement_km(torch.nan_to_num(stacked - mean.unsqueeze(0)), latitude.view(1, -1, 1)).norm(dim=-1)
              * have).sum(0) / count_have.clamp_min(1)
    out["rollout_ensemble"] = mean
    out["rollout_ensemble_spread_km"] = torch.where(count_have > 1, spread, torch.full_like(spread, float("nan")))
    return out


class AnalogForecaster:
    """
    Historical look-alikes: where storms in the same place, moving the same way, at the same strength and time of
    year went next. A real forecast method (trajectory analogs predate the numerical models) and a member with
    errors unlike any dynamical one -- which is exactly what a consensus wants.

    Built from training-season tracks only. Every hemisphere in the poleward frame, so a southern storm finds
    northern analogs mirrored and vice versa. Prediction is a distance-weighted mean of the ``k`` nearest
    analogs' displacements and intensity changes at every lead.
    """

    FEATURES = ("abs_latitude", "lon_sin", "lon_cos", "motion_poleward", "motion_east", "wind", "doy_sin", "doy_cos")
    SCALES = (5.0, 0.15, 0.15, 1.0, 1.0, 25.0, 0.3, 0.3)

    def __init__(self, features: torch.Tensor, displacement: torch.Tensor, change: torch.Tensor, leads) -> None:
        self.features, self.displacement, self.change = features, displacement, change
        self.leads = tuple(int(h) for h in leads)

    @classmethod
    def from_tracks(cls, tracks, lead_times_hours, cadence_hours: int = 6) -> AnalogForecaster:
        import numpy as np

        from .storms import lead_track_points, storm_state_vector

        offsets = [int(h) // cadence_hours for h in lead_times_hours]
        feats, disps, changes = [], [], []
        for track in tracks:
            seconds = track.time.astype("datetime64[s]").astype(np.int64)
            for point in range(len(track)):
                if seconds[point] % (cadence_hours * 3600) or not np.isfinite(track.max_wind_kt[point]):
                    continue
                state = storm_state_vector(track, point, cadence_hours)
                lat, lon = float(track.latitude[point]), float(track.longitude[point])
                sign = 1.0 if lat >= 0 else -1.0
                doy = (seconds[point] % 31557600) / 31557600.0
                feats.append([abs(lat), math.sin(math.radians(lon)), math.cos(math.radians(lon)), state[2] * sign,
                              state[3], float(track.max_wind_kt[point]), math.sin(2 * math.pi * doy),
                              math.cos(2 * math.pi * doy)])
                ends = lead_track_points(track, point, offsets, cadence_hours)
                d, c = [], []
                for end in ends:
                    if end < 0:
                        d.append([float("nan"), float("nan")])
                        c.append(float("nan"))
                        continue
                    dlat = float(track.latitude[end] - track.latitude[point]) * sign
                    dlon = float((track.longitude[end] - track.longitude[point] + 180.0) % 360.0 - 180.0)
                    d.append([dlat, dlon])
                    c.append(float(track.max_wind_kt[end] - track.max_wind_kt[point]))
                disps.append(d)
                changes.append(c)
        return cls(torch.tensor(feats, dtype=torch.float32), torch.tensor(disps, dtype=torch.float32),
                   torch.tensor(changes, dtype=torch.float32), lead_times_hours)

    def state_dict(self) -> dict:
        return {"features": self.features, "displacement": self.displacement, "change": self.change, "leads": self.leads}

    @classmethod
    def from_state_dict(cls, state: dict) -> AnalogForecaster:
        return cls(state["features"], state["displacement"], state["change"], state["leads"])

    def query_features(self, storm_center: torch.Tensor, storm_state: torch.Tensor, wind_kt: torch.Tensor,
                       day_of_year: torch.Tensor) -> torch.Tensor:
        lat, lon = storm_center[:, 0].float(), storm_center[:, 1].float()
        sign = torch.where(lat >= 0, 1.0, -1.0)
        state = torch.nan_to_num(storm_state.float())
        return torch.stack([lat.abs(), torch.sin(torch.deg2rad(lon)), torch.cos(torch.deg2rad(lon)), state[:, 2] * sign,
                            state[:, 3], wind_kt.float(), torch.sin(2 * math.pi * day_of_year.float()),
                            torch.cos(2 * math.pi * day_of_year.float())], -1)

    @torch.no_grad()
    def predict(self, storm_center: torch.Tensor, storm_state: torch.Tensor, wind_kt: torch.Tensor,
                day_of_year: torch.Tensor, k: int = 30, chunk: int = 2048) -> dict[str, torch.Tensor]:
        """``{"displacement": (B, L, 2) degrees, "change_kt": (B, L)}``; NaN where no analog reached that lead."""
        scales = torch.tensor(self.SCALES)
        base = self.features / scales
        query = self.query_features(storm_center.cpu(), storm_state.cpu(), wind_kt.cpu(), day_of_year.cpu()) / scales
        sign = torch.where(storm_center[:, 0].cpu() >= 0, 1.0, -1.0)
        displacement, change = [], []
        for start in range(0, query.shape[0], chunk):
            distance = torch.cdist(query[start : start + chunk], base)
            near = distance.topk(min(k, base.shape[0]), largest=False)
            weight = 1.0 / (near.values + 0.1)                                         # (b, k)
            d = self.displacement[near.indices]                                         # (b, k, L, 2)
            c = self.change[near.indices]                                               # (b, k, L)
            ok = torch.isfinite(d).all(-1)
            w = weight.unsqueeze(-1) * ok
            total = w.sum(1)
            mean_d = (torch.nan_to_num(d) * w.unsqueeze(-1)).sum(1) / total.clamp_min(1e-9).unsqueeze(-1)
            mean_c = (torch.nan_to_num(c) * w).sum(1) / total.clamp_min(1e-9)
            enough = ok.sum(1) >= max(3, k // 5)
            mean_d = torch.where(enough.unsqueeze(-1), mean_d, torch.full_like(mean_d, float("nan")))
            mean_c = torch.where(enough, mean_c, torch.full_like(mean_c, float("nan")))
            displacement.append(mean_d)
            change.append(mean_c)
        displacement = torch.cat(displacement)
        displacement[..., 0] = displacement[..., 0] * sign.view(-1, 1)
        return {"displacement": displacement, "change_kt": torch.cat(change)}


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


__all__ = ["AnalogForecaster", "find_storms", "multi_fix_centre", "rollout_member_tracks", "track_storm_in_rollout",
           "track_vortex", "vortex_centre"]
