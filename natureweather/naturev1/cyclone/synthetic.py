# Copyright 2026 Nathan. Apache-2.0.
"""
Synthetic storms in hard situations -- Dorian's stall, ridge breakdowns, trough captures, loops, binary storms,
centre jumps -- with known truth, for testing the forecast system and for training on rare regimes.

The idea: the regimes that break track models are rare in the record (a category 5 parked for two days at
26 N happens once in decades), so a model trained on the record learns that storms keep moving. Synthetic cases
fix the sampling: each scenario prescribes how the large-scale flow *evolves* -- a subtropical high that
weakens and slides east, a trough that digs or fails to dig, a col that opens under the storm -- and the true
track is integrated through that evolving flow with beta drift (and, where the scenario has them, a second
vortex's induced motion or a centre reformation). The forecast system only ever sees the scenario at its
initial time and 12 hours earlier, exactly as with a real analysis -- so whether it anticipates the stall is a
fair test of what it can infer from the flow's structure and tendency.

Every scenario is built from a few analytic pieces on the storm's tangent plane:

* a uniform background current (the trades or the westerlies);
* Gaussian streamfunction highs and lows (ridges, troughs, monsoon gyres) whose amplitude, position and size
  change linearly in time;
* deep-layer winds turned into a vertical profile with a chosen shear, and geopotential in geostrophic balance
  (``Phi' = f0 psi``), so an analysis built from it looks like one (:func:`scenario_environment`);
* optional events: a binary partner (Fujiwhara rotation about the pair's centroid), a reformation jump.

:func:`implant_vortex` adds a balanced, warm-core vortex to any analysis, for turning scenarios into full model
inputs for training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fields import Environment, synthetic_environment
from .geometry import beta as beta_at, coriolis, great_circle_km, hemisphere, move, rotate, to_local


@dataclass
class Blob:
    """A Gaussian streamfunction feature: amplitude m^2/s (positive = anticyclone in the north), centre km, scale km,
    each changing linearly per hour."""

    amplitude: float
    x_km: float
    y_km: float
    scale_km: float
    d_amplitude: float = 0.0
    dx_kmh: float = 0.0
    dy_kmh: float = 0.0

    def at(self, hours: float) -> tuple[float, float, float, float]:
        amplitude = self.amplitude + self.d_amplitude * hours
        if amplitude * self.amplitude < 0:                  # a decaying high fades out; it does not become a low
            amplitude = 0.0
        return amplitude, self.x_km + self.dx_kmh * hours, self.y_km + self.dy_kmh * hours, self.scale_km


@dataclass
class Scenario:
    name: str
    lat0: float
    lon0: float
    background: tuple[float, float] = (0.0, 0.0)          # m/s
    blobs: list = field(default_factory=list)
    shear_ms: tuple[float, float] = (0.0, 0.0)             # 200 minus 850 hPa
    outer_ms: float = 6.0                                   # vortex outer strength (beta drift)
    wind_kt: float = 100.0
    partner: dict | None = None                             # {"x_km", "y_km", "gamma": circulation m^2/s}
    jump: dict | None = None                                # {"hour", "east_km", "north_km"}
    description: str = ""

    def flow(self, x_km, y_km, hours: float):
        """Deep-layer (u, v) m/s at tangent-plane points (km) at ``hours``."""
        u = np.full(np.shape(x_km), self.background[0], dtype=np.float64)
        v = np.full(np.shape(x_km), self.background[1], dtype=np.float64)
        for blob in self.blobs:
            a, cx, cy, s = blob.at(hours)
            dx, dy = (np.asarray(x_km) - cx) * 1000.0, (np.asarray(y_km) - cy) * 1000.0
            l2 = (s * 1000.0) ** 2
            g = a * np.exp(-0.5 * (dx * dx + dy * dy) / l2)
            u = u + g * dy / l2
            v = v - g * dx / l2
        return u, v

    def streamfunction(self, x_km, y_km, hours: float):
        psi = -self.background[0] * np.asarray(y_km) * 1000.0 + self.background[1] * np.asarray(x_km) * 1000.0
        for blob in self.blobs:
            a, cx, cy, s = blob.at(hours)
            dx, dy = (np.asarray(x_km) - cx), (np.asarray(y_km) - cy)
            psi = psi + a * np.exp(-0.5 * (dx * dx + dy * dy) / (s * s))
        return psi


def scenario_library() -> dict[str, Scenario]:
    """The hard regimes, each as a :class:`Scenario` (northern hemisphere unless stated)."""
    lib = {}
    # Dorian 2019: the high north of the storm weakens and slides east; the trough that should pick the storm up
    # stays far to the west for two days; the residual flow just cancels the storm's beta drift. West at 5 m/s,
    # slowing, a ~48-hour stall below 1 m/s, then a slow turn north as the trough finally arrives.
    lib["dorian_stall"] = Scenario(
        "dorian_stall", 26.5, -76.0, background=(1.09, -1.30),
        blobs=[Blob(8.0e6, 0.0, 800.0, 900.0, d_amplitude=-8.0e6 / 30.0, dx_kmh=10.0),
               Blob(-1.2e7, -3400.0, 0.0, 900.0, dx_kmh=10.0)],
        outer_ms=7.0, wind_kt=155.0,
        description="ridge to the north fades and slides east, trough held far west: steering cancels beta drift")
    lib["ridge_breakdown"] = Scenario(
        "ridge_breakdown", 22.0, -60.0, background=(-2.0, 0.0),
        blobs=[Blob(7.0e6, 0.0, 1000.0, 1100.0, d_amplitude=-4.0e4),
               Blob(-9.0e6, -2600.0, 1100.0, 900.0, dx_kmh=14.0, dy_kmh=-3.0)],
        description="steady westward motion until an approaching trough breaks the ridge and the storm recurves")
    lib["trough_capture"] = Scenario(
        "trough_capture", 28.0, -70.0, background=(2.0, 0.0),
        blobs=[Blob(-9.0e6, -1600.0, 900.0, 900.0, dx_kmh=18.0, dy_kmh=-3.0)],
        description="a deep trough sweeps in and accelerates the storm north-east")
    lib["straight_runner"] = Scenario(
        "straight_runner", 15.0, -45.0, background=(-5.0, 0.3),
        blobs=[Blob(5.0e6, 0.0, 1300.0, 1300.0)], description="steady trade-wind easterlies under a strong ridge")
    lib["loop"] = Scenario(
        "loop", 30.0, -72.0, background=(0.0, 0.0),
        blobs=[Blob(5.0e6, 700.0, 600.0, 700.0, dx_kmh=-9.0, dy_kmh=-7.0), Blob(-4.0e6, -600.0, -300.0, 700.0, dx_kmh=8.0)],
        outer_ms=5.0, description="weak, rotating steering between a migrating high and low: a loop")
    lib["fujiwhara"] = Scenario(
        "fujiwhara", 18.0, 135.0, background=(-3.0, 1.0), partner={"x_km": 600.0, "y_km": 200.0, "gamma": 4.0e7},
        description="a second storm 630 km east: the pair rotates cyclonically about their centroid")
    lib["reformation"] = Scenario(
        "reformation", 15.0, -63.0, background=(-4.5, 0.5), blobs=[Blob(4.0e6, 200.0, 1100.0, 1000.0)],
        shear_ms=(9.0, 2.0), wind_kt=45.0, outer_ms=4.0, jump={"hour": 9.0, "east_km": 55.0, "north_km": 20.0},
        description="a sheared tropical storm whose centre re-forms 60 km downshear (east-north-east), Dorian 2019's early life")
    lib["sharp_turn"] = Scenario(
        "sharp_turn", 20.0, -85.0, background=(-1.0, 0.0),
        blobs=[Blob(7.0e6, 900.0, 300.0, 800.0, dx_kmh=10.0, d_amplitude=-3.0e4)],
        description="a ridge retreating east turns a westward storm sharply north")
    lib["southern_recurve"] = Scenario(
        "southern_recurve", -18.0, 110.0, background=(-2.0, 0.0),
        blobs=[Blob(-7.0e6, 0.0, -1000.0, 1100.0, d_amplitude=4.0e4),
               Blob(9.0e6, -2600.0, -1100.0, 900.0, dx_kmh=14.0, dy_kmh=3.0)],
        description="southern-hemisphere mirror of the ridge breakdown (signs of the highs and lows flipped)")
    return lib


def truth_track(scenario: Scenario, hours, step_h: float = 1.0) -> np.ndarray:
    """``(T, 2)`` lat/lon of the scenario's storm at ``hours`` (kinematic steering + beta + events)."""
    hours = np.asarray(hours, np.float64)
    out = np.full((hours.size, 2), np.nan)
    lat, lon = scenario.lat0, scenario.lon0
    sign = float(hemisphere(scenario.lat0))
    partner = None if scenario.partner is None else np.array([scenario.partner["x_km"], scenario.partner["y_km"]], float)
    t, k = 0.0, 0
    jumped = False
    end = hours.max()
    while k < hours.size and hours[k] <= 0:
        out[k] = lat, lon
        k += 1
    while t < end - 1e-9:
        x, y = to_local(scenario.lat0, scenario.lon0, lat, lon)
        u, v = scenario.flow(x, y, t)
        b = beta_at(lat) / beta_at(20.0)
        speed = 1.6 * np.sqrt(max(b, 0.05)) * np.sqrt(scenario.outer_ms / 6.0)
        du, dv = rotate(0.0, sign * speed, sign * 40.0)
        u, v = float(u) + du, float(v) + dv
        if partner is not None:
            # Fujiwhara: each vortex is carried by the other's circulation, Gamma / (2 pi d) tangentially.
            rx, ry = (x - partner[0]) * 1000.0, (y - partner[1]) * 1000.0
            d2 = rx * rx + ry * ry + 1e8
            gamma = sign * scenario.partner["gamma"]
            u += -gamma * ry / (2 * np.pi * d2)
            v += gamma * rx / (2 * np.pi * d2)
            pu, pv = scenario.flow(partner[0], partner[1], t)
            partner = partner + np.array([float(pu) + gamma * ry / (2 * np.pi * d2), float(pv) - gamma * rx / (2 * np.pi * d2)]) * step_h * 3.6
        lat, lon = move(lat, lon, u * step_h * 3.6, v * step_h * 3.6)
        t += step_h
        if scenario.jump is not None and not jumped and t >= scenario.jump["hour"]:
            lat, lon = move(lat, lon, scenario.jump["east_km"], scenario.jump["north_km"])
            jumped = True
        while k < hours.size and hours[k] <= t + 1e-9:
            out[k] = float(lat), float(lon)
            k += 1
    return out


def scenario_environment(scenario: Scenario, hours: float = 0.0, half_lat: float = 30.0, half_lon: float = 45.0) -> Environment:
    """An analysis-like :class:`Environment` of the scenario at ``hours`` (no vortex in it)."""
    levels = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
    lat0, lon0 = scenario.lat0, scenario.lon0
    f0 = float(coriolis(lat0))

    def profile(p):
        return np.clip((850.0 - p) / 650.0, -0.3, 1.2)

    def u_fn(lat, lon, p):
        x, y = to_local(lat0, lon0, lat, lon)
        u, _ = scenario.flow(x, y, hours)
        return u + scenario.shear_ms[0] * (profile(p) - 0.5)

    def v_fn(lat, lon, p):
        x, y = to_local(lat0, lon0, lat, lon)
        _, v = scenario.flow(x, y, hours)
        return v + scenario.shear_ms[1] * (profile(p) - 0.5)

    env = synthetic_environment(lat0, lon0, half_lat, half_lon, u=u_fn, v=v_fn, levels=levels)
    lat_g, lon_g = np.meshgrid(env.lat, env.lon, indexing="ij")
    x, y = to_local(lat0, lon0, lat_g, lon_g)
    psi = scenario.streamfunction(x, y, hours)
    env.fields["z"] = env.fields["z"] + f0 * psi[None]
    env.fields["msl"] = env.fields["msl"] + 0.3 * 1.2 * f0 * psi
    env.time = np.datetime64("2019-09-01T12", "s") + np.timedelta64(int(hours * 3600), "s")
    return env


def scenario_case(scenario: Scenario, leads, history_hours=(-48, -42, -36, -30, -24, -18, -12, -6, 0)):
    """A :class:`naturev1.cyclone.data.Case` for the scenario: the track before t=0 from integrating backwards
    is not defined, so the history is the steady motion of the first hours mirrored back; truth is the scenario's."""
    from .data import Case

    ahead = truth_track(scenario, np.array([0.0, 6.0, 12.0]))
    e6, n6 = to_local(ahead[0, 0], ahead[0, 1], ahead[1, 0], ahead[1, 1])
    hist = {"lat": [], "lon": [], "wind": [], "pressure": []}
    for h in history_hours:
        la, lo = move(scenario.lat0, scenario.lon0, e6 * h / 6.0, n6 * h / 6.0)
        hist["lat"].append(float(la))
        hist["lon"].append(float(lo))
        hist["wind"].append(scenario.wind_kt)
        hist["pressure"].append(1010.0 - scenario.wind_kt * 0.8)
    truth = truth_track(scenario, np.asarray(leads, float))
    case = Case(storm_id=f"SYN-{scenario.name}", name=scenario.name.upper(), basin="NA" if scenario.lat0 >= 0 else "SI",
                time=np.datetime64("2019-09-01T12", "s"), lat=scenario.lat0, lon=scenario.lon0, wind_kt=scenario.wind_kt,
                pressure_hpa=1010.0 - scenario.wind_kt * 0.8, status="HU" if scenario.wind_kt >= 64 else "TS",
                history={k: np.asarray(v, float) for k, v in hist.items()},
                truth={"lat": truth[:, 0], "lon": truth[:, 1], "wind": np.full(len(leads), scenario.wind_kt),
                       "pressure": np.full(len(leads), np.nan), "valid": np.ones(len(leads), bool)})
    return case


def scenario_inputs(scenario: Scenario, leads):
    """``(case, environment)`` with the environment's ``previous`` fields from 12 hours earlier."""
    from .data import PREVIOUS_LEVELS

    case = scenario_case(scenario, leads)
    env = scenario_environment(scenario, 0.0)
    before = scenario_environment(scenario, -12.0)
    idx = [list(before.levels).index(p) for p in PREVIOUS_LEVELS]
    env.previous = {"u": before.fields["u"][idx], "v": before.fields["v"][idx],
                    "z500": before.fields["z"][list(before.levels).index(500)], "msl": before.fields["msl"]}
    env.previous_levels = PREVIOUS_LEVELS
    return case, env


def implant_vortex(env: Environment, lat0: float, lon0: float, vmax_ms: float, rmax_km: float = 40.0, b: float = 1.5,
                   top_hpa: float = 150.0) -> Environment:
    """
    A copy of ``env`` with a balanced, warm-core Holland vortex at (lat0, lon0): tangential wind decreasing with
    height to zero at ``top_hpa``, geopotential from gradient-wind balance at each level, and temperature from
    hydrostatic balance between levels -- a storm a 1.5-degree analysis would otherwise not hold.
    """
    sign = float(hemisphere(lat0))
    f = abs(float(coriolis(lat0)))
    lat_g, lon_g = np.meshgrid(env.lat, env.lon, indexing="ij")
    x, y = to_local(lat0, lon0, lat_g, lon_g)
    r = np.maximum(np.hypot(x, y), 1.0)
    fields = {k: v.copy() for k, v in env.fields.items()}
    radii = np.linspace(1.0, 3000.0, 3000)
    profile = vmax_ms * np.sqrt((rmax_km / radii) ** b * np.exp(1.0 - (rmax_km / radii) ** b))
    # Gradient balance: dPhi/dr = v^2 / r + f v  -> integrate inward from the outer edge.
    dphi = (profile ** 2 / (radii * 1000.0) + f * profile) * 1000.0 * (radii[1] - radii[0])
    phi = -np.cumsum(dphi[::-1])[::-1]
    phi -= phi[-1]
    levels = np.asarray(env.levels, np.float64)
    weight = np.clip((levels - top_hpa) / (1000.0 - top_hpa), 0.0, 1.0) ** 0.7
    vt = np.interp(r, radii, profile)
    deficit = np.interp(r, radii, phi)
    for k, w in enumerate(weight):
        fields["u"][k] += -sign * w * vt * y / r
        fields["v"][k] += sign * w * vt * x / r
        fields["z"][k] += w * deficit
    fields["msl"] = fields["msl"] + 1.2 * weight[np.argmax(levels)] * deficit
    # Hydrostatic temperature anomaly between levels: T' = -(p / R) dPhi'/dp -> warm where the deficit shrinks upward.
    for k in range(len(levels)):
        j = min(k + 1, len(levels) - 1) if levels[min(k + 1, len(levels) - 1)] != levels[k] else max(k - 1, 0)
        dp = (levels[j] - levels[k]) * 100.0
        if dp == 0:
            continue
        dphi_k = (weight[j] - weight[k]) * deficit
        fields["t"][k] += -(levels[k] * 100.0 / 287.0) * dphi_k / dp
    out = Environment(lat=env.lat, lon=env.lon, levels=env.levels, fields=fields, previous=env.previous, time=env.time)
    out.previous_levels = env.previous_levels
    return out


def stall_hours(track: np.ndarray, hours, speed_ms: float = 1.0) -> float:
    """Hours the track spends moving slower than ``speed_ms`` (6-hourly segments)."""
    hours = np.asarray(hours, np.float64)
    d = great_circle_km(track[:-1, 0], track[:-1, 1], track[1:, 0], track[1:, 1])
    dt = np.diff(hours)
    slow = d / np.maximum(dt, 1e-9) / 3.6 < speed_ms
    return float(dt[slow].sum())


__all__ = ["Blob", "Scenario", "implant_vortex", "scenario_case", "scenario_environment", "scenario_inputs",
           "scenario_library", "stall_hours", "truth_track"]
