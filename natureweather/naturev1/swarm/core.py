# Copyright 2026 Nathan. Apache-2.0.
"""
The swarm sandbox: a particle filter over possible futures of one storm.

Idea: Nathan (CorX Labs). Instead of computing one or two tracks, spawn a swarm of possible worlds, each with its own
slightly different physics, let every one of them play out, and let the storm itself decide which ones were right:
each new satellite or best-track fix re-weights the swarm, the worlds that disagree with what the storm actually did
are pruned, and the survivors are what the forecast is made of. The cone is the shape of the surviving swarm, not a
fixed statistical radius, and the sudden turn or stall that a single deterministic track misses is already being
explored by part of the swarm before it happens.

Every particle is one complete possible world, carrying its own

* **steering-layer depth** -- a continuous blend of the 850-700, 850-400 and 850-200 hPa layer means (how deep the
  storm reaches is one of the largest real uncertainties in its track);
* **beta drift** -- strength and angle around the drift of :mod:`naturev1.cyclone.beta`;
* **trend persistence** -- how far the steering flow keeps changing the way it changed over the last 12 hours
  (a ridge that was building, a trough that was coming);
* **climatological memory** -- how fast the flow's departure from climatology fades (when a climatology is given);
* **steering perturbations** -- coefficients on a set of streamfunction modes, a static part (the analysis error,
  which the fixes can learn) and a random walk that grows with lead (the error the future will add);
* **a motion residual** -- whatever the physics does not explain about how the storm is moving now, fading with a
  particle-specific time scale.

Guardrails, so that a swarm of millions cannot run away into impossible states: the perturbations are derived from a
streamfunction, so they are non-divergent and add no spurious convergence of vorticity; translation speed is capped
at 35 m/s; every parameter is bounded; latitudes stay inside the analysis box; and resampling adds only small,
shrinking jitter (regularised particle filter), so the swarm neither collapses onto one world nor wanders off.

Scale. All particles are integrated together as tensors (PyTorch, CPU or GPU) with one bilinear ``grid_sample``
per stage for every field they need, so the cost is linear in the swarm size: hundreds of thousands of worlds per
second on a laptop CPU and far more on a GPU. :meth:`Swarm.stream` goes past memory: it keeps drawing fresh worlds
from the filtered posterior in batches and accumulates only the summaries (mean track, cone, strike probabilities,
scenarios), so the swarm size is limited by time, not memory. These are lightweight steering-physics worlds, not
three-dimensional atmospheres; nobody can run billions of those.

Usage::

    swarm = Swarm.from_case(case, env, climatology, particles=100_000)   # launched 24 h before the latest fix
    swarm.assimilate_history(case)                                        # the fixes of the last 24 h re-weight it
    forecast = swarm.forecast(LEADS)                                      # survivors, launched from the latest fix
    forecast.mean_track(), forecast.cone_km(), forecast.scenarios(), forecast.strike_probability(lat, lon)

    # live: each new fix as it arrives
    swarm.advance_to(hours, ...); swarm.update(lat, lon, sigma_km=25.0)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..cyclone.geometry import EARTH_RADIUS_KM, OMEGA, wrap_longitude

#: Translation speeds above this are clipped (m/s).
SPEED_CAP_MS = 35.0
#: Integration step (hours).
STEP_HOURS = 3.0


@dataclass
class SwarmPriors:
    """Prior spread of every per-particle parameter."""

    layer_depth: float = 0.5                  # 0 = 850-700 hPa, 0.5 = 850-400, 1 = 850-200
    layer_sd: float = 0.22
    beta_log_sd: float = 0.35                 # beta-drift strength multiplier, log-normal
    beta_angle_sd_deg: float = 15.0
    persistence_max: float = 2.0              # trend persistence, uniform on [0, max] (in 12-hour changes)
    anomaly_tau_h: tuple[float, float] = (48.0, 240.0)
    static_ms: float = 1.2                    # steering analysis error per mode coefficient (m/s)
    growth_ms_per_sqrt_h: float = 0.22        # random-walk growth of the steering error (m/s per sqrt(hour))
    residual_ms: float = 1.5                  # motion the physics does not explain (m/s)
    residual_tau_h: tuple[float, float] = (12.0, 48.0)
    fix_sigma_km: float = 30.0                # likelihood width for a best-track fix
    launch_sigma_km: float = 10.0             # position spread when the swarm is (re)launched at a fix
    resample_ess: float = 0.5                 # resample when the effective sample size falls below this fraction
    roughen: float = 0.15                     # jitter after resampling, as a fraction of the parameter's spread
    depth_tau_h: float = 12.0                 # how fast each world's steering depth follows its target depth
    depth_walk_per_sqrt_h: float = 0.04       # and how much it wanders (depth units per sqrt(hour))
    wobble_ms: float = 1.0                    # trochoidal wobble: stationary spread (m/s)
    wobble_tau_h: float = 1.5                 # and memory (hours) -- sub-hourly fixes can lock onto it


def _modes(lat: np.ndarray, lon: np.ndarray, rows: int = 4, cols: int = 6, radius_km: float = 900.0):
    """
    Non-divergent velocity modes ``(K, H, W)``: Gaussian streamfunction blobs on a lattice over the box, turned into
    winds on the sphere (``u = -(1/a) dpsi/dphi``, ``v = (1/(a cos phi)) dpsi/dlambda``) with the same centred
    differences the divergence would use, so their discrete divergence vanishes. Scaled so that unit coefficients
    give a velocity-component spread of about 1 m/s.
    """
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    km_per_deg = math.pi * EARTH_RADIUS_KM / 180.0
    lat_c = np.linspace(lat[0], lat[-1], rows + 2)[1:-1]
    lon_c = np.linspace(lon[0], lon[-1], cols + 2)[1:-1]
    dphi, dlam = np.radians(lat[1] - lat[0]), np.radians(lon[1] - lon[0])
    cos = np.maximum(np.cos(np.radians(la)), 1e-3)
    us, vs = [], []
    for a in lat_c:
        for b in lon_c:
            x = (lo - b) * km_per_deg * np.cos(np.radians(a))
            y = (la - a) * km_per_deg
            psi = np.exp(-(x * x + y * y) / (2.0 * radius_km ** 2))
            us.append(-np.gradient(psi, dphi, axis=0))
            vs.append(np.gradient(psi, dlam, axis=1) / cos)
    u, v = np.stack(us), np.stack(vs)
    scale = math.sqrt(0.5 * float(np.mean((u ** 2 + v ** 2).sum(0))))
    return u / scale, v / scale


_LAYER_NAMES = (("shal", 700, 850), ("med", 400, 850), ("dlm", 200, 850))


class SwarmFields:
    """Every gridded field the swarm samples, stacked for one ``grid_sample`` per evaluation."""

    def __init__(self, lat, lon, shallow, medium, deep, before=None, climate=None, beta_gradient=None,
                 outer_ms: float = 6.0, drift_ms: float = 1.6, angle_deg: float = 40.0, device="cpu",
                 dtype=torch.float32, modes=None, forecast=None) -> None:
        self.lat = np.asarray(lat, np.float64)
        self.lon = np.asarray(lon, np.float64)
        self.device, self.dtype = torch.device(device), dtype
        self.has_before, self.has_climate = before is not None, climate is not None
        self.outer_ms, self.drift_ms, self.angle_deg = float(outer_ms), float(drift_ms), float(angle_deg)
        zero = np.zeros_like(medium[0])
        mode_u, mode_v = modes if modes is not None else _modes(self.lat, self.lon)
        self.n_modes = mode_u.shape[0]
        gx, gy = beta_gradient if beta_gradient is not None else (zero, zero)
        channels = [shallow[0], shallow[1], medium[0], medium[1], deep[0], deep[1],
                    (before or medium)[0], (before or medium)[1], (climate or medium)[0], (climate or medium)[1],
                    gx * 1e11, gy * 1e11, *mode_u, *mode_v]
        self.base_channels = len(channels)
        self.forecast_hours, self.n_sources, self.source_names = None, 0, []
        if forecast is not None:
            # dynamical models' forecast steering (naturev1.cyclone.forecastflow.ForecastFlow, one or several): six
            # channels per forecast hour per model; every world follows one model, drawn with equal prior probability
            flows = list(forecast.values()) if isinstance(forecast, dict) else (list(forecast) if isinstance(forecast, (list, tuple)) else [forecast])
            self.source_names = list(forecast.keys()) if isinstance(forecast, dict) else [getattr(f, "source", str(i)) for i, f in enumerate(flows)]
            self.forecast_hours = np.asarray(flows[0].hours, np.float64)
            self.n_sources = len(flows)
            for flow in flows:
                data = np.asarray(flow.data, np.float32)
                if data.shape[0] != self.forecast_hours.size or not np.allclose(flow.hours, self.forecast_hours):
                    # a model on other forecast hours: interpolated onto the first model's
                    data = np.stack([np.concatenate([np.stack(flow.field(h, name)) for name, _, _ in _LAYER_NAMES])
                                     for h in self.forecast_hours]).astype(np.float32)
                channels.extend(data.reshape(-1, *np.shape(medium[0])))
        stack = np.nan_to_num(np.stack(channels).astype(np.float32))
        self.grid = torch.as_tensor(stack, dtype=dtype, device=self.device)[None]          # (1, C, H, W)
        self._base_index = torch.arange(self.base_channels, device=self.device)

    @classmethod
    def from_context(cls, ctx, device="cpu", dtype=torch.float32, forecast=None) -> "SwarmFields":
        """From a :class:`naturev1.cyclone.members.PhysicsContext` (vortex removed, flows built)."""
        f = ctx.flows
        before = (ctx.before.u, ctx.before.v) if ctx.before is not None else None
        climate = ctx.climate_flows.get("med")
        climate = (climate.u, climate.v) if climate is not None else None
        return cls(ctx.clean.lat, ctx.clean.lon, (f["shal"].u, f["shal"].v), (f["med"].u, f["med"].v),
                   (f["dlm"].u, f["dlm"].v), before, climate, (ctx.beta.gx, ctx.beta.gy), ctx.outer_ms,
                   ctx.beta.drift_ms, ctx.beta.angle_deg, device, dtype, forecast=forecast)

    def to_box(self, lon) -> np.ndarray:
        """Longitudes in the box's continuous frame."""
        return self.lon[0] + (np.asarray(lon, np.float64) - self.lon[0]) % 360.0

    def sample(self, lat: torch.Tensor, lon: torch.Tensor, channels: torch.Tensor | None = None) -> torch.Tensor:
        """``(C, N)`` bilinear values at the particles (outside the box: clamped to its edge) -- the analysis
        channels, or the given ``channels``."""
        x = 2.0 * (lon - self.lon[0]) / (self.lon[-1] - self.lon[0]) - 1.0
        y = 2.0 * (lat - self.lat[0]) / (self.lat[-1] - self.lat[0]) - 1.0
        g = torch.stack([x, y], -1).to(self.dtype)[None, None]                               # (1, 1, N, 2)
        grid = self.grid[:, self._base_index if channels is None else channels]
        return F.grid_sample(grid, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0]

    def forecast_steering(self, lat: torch.Tensor, lon: torch.Tensor, hours: float,
                          source: torch.Tensor | None = None) -> torch.Tensor | None:
        """``(6, N)`` forecast layer winds (shal u, v, med u, v, dlm u, v) at ``hours`` -- each particle's from the
        model ``source`` ``(N,)`` it follows -- or None without a forecast."""
        if self.forecast_hours is None:
            return None
        fh = self.forecast_hours
        h = float(np.clip(hours, fh[0], fh[-1]))
        k = int(np.clip(np.searchsorted(fh, h, side="right") - 1, 0, fh.size - 2))
        a = (h - fh[k]) / max(fh[k + 1] - fh[k], 1e-9)
        per_source = 6 * fh.size
        if self.n_sources == 1 or source is None:
            base = self.base_channels
            idx = torch.arange(base + 6 * k, base + 6 * k + 12, device=self.device)
            s = self.sample(lat, lon, idx)
            return (1.0 - a) * s[:6] + a * s[6:]
        idx = torch.cat([torch.arange(self.base_channels + j * per_source + 6 * k, self.base_channels + j * per_source + 6 * k + 12,
                                      device=self.device) for j in range(self.n_sources)])
        s = self.sample(lat, lon, idx).reshape(self.n_sources, 12, -1)                       # (S, 12, N)
        pick = source.to(torch.long).clamp(0, self.n_sources - 1)
        s = s[pick, :, torch.arange(s.shape[-1], device=self.device)].T                      # (12, N)
        return (1.0 - a) * s[:6] + a * s[6:]


@dataclass
class SwarmForecast:
    """The surviving worlds at each lead: ``positions`` ``(N, L, 2)`` (lat, lon), ``weights`` ``(N,)``."""

    leads: list
    positions: np.ndarray
    weights: np.ndarray
    track: np.ndarray | None = None                 # (N, T, 2) every STEP_HOURS, for strike maps and scenarios
    track_hours: np.ndarray | None = None
    track_depth: np.ndarray | None = None           # (N, T) steering depth: with ``track``, the 4D worldlines
    hemisphere: float = 1.0
    meta: dict = field(default_factory=dict)

    def mean_track(self) -> np.ndarray:
        w = self.weights / self.weights.sum()
        lat = (self.positions[..., 0] * w[:, None]).sum(0)
        lon = (self.positions[..., 1] * w[:, None]).sum(0)
        return np.stack([lat, wrap_longitude(lon)], -1)

    def distances_km(self, track: np.ndarray | None = None) -> np.ndarray:
        """``(N, L)`` great-circle distance of every world from ``track`` (default the mean)."""
        ref = self.mean_track() if track is None else track
        return _haversine_np(self.positions[..., 0], self.positions[..., 1], ref[None, :, 0], ref[None, :, 1])

    def cone_km(self, coverage: float = 0.67) -> np.ndarray:
        """Radius around the mean track holding ``coverage`` of the swarm's weight, per lead."""
        d = self.distances_km()
        return np.array([_weighted_quantile(d[:, k], self.weights, coverage) for k in range(d.shape[1])])

    def strike_probability(self, grid_lat, grid_lon, radius_km: float = 111.0, until_h: float | None = None) -> np.ndarray:
        """Probability that the centre passes within ``radius_km`` of each point of a regular grid
        (``(len(grid_lat), len(grid_lon))``), up to ``until_h`` hours."""
        track = self.track if self.track is not None else self.positions
        hours = self.track_hours if self.track is not None else np.asarray(self.leads, float)
        if until_h is not None:
            track = track[:, hours <= until_h + 1e-9]
        glat = np.asarray(grid_lat, np.float64)
        glon = np.asarray(grid_lon, np.float64)
        glon_box = glon[0] + (glon - glon[0]) % 360.0 if np.all(np.diff(glon) > 0) else glon
        shift = 360.0 * np.round((np.nanmedian(track[..., 1]) - glon_box.mean()) / 360.0)
        return _strike_counts(track, glat, glon_box + shift, radius_km, self.weights / self.weights.sum())

    def scenarios(self) -> dict:
        """Weight and mean track of each kind of future the swarm contains: ``loop``, ``stall``, ``recurve``,
        ``steady``."""
        track = self.track if self.track is not None else self.positions
        hours = self.track_hours if self.track is not None else np.asarray(self.leads, float)
        labels = classify_tracks(track, hours, self.hemisphere)
        w = self.weights / self.weights.sum()
        out = {}
        for name in ("loop", "stall", "recurve", "steady"):
            mask = labels == name
            p = float(w[mask].sum())
            if p <= 0:
                out[name] = {"probability": 0.0, "mean_track": None}
                continue
            ww = w[mask] / p
            pos = self.positions[mask]
            out[name] = {"probability": p,
                         "mean_track": np.stack([(pos[..., 0] * ww[:, None]).sum(0),
                                                 wrap_longitude((pos[..., 1] * ww[:, None]).sum(0))], -1)}
        return out


def classify_tracks(track: np.ndarray, hours: np.ndarray, hemisphere: float = 1.0) -> np.ndarray:
    """
    Label every track (``(N, T, 2)``) by the kind of future it is:

    * ``loop`` -- the heading turns through more than 300 degrees;
    * ``stall`` -- some 48-hour window with less than 250 km of net motion (about 1.4 m/s);
    * ``recurve`` -- moving westward over the first day and eastward and poleward over the last;
    * ``steady`` -- everything else.
    """
    lat, lon = track[..., 0], track[..., 1]
    n, t = lat.shape
    dt = float(np.median(np.diff(hours))) if t > 1 else STEP_HOURS
    east = np.diff(lon, axis=1) * np.cos(np.radians(0.5 * (lat[:, 1:] + lat[:, :-1])))
    north = np.diff(lat, axis=1) * hemisphere
    heading = np.unwrap(np.arctan2(north, east), axis=1)
    turning = np.abs(np.degrees(heading[:, -1] - heading[:, 0])) if t > 2 else np.zeros(n)
    labels = np.full(n, "steady", dtype=object)
    w = max(int(round(48.0 / dt)), 1)
    stall = np.zeros(n, bool)
    if t > w:
        net = _haversine_np(lat[:, w:], lon[:, w:], lat[:, :-w], lon[:, :-w])
        stall = (net < 250.0).any(1)
    k1 = max(int(round(24.0 / dt)), 1)
    early_east = east[:, :k1].sum(1)
    late_east, late_north = east[:, -k1:].sum(1), north[:, -k1:].sum(1)
    recurve = (early_east < 0) & (late_east > 0) & (late_north > 0)
    labels[recurve] = "recurve"
    labels[stall] = "stall"
    labels[turning > 300.0] = "loop"
    return labels


class Swarm:
    """
    A swarm of ``particles`` possible worlds for one storm (see the module docstring).

    Args:
        fields: :class:`SwarmFields` of the analysis.
        lat, lon: where the swarm starts.
        start_hours: the time of that start relative to the analysis (``-24`` for a swarm launched a day before the
            latest fix, so that the fixes since can filter it).
        wind_kt: current intensity, for the prior on the steering-layer depth.
    """

    def __init__(self, fields: SwarmFields, lat: float, lon: float, start_hours: float = 0.0, particles: int = 20000,
                 wind_kt: float = 65.0, priors: SwarmPriors | None = None, seed: int = 0, step_hours: float = STEP_HOURS,
                 record: int = 0) -> None:
        self.fields, self.p = fields, priors or SwarmPriors()
        self.n = int(particles)
        self.step_hours = float(step_hours)
        self.depth_target = depth_for_wind(wind_kt)
        self.record = int(record)
        self.worldlines: list = []
        self.dev, self.dt = fields.device, torch.float64
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.hours = float(start_hours)
        self.hemisphere = 1.0 if lat >= 0 else -1.0
        self.params = self._draw_prior(self.n, wind_kt)
        self.lat, self.lon = self._launch(lat, float(fields.to_box(lon)), self.n)
        self.logw = torch.zeros(self.n, dtype=self.dt, device=self.dev)
        self.history: list[dict] = []
        self.wind_kt = wind_kt

    # ------------------------------------------------------------------------------------------------ set-up --

    @classmethod
    def from_case(cls, case, env, climatology=None, particles: int = 20000, window_h: float = 24.0,
                  priors: SwarmPriors | None = None, device="cpu", seed: int = 0) -> "Swarm":
        """A swarm launched at the fix ``window_h`` hours before the case (or at the case if there is none)."""
        from ..cyclone.members import build_context

        ctx = build_context(case, env, climatology)
        if ctx is None:
            raise ValueError("the storm has no motion history")
        return cls.from_context(case, ctx, particles, window_h, priors, device, seed)

    @classmethod
    def from_context(cls, case, ctx, particles: int = 20000, window_h: float = 24.0, priors: SwarmPriors | None = None,
                     device="cpu", seed: int = 0, fields: "SwarmFields | None" = None) -> "Swarm":
        """As :meth:`from_case`, from an already built :class:`naturev1.cyclone.members.PhysicsContext`."""
        from ..cyclone.data import HISTORY_HOURS

        fields = fields or SwarmFields.from_context(ctx, device=device)
        start, la, lo = 0.0, case.lat, case.lon
        for h in sorted(HISTORY_HOURS):
            k = HISTORY_HOURS.index(h)
            if -window_h - 1e-9 <= h < 0 and np.isfinite(case.history["lat"][k]):
                start, la, lo = float(h), float(case.history["lat"][k]), float(case.history["lon"][k])
                break
        if priors is None:
            priors = SwarmPriors(layer_depth=depth_for_wind(case.wind_kt))
        swarm = cls(fields, la, lo, start, particles, case.wind_kt, priors, seed)
        swarm.context = ctx
        return swarm

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, dtype=self.dt).to(self.dev)

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, dtype=self.dt).to(self.dev)

    def _draw_prior(self, n: int, wind_kt: float) -> dict:
        p, k = self.p, self.fields.n_modes
        lo, hi = p.anomaly_tau_h
        rlo, rhi = p.residual_tau_h
        bias = p.layer_sd * self._randn(n)
        return {
            "depth_bias": bias,
            "depth": (p.layer_depth + bias).clamp(0.0, 1.0),
            "wob_u": p.wobble_ms * self._randn(n),
            "wob_v": p.wobble_ms * self._randn(n),
            "beta": torch.exp(p.beta_log_sd * self._randn(n)).clamp(0.3, 3.0),
            "angle": (p.beta_angle_sd_deg * self._randn(n)).clamp(-40.0, 40.0),
            "persist": p.persistence_max * self._rand(n),
            "anom_tau": lo + (hi - lo) * self._rand(n),
            "static": p.static_ms * self._randn(n, k),
            "res_u": p.residual_ms * self._randn(n),
            "res_v": p.residual_ms * self._randn(n),
            "res_tau": rlo + (rhi - rlo) * self._rand(n),
            "walk": torch.zeros(n, k, dtype=self.dt, device=self.dev),
            # which model's forecast winds this world follows (several models: equal prior, the multi-model idea)
            "source": torch.randint(0, max(self.fields.n_sources, 1), (n,), generator=self.gen).to(self.dev)
            if self.fields.n_sources > 1 else torch.zeros(n, dtype=torch.long, device=self.dev),
        }

    def _launch(self, lat: float, lon_box: float, n: int):
        s = self.p.launch_sigma_km / (math.pi * EARTH_RADIUS_KM / 180.0)
        la = lat + s * self._randn(n)
        lo = lon_box + s * self._randn(n) / max(math.cos(math.radians(lat)), 0.05)
        return la, lo

    # ------------------------------------------------------------------------------------------------ physics --

    def velocity(self, lat: torch.Tensor, lon: torch.Tensor, hours: float, prm: dict | None = None):
        """``(u, v)`` m/s of every particle at ``hours`` relative to the analysis."""
        prm = self.params if prm is None else prm
        s = self.fields.sample(lat, lon).to(self.dt)
        d = prm["depth"]
        w_sh = (1.0 - 2.0 * d).clamp(min=0.0)
        w_dp = (2.0 * d - 1.0).clamp(min=0.0)
        w_md = 1.0 - w_sh - w_dp
        ff = self.fields.forecast_steering(lat, lon, hours, prm.get("source")) if hours >= 0 else None
        if ff is not None:
            # the model's forecast winds already evolve: no tendency or climatology terms on top of them
            ff = ff.to(self.dt)
            u = w_sh * ff[0] + w_md * ff[2] + w_dp * ff[4]
            v = w_sh * ff[1] + w_md * ff[3] + w_dp * ff[5]
        else:
            u = w_sh * s[0] + w_md * s[2] + w_dp * s[4]
            v = w_sh * s[1] + w_md * s[3] + w_dp * s[5]
        if ff is None and self.fields.has_before:
            if hours < 0:
                g = torch.full_like(d, hours / 12.0)
            else:
                tau = 12.0 * prm["persist"].clamp(min=1e-3)
                g = tau * (1.0 - torch.exp(-hours / tau)) / 12.0
            u = u + g * (s[2] - s[6])
            v = v + g * (s[3] - s[7])
        if ff is None and self.fields.has_climate and hours > 0:
            m = torch.exp(-hours / prm["anom_tau"])
            u = s[8] + m * (u - s[8])
            v = s[9] + m * (v - s[9])
        k = self.fields.n_modes
        coef = prm["static"] + prm["walk"]                                                   # (N, K)
        u = u + (coef * s[12:12 + k].T).sum(1)
        v = v + (coef * s[12 + k:12 + 2 * k].T).sum(1)
        bu, bv = self._beta(lat, s[10] * 1e-11, s[11] * 1e-11, prm)
        decay = 1.0 if hours <= 0 else torch.exp(-hours / prm["res_tau"])
        u = u + bu + prm["res_u"] * decay + prm["wob_u"]
        v = v + bv + prm["res_v"] * decay + prm["wob_v"]
        speed = torch.hypot(u, v)
        scale = torch.where(speed > SPEED_CAP_MS, SPEED_CAP_MS / speed.clamp(min=1e-9), torch.ones_like(speed))
        return u * scale, v * scale

    def _beta(self, lat, gx, gy, prm):
        """Beta drift (:class:`naturev1.cyclone.beta.BetaDrift`) with each particle's strength and angle."""
        rad = torch.deg2rad(lat)
        planetary = 2.0 * OMEGA * torch.cos(rad) / (EARTH_RADIUS_KM * 1000.0)
        sign = torch.where(lat >= 0, 1.0, -1.0).to(self.dt)
        gx, gy = sign * gx, sign * (gy + planetary)
        size = torch.hypot(gx, gy)
        beta20 = 2.0 * OMEGA * math.cos(math.radians(20.0)) / (EARTH_RADIUS_KM * 1000.0)
        speed = (self.fields.drift_ms * prm["beta"] * torch.sqrt((size / beta20).clamp(0.05, 4.0))
                 * math.sqrt(max(self.fields.outer_ms, 0.0) / 6.0))
        ex, ey = gx / size.clamp(min=1e-15), gy / size.clamp(min=1e-15)
        a = torch.deg2rad(sign * (self.fields.angle_deg + prm["angle"]))
        c, s_ = torch.cos(a), torch.sin(a)
        return speed * (c * ex - s_ * ey), speed * (s_ * ex + c * ey)

    def _step(self, lat, lon, hours: float, h: float, prm=None):
        """One midpoint (RK2) step of ``h`` hours on the sphere."""
        u1, v1 = self.velocity(lat, lon, hours, prm)
        la, lo = _move(lat, lon, u1, v1, 0.5 * h)
        u2, v2 = self.velocity(la, lo, hours + 0.5 * h, prm)
        la, lo = _move(lat, lon, u2, v2, h)
        lim = self.fields.lat
        return la.clamp(float(lim[0]), float(lim[-1])), lo

    def _evolve(self, prm: dict, h: float, n: int, hours: float, target: float | None = None) -> None:
        """Advance every world's internal state by ``h`` hours: its steering depth relaxes toward the target depth
        and wanders (the z of its worldline), its wobble is an Ornstein-Uhlenbeck velocity, and after the analysis
        time its steering error grows as a random walk."""
        p = self.p
        target = self.depth_target if target is None else target
        a = math.exp(-h / max(p.depth_tau_h, 1e-6))
        aim = (target + prm["depth_bias"]).clamp(0.0, 1.0)
        prm["depth"] = (aim + (prm["depth"] - aim) * a + p.depth_walk_per_sqrt_h * math.sqrt(h) * self._randn(n)).clamp(0.0, 1.0)
        b = math.exp(-h / max(p.wobble_tau_h, 1e-6))
        # Wobbles are trackable while fixes arrive, not predictable: after the analysis they only decay.
        q = p.wobble_ms * math.sqrt(max(1.0 - b * b, 0.0)) if hours < 0 else 0.0
        prm["wob_u"] = prm["wob_u"] * b + q * self._randn(n)
        prm["wob_v"] = prm["wob_v"] * b + q * self._randn(n)
        if hours >= 0:
            prm["walk"] = prm["walk"] + p.growth_ms_per_sqrt_h * math.sqrt(h) * self._randn(n, self.fields.n_modes)

    def _remember(self, hours: float, lat, lon, prm) -> None:
        """Record the tracked worlds (``record`` of them, chosen at random and followed through every resampling
        to one of their own descendants)."""
        if not self.record:
            return
        if getattr(self, "tracked", None) is None:
            k = min(self.record, self.n)
            self.tracked = torch.randperm(self.n, generator=self.gen)[:k].to(self.dev)
            self.broken = np.zeros(k, bool)
        t = self.tracked
        self.worldlines.append((hours, _to_np(lat[t], lon[t]), prm["depth"][t].detach().cpu().numpy(), self.broken.copy()))
        self.broken[:] = False

    def _follow(self, idx: torch.Tensor) -> None:
        """After resampling (new world j is a copy of old world ``idx[j]``), move each tracked world to one of its
        descendants; a world with none ended here, and its slot starts following a new one."""
        if getattr(self, "tracked", None) is None:
            return
        first = torch.full((self.n,), -1, dtype=torch.long, device=self.dev)
        order = torch.arange(idx.shape[0], device=self.dev)
        first.scatter_reduce_(0, idx, order, reduce="amin", include_self=False)
        heir = first[self.tracked]
        dead = heir < 0
        if bool(dead.any()):
            heir[dead] = torch.randint(0, self.n, (int(dead.sum()),), generator=self.gen).to(self.dev)
            self.broken |= dead.cpu().numpy()
        self.tracked = heir

    def worldline_array(self) -> tuple[np.ndarray, np.ndarray]:
        """The recorded worldlines ``(K, T, 3)`` -- lat, lon and steering depth (0 = 850-700 hPa, 1 = 850-200) -- at
        ``(T,)`` hours. A world that died out in a resampling ends with a NaN gap and its slot follows another."""
        if not self.worldlines:
            return np.zeros((0, 0, 3)), np.zeros(0)
        hours = np.array([w[0] for w in self.worldlines])
        arr = np.stack([np.concatenate([w[1], w[2][:, None]], -1) for w in self.worldlines], 1)
        for t, w in enumerate(self.worldlines):
            if t > 0 and w[3].any():
                arr[w[3], t - 1] = np.nan                                  # break the line before its new world
        return arr, hours

    # ------------------------------------------------------------------------------------------------ filter --

    def advance_to(self, hours: float) -> None:
        """Integrate every world to ``hours`` (relative to the analysis) in micro-segments of at most
        ``step_hours`` -- fixes can arrive at any minute."""
        if not self.worldlines:
            self._remember(self.hours, self.lat, self.lon, self.params)
        while self.hours < hours - 1e-9:
            h = min(self.step_hours, hours - self.hours)
            self.lat, self.lon = self._step(self.lat, self.lon, self.hours, h)
            self._evolve(self.params, h, self.n, self.hours)
            self.hours += h
            self._remember(self.hours, self.lat, self.lon, self.params)

    def update(self, lat: float, lon: float, sigma_km: float | None = None, wind_kt: float | None = None) -> float:
        """Re-weight the swarm by a fix at the current time; resample when it has thinned. Returns the effective
        sample size as a fraction of the swarm. A fix that carries an intensity also moves the steering depth the
        worlds aim for (a strengthening storm reaches higher)."""
        sigma = sigma_km or self.p.fix_sigma_km
        if wind_kt is not None and np.isfinite(wind_kt):
            self.depth_target = depth_for_wind(wind_kt)
        d = _haversine(self.lat, self.lon, lat, float(self.fields.to_box(lon)))
        self.logw = self.logw - 0.5 * (d / sigma) ** 2
        self.logw = self.logw - torch.logsumexp(self.logw, 0)
        w = torch.exp(self.logw)
        ess = float(1.0 / (w * w).sum()) / self.n
        self.history.append({"hours": self.hours, "ess": ess, "median_km": float(d.median())})
        if ess < self.p.resample_ess:
            self.resample()
        return ess

    def resample(self) -> None:
        """Systematic resampling, then a little jitter on every continuous parameter (regularised filter)."""
        w = torch.exp(self.logw - torch.logsumexp(self.logw, 0))
        idx = _systematic(w, self.n, self._rand(1))
        self._follow(idx)
        self.params = {k: v[idx].clone() for k, v in self.params.items()}
        self.lat, self.lon = self.lat[idx].clone(), self.lon[idx].clone()
        self.logw = torch.full((self.n,), -math.log(self.n), dtype=self.dt, device=self.dev)
        self._roughen(self.params, self.n)

    def _roughen(self, prm: dict, n: int) -> None:
        r, p = self.p.roughen, self.p
        jitter = r * p.layer_sd * self._randn(n)
        prm["depth_bias"] = prm["depth_bias"] + jitter
        prm["depth"] = (prm["depth"] + jitter).clamp(0.0, 1.0)
        prm["beta"] = (prm["beta"] * torch.exp(r * p.beta_log_sd * self._randn(n))).clamp(0.3, 3.0)
        prm["angle"] = (prm["angle"] + r * p.beta_angle_sd_deg * self._randn(n)).clamp(-40.0, 40.0)
        prm["persist"] = (prm["persist"] + r * p.persistence_max * 0.3 * self._randn(n)).clamp(0.0, p.persistence_max)
        lo, hi = p.anomaly_tau_h
        prm["anom_tau"] = (prm["anom_tau"] + r * (hi - lo) * 0.3 * self._randn(n)).clamp(lo, hi)
        prm["static"] = prm["static"] + r * p.static_ms * self._randn(n, self.fields.n_modes)
        prm["res_u"] = prm["res_u"] + r * p.residual_ms * self._randn(n)
        prm["res_v"] = prm["res_v"] + r * p.residual_ms * self._randn(n)

    def assimilate_history(self, case, sigma_km: float | None = None) -> list:
        """Run the swarm through the case's own fixes from its start to the analysis time, re-weighting at each."""
        from ..cyclone.data import HISTORY_HOURS

        for h in sorted(HISTORY_HOURS):
            if h <= self.hours + 1e-9:
                continue
            k = HISTORY_HOURS.index(h)
            la, lo = case.history["lat"][k], case.history["lon"][k]
            if h == 0:
                la, lo = case.lat, case.lon
            if not np.isfinite(la):
                continue
            self.advance_to(float(h))
            wind = case.wind_kt if h == 0 else case.history["wind"][k]
            self.update(float(la), float(lo), sigma_km, float(wind) if np.isfinite(wind) else None)
        return self.history

    # ------------------------------------------------------------------------------------------------ forecast --

    def posterior(self, n: int | None = None, roughen: bool = True) -> dict:
        """``n`` worlds drawn from the filtered swarm (parameters only)."""
        n = n or self.n
        w = torch.exp(self.logw - torch.logsumexp(self.logw, 0))
        idx = _systematic(w, n, self._rand(1))
        prm = {k: v[idx].clone() for k, v in self.params.items()}
        prm["walk"] = torch.zeros(n, self.fields.n_modes, dtype=self.dt, device=self.dev)
        if roughen:
            self._roughen(prm, n)
        return prm

    def forecast(self, leads, lat: float | None = None, lon: float | None = None, particles: int | None = None,
                 keep_track: bool = True, intensity_kt=None, intensity_hours=None) -> SwarmForecast:
        """
        Forecast from the latest fix (``lat, lon``; default the swarm's own weighted position) with worlds drawn
        from the filtered posterior. The fix is more precise than any one world's position, so the swarm is
        relaunched there: the filter has done its job by choosing the physics.
        """
        n = particles or self.n
        prm = self.posterior(n)
        if lat is None:
            w = torch.exp(self.logw - torch.logsumexp(self.logw, 0))
            lat, lon = float((w * self.lat).sum()), float((w * self.lon).sum())
        lat0 = lat
        la, lo = self._launch(lat, float(self.fields.to_box(lon)), n)
        schedule = None
        if intensity_kt is not None:
            schedule = (np.asarray(intensity_hours, float), np.array([depth_for_wind(v) for v in intensity_kt]))
        pos, track, hours = self._propagate(la, lo, prm, leads, keep_track, schedule)
        depth = getattr(self, "_last_depths", None)
        return SwarmForecast(list(leads), pos, np.full(n, 1.0 / n), track=track, track_hours=hours,
                             track_depth=depth if keep_track else None, hemisphere=1.0 if lat0 >= 0 else -1.0,
                             meta={"particles": n, "filter": list(self.history)})

    def _propagate(self, lat, lon, prm, leads, keep_track: bool, schedule=None):
        leads = [float(x) for x in leads]
        n = lat.shape[0]
        out = np.full((n, len(leads), 2), np.nan)
        steps = [(0.0, lat, lon)]
        depths = [prm["depth"].detach().cpu().numpy().astype(np.float32)] if keep_track else []
        t, k = 0.0, 0
        while k < len(leads) and leads[k] <= 1e-9:
            out[:, k] = _to_np(lat, lon)
            k += 1
        end = max(leads) if leads else 0.0
        step = self.step_hours if keep_track else max(self.step_hours, STEP_HOURS)
        while t < end - 1e-9:
            h = min(step, end - t)
            lat, lon = self._step(lat, lon, t, h, prm)
            target = None if schedule is None else float(np.interp(t + h, schedule[0], schedule[1]))
            self._evolve(prm, h, n, t, target)
            t += h
            if keep_track:
                steps.append((t, lat, lon))
                depths.append(prm["depth"].detach().cpu().numpy().astype(np.float32))
            while k < len(leads) and leads[k] <= t + 1e-9:
                out[:, k] = _to_np(lat, lon)
                k += 1
        if not keep_track:
            return out, None, None
        hours = np.array([s[0] for s in steps])
        track = np.stack([_to_np(s[1], s[2]) for s in steps], 1)
        self._last_depths = np.stack(depths, 1)
        return out, track, hours

    def stream(self, total: int, leads, lat: float, lon: float, batch: int = 100_000, grid_step_deg: float = 0.5,
               strike_radius_km: float = 111.0, coverage: float = 0.67, seconds: float | None = None) -> dict:
        """
        Forecast with ``total`` worlds in batches of ``batch``, drawn afresh from the filtered posterior, keeping only
        running summaries: so the swarm size is limited by time, not memory. Returns the mean track, the cone, a
        strike-probability map on a ``grid_step_deg`` grid over the analysis box, the scenario weights, the number of
        worlds actually run and the throughput.
        """
        glat = np.arange(self.fields.lat[0], self.fields.lat[-1] + 1e-9, grid_step_deg)
        glon = np.arange(self.fields.lon[0], self.fields.lon[-1] + 1e-9, grid_step_deg)
        hist = np.zeros((len(leads), glat.size, glon.size))
        strike = np.zeros((glat.size, glon.size))
        sums = np.zeros((len(leads), 2))
        counts = {"loop": 0.0, "stall": 0.0, "recurve": 0.0, "steady": 0.0}
        done, start = 0, time.time()
        while done < total and (seconds is None or time.time() - start < seconds):
            n = min(batch, total - done)
            prm = self.posterior(n)
            la, lo = self._launch(lat, float(self.fields.to_box(lon)), n)
            pos, track, hours = self._propagate(la, lo, prm, leads, True)
            sums += np.nansum(pos, 0)
            for k in range(len(leads)):
                i = np.clip(np.round((pos[:, k, 0] - glat[0]) / grid_step_deg).astype(int), 0, glat.size - 1)
                j = np.clip(np.round((pos[:, k, 1] - glon[0]) / grid_step_deg).astype(int), 0, glon.size - 1)
                np.add.at(hist[k], (i, j), 1.0)
            strike += _strike_counts(track, glat, glon, strike_radius_km)
            labels = classify_tracks(track, hours, self.hemisphere)
            for name in counts:
                counts[name] += float((labels == name).sum())
            done += n
        elapsed = time.time() - start
        mean = sums / max(done, 1)
        la_c, lo_c = np.meshgrid(glat, glon, indexing="ij")
        cone = []
        for k in range(len(leads)):
            d = _haversine_np(la_c, lo_c, mean[k, 0], mean[k, 1]).ravel()
            cone.append(_weighted_quantile(d, hist[k].ravel(), coverage))
        return {"particles": done, "seconds": elapsed, "worlds_per_second": done / max(elapsed, 1e-9),
                "mean_track": np.stack([mean[:, 0], wrap_longitude(mean[:, 1])], -1), "cone_km": np.array(cone),
                "strike_probability": strike / max(done, 1), "grid_lat": glat, "grid_lon": wrap_longitude(glon),
                "scenarios": {k: v / max(done, 1) for k, v in counts.items()}}


def rescale_worlds(fc, case, ref_track: np.ndarray, leads, factor: np.ndarray) -> None:
    """
    Scale every world's displacement from ``ref_track`` ``(L, 2)`` by ``factor`` ``(L,)`` per lead, in place -- the
    positions at the leads and the hourly worldlines (the factor interpolated in time, 1 at the analysis). An affine
    change about the reference: the worlds keep their shapes and their weights, only their spread changes.
    """
    from ..cyclone.geometry import local_displacement_km, move

    factor = np.asarray(factor, np.float64)
    e, n = local_displacement_km(ref_track[None, :, 0], ref_track[None, :, 1], fc.positions[..., 0], fc.positions[..., 1])
    la, lo = move(ref_track[None, :, 0], ref_track[None, :, 1], e * factor[None, :], n * factor[None, :])
    lo = ref_track[None, :, 1] + ((lo - ref_track[None, :, 1] + 180.0) % 360.0 - 180.0)      # stay in the swarm's continuous frame
    fc.positions = np.stack([la, lo], -1)
    if getattr(fc, "track", None) is not None:
        hours = np.concatenate([[0.0], np.asarray(leads, float)])
        f_h = np.interp(fc.track_hours, hours, np.concatenate([[1.0], factor]))
        lat_h = np.interp(fc.track_hours, hours, np.concatenate([[case.lat], ref_track[:, 0]]))
        lon_h = np.degrees(np.interp(fc.track_hours, hours, np.unwrap(np.radians(np.concatenate([[case.lon], ref_track[:, 1]])))))
        e, n = local_displacement_km(lat_h[None, :], lon_h[None, :], fc.track[..., 0], fc.track[..., 1])
        la, lo = move(lat_h[None, :], lon_h[None, :], e * f_h[None, :], n * f_h[None, :])
        lo = lon_h[None, :] + ((lo - lon_h[None, :] + 180.0) % 360.0 - 180.0)
        fc.track = np.stack([la, lo], -1)


def swarm_products(case, ctx, track: np.ndarray, leads, particles: int = 20000, device="cpu", seed: int = 0,
                   grid_step_deg: float = 0.5, strike_radius_km: float = 111.0, fixes: list | None = None,
                   micro_minutes: float = 15.0, intensity_kt=None, forecast_flow=None, target_cone_km=None) -> dict:
    """
    The swarm as the probabilistic half of :func:`naturev1.cyclone.forecast.physics_forecast`: filtered on the last
    day's fixes, then re-centred on ``track`` (the consensus, which stays the best single estimate: on held-out
    seasons the swarm's own mean is worse) so that it contributes what a single track cannot -- how likely each kind
    of future is, where the centre may pass, and a cone whose size is calibrated on past seasons
    (``coefficients.SWARM``, or ``SWARM_FF`` when the worlds are steered by a forecast flow).

    ``target_cone_km`` (per lead) sharpens or widens the swarm to a cone sized for *this* forecast: the worlds'
    displacements about the consensus are rescaled per lead so the swarm's 67% radius equals it (the factor clipped
    to 0.2-2.0). With the flow-dependent cone (``CONE_FLOW``, verified on held-out seasons) the swarm is
    tight when the models agree and wide when they do not, and every product built on the worlds -- strike
    probabilities, landfall timing and probability, the parish numbers -- sharpens with it. The scenario
    probabilities are read before the rescaling (they are about the worlds' shapes).
    """
    from ..cyclone import coefficients

    # steered by a forecast flow, the worlds spread differently: that swarm has its own cone calibration
    table = (getattr(coefficients, "SWARM_FF", None) if forecast_flow is not None else None) or getattr(coefficients, "SWARM", None) or {}
    pri = SwarmPriors(layer_depth=depth_for_wind(case.wind_kt), **table.get("priors", {}))
    fields = SwarmFields.from_context(ctx, device=device, forecast=forecast_flow)
    sw = Swarm.from_context(case, ctx, particles, priors=pri, device=device, seed=seed, fields=fields)
    if fixes:
        # the satellite path: every centre fix at its own minute, the worlds advancing in micro-segments between them
        from .fixes import assimilate

        sw.step_hours = micro_minutes / 60.0
        assimilate(sw, fixes, case.time)
        if sw.hours < -1e-6 or not sw.history or abs(sw.history[-1]["hours"]) > 1e-6:
            sw.advance_to(0.0)
            sw.update(case.lat, case.lon, None, case.wind_kt)
        sw.step_hours = STEP_HOURS
    else:
        sw.assimilate_history(case)
    schedule = {}
    if intensity_kt is not None:
        kt = np.asarray(intensity_kt, float)
        ok = np.isfinite(kt)
        schedule = {"intensity_kt": np.concatenate([[case.wind_kt], kt[ok]]),
                    "intensity_hours": np.concatenate([[0.0], np.asarray(leads, float)[ok]])}
    fc = sw.forecast(leads, case.lat, case.lon, **schedule)
    raw_mean = fc.mean_track()
    good = np.isfinite(track[:, 0])
    if good.any():
        shift = np.where(good[:, None], track - raw_mean, 0.0)
        shift[:, 1] = (shift[:, 1] + 180.0) % 360.0 - 180.0
        fc.positions = fc.positions + shift[None]
        if fc.track is not None:
            hours = np.concatenate([[0.0], np.asarray(leads, float)])
            full = np.concatenate([[[0.0, 0.0]], shift])
            fc.track = fc.track + np.stack([np.interp(fc.track_hours, hours, full[:, i]) for i in range(2)], -1)[None]
    scenarios = fc.scenarios()
    spread = np.array([_weighted_quantile(fc.distances_km()[:, k], fc.weights, 0.67) for k in range(len(leads))])
    scale = np.interp(np.asarray(leads, float), table.get("leads", list(leads)), table.get("cone_scale", np.ones(len(leads))))
    sharpen = np.ones(len(leads))
    cone = scale * spread
    if target_cone_km is not None and good.any():
        # the worlds' raw spread is wider than the truth distribution (that is what cone_scale < 1 says), and it is the
        # raw positions that make the strike, landfall and parish probabilities: rescale them so the swarm's own 67%
        # radius about the consensus equals the cone sized for this forecast
        target = np.asarray(target_cone_km, np.float64)
        ok = np.isfinite(target) & np.isfinite(spread) & (spread > 1.0) & good
        sharpen[ok] = np.clip(target[ok] / spread[ok], 0.2, 2.0)
        ref = np.where(good[:, None], track, raw_mean)
        rescale_worlds(fc, case, ref, leads, sharpen)
        spread = np.array([_weighted_quantile(fc.distances_km()[:, k], fc.weights, 0.67) for k in range(len(leads))])
        cone = np.where(ok, spread, scale * spread)
    lat_c, lon_c = float(np.nanmean(fc.positions[..., 0])), float(np.nanmean(fc.positions[..., 1]))
    glat = np.arange(np.floor(lat_c) - 15.0, np.floor(lat_c) + 15.0 + 1e-9, grid_step_deg)
    glon = np.arange(np.floor(lon_c) - 25.0, np.floor(lon_c) + 25.0 + 1e-9, grid_step_deg)
    return {"forecast": fc, "raw_mean_track": raw_mean, "cone_km": cone, "scenarios": scenarios, "sharpen": sharpen,
            "strike_probability": fc.strike_probability(glat, glon, strike_radius_km), "grid_lat": glat,
            "grid_lon": wrap_longitude(glon), "filter": list(sw.history)}


# ---------------------------------------------------------------------------------------------------- helpers --

def depth_for_wind(wind_kt: float) -> float:
    """Prior steering depth on the 0 (850-700) .. 1 (850-200 hPa) scale: weak storms shallow, strong ones deep."""
    from ..cyclone.steering import intensity_layer

    top, _ = intensity_layer(float(np.nan_to_num(wind_kt, nan=50.0)))
    return float(np.clip(np.interp(top, [200.0, 400.0, 700.0], [1.0, 0.5, 0.0]), 0.0, 1.0))


def _move(lat, lon, u, v, hours: float):
    """Move by (u, v) m/s for ``hours``, as a short great-circle step (equirectangular is exact enough at 3 h)."""
    dist = hours * 3.6
    dlat = (v * dist) / (math.pi * EARTH_RADIUS_KM / 180.0)
    dlon = (u * dist) / ((math.pi * EARTH_RADIUS_KM / 180.0) * torch.cos(torch.deg2rad(lat + 0.5 * dlat)).clamp(min=0.05))
    return lat + dlat, lon + dlon


def _haversine(lat1: torch.Tensor, lon1: torch.Tensor, lat2, lon2) -> torch.Tensor:
    """Great-circle km between particles and a point (or matching tensors)."""
    lat2 = torch.as_tensor(lat2, dtype=lat1.dtype, device=lat1.device)
    lon2 = torch.as_tensor(lon2, dtype=lat1.dtype, device=lat1.device)
    p1, p2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    a = torch.sin((p2 - p1) / 2) ** 2 + torch.cos(p1) * torch.cos(p2) * torch.sin(torch.deg2rad(lon2 - lon1) / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))


def _haversine_np(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _systematic(w: torch.Tensor, n: int, u: torch.Tensor) -> torch.Tensor:
    cdf = torch.cumsum(w, 0)
    cdf = cdf / cdf[-1]
    points = (u + torch.arange(n, dtype=w.dtype, device=w.device)) / n
    return torch.searchsorted(cdf, points.clamp(max=1.0 - 1e-12)).clamp(max=w.shape[0] - 1)


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return float("nan")
    order = np.argsort(x[ok])
    xs, ws = x[ok][order], w[ok][order]
    c = np.cumsum(ws)
    return float(xs[np.searchsorted(c, q * c[-1])])


def _to_np(lat, lon) -> np.ndarray:
    return torch.stack([lat, lon], -1).detach().cpu().numpy().astype(np.float64)


def _strike_counts(track: np.ndarray, glat: np.ndarray, glon: np.ndarray, radius_km: float,
                   weights: np.ndarray | None = None, chunk: int = 4000) -> np.ndarray:
    """
    Weighted count of tracks passing within ``radius_km`` of each cell of a regular grid (each track counted once
    per cell). Tracks are densified until consecutive points are at most half the radius apart, each track's cells
    are de-duplicated, and only then dilated by the radius, in chunks: memory stays flat at any swarm size.
    """
    di_step, dj_step = float(glat[1] - glat[0]), float(glon[1] - glon[0])
    ni, nj = glat.size, glon.size
    out = np.zeros(ni * nj)
    w_all = np.ones(track.shape[0]) if weights is None else np.asarray(weights, np.float64)
    seg = _haversine_np(track[:, 1:, 0], track[:, 1:, 1], track[:, :-1, 0], track[:, :-1, 1])
    sub = int(np.clip(np.ceil(np.nanmax(seg) / (0.5 * radius_km)), 1, 12)) if seg.size and np.isfinite(seg).any() else 1
    if sub > 1:
        f = np.linspace(0.0, 1.0, sub + 1)[:-1]
        a, b = track[:, :-1], track[:, 1:]
        dense = (a[:, :, None] + (b - a)[:, :, None] * f[None, None, :, None]).reshape(track.shape[0], -1, 2)
        track = np.concatenate([dense, track[:, -1:]], 1)
    coslat = max(math.cos(math.radians(min(float(np.nanmax(np.abs(track[..., 0]))), 80.0))), 0.15)
    ri = int(math.ceil(radius_km / (111.2 * di_step)))
    rj = int(math.ceil(radius_km / (111.2 * dj_step * coslat)))
    oi, oj = np.meshgrid(np.arange(-ri, ri + 1), np.arange(-rj, rj + 1), indexing="ij")
    mid = math.cos(math.radians(float(np.nanmean(np.abs(track[..., 0])))))
    keep = 111.2 * np.hypot(oi * di_step, oj * dj_step * mid) <= radius_km + 0.5 * 111.2 * max(di_step, dj_step)
    oi, oj = oi[keep], oj[keep]
    for s0 in range(0, track.shape[0], chunk):
        t = track[s0:s0 + chunk]
        n = t.shape[0]
        i = np.round((t[..., 0] - glat[0]) / di_step)
        j = np.round((t[..., 1] - glon[0]) / dj_step)
        # points farther off the grid than the radius cannot strike it -- and would overflow the packed index below
        ok = np.isfinite(i) & np.isfinite(j)
        ok &= (np.nan_to_num(i, nan=-1e9) >= -ri) & (np.nan_to_num(i, nan=1e9) < ni + ri)
        ok &= (np.nan_to_num(j, nan=-1e9) >= -rj) & (np.nan_to_num(j, nan=1e9) < nj + rj)
        pid = np.broadcast_to(np.arange(n)[:, None], i.shape)[ok].astype(np.int64)
        cells = np.unique(pid * (ni + 2 * ri) * (nj + 2 * rj) + (i[ok].astype(np.int64) + ri) * (nj + 2 * rj)
                          + (j[ok].astype(np.int64) + rj))
        pid, rem = np.divmod(cells, (ni + 2 * ri) * (nj + 2 * rj))
        ci, cj = np.divmod(rem, nj + 2 * rj)
        ci, cj = ci - ri, cj - rj
        ii = (ci[:, None] + oi[None]).ravel()
        jj = (cj[:, None] + oj[None]).ravel()
        pp = np.repeat(pid, oi.size)
        ok = (ii >= 0) & (ii < ni) & (jj >= 0) & (jj < nj)
        keys = np.unique(pp[ok] * (ni * nj) + ii[ok] * nj + jj[ok])
        p_ix, cell = np.divmod(keys, ni * nj)
        np.add.at(out, cell, w_all[s0:s0 + chunk][p_ix])
    return out.reshape(ni, nj)


__all__ = ["SPEED_CAP_MS", "Swarm", "SwarmFields", "SwarmForecast", "SwarmPriors", "classify_tracks", "depth_for_wind",
           "rescale_worlds",
           "swarm_products"]
