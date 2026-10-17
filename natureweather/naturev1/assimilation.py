# Copyright 2026 Nathan. Apache-2.0.
"""
Data assimilation for live running: scrub real-time inputs before they reach a model trained on ERA5.

NatureV1 (like every machine-learning weather model) learned on ERA5, a reanalysis polished by 4D-Var with
observations that arrived for years after the fact. In real time the inputs are messier: another centre's
operational analysis with its own biases and resolution, late or missing observations, a radiosonde with a
transmission error. Fed straight in, those differences become forecast error. ECMWF's AIFS sidesteps this by
running inside an operational data-assimilation cycle; this module gives NatureV1 a compact version of one:

1. **Background** -- NatureV1's own 6-hour forecast from the previous cycle, with its own predicted variance
   (``state_delta_log_var``) as the background error variance: flow-dependent, large where the model is
   unsure (a developing storm, a front), small where it is confident.
2. **Bias correction** (:class:`BiasCorrector`) -- the running mean of innovations (observation minus background)
   per channel and latitude band, an exponentially-weighted version of variational bias correction, removed
   from each source before it is used; a real-time analysis that is systematically 0.5 K warmer than ERA5 at
   850 hPa stops pulling NatureV1 off its climate.
3. **Quality control** (:func:`quality_control`) -- a gross check against physical limits, a background check
   (reject when ``|O - B| > k sqrt(sigma_o^2 + sigma_b^2)``), and a buddy check (reject an observation whose
   innovation disagrees with the median of its neighbours' by more than ``k`` of their spread).
4. **Analysis** (:func:`analyse`) -- optimal interpolation, solved locally in tiles (as in a local ensemble
   transform filter, without the ensemble): the observations within a Gaspari-Cohn localisation radius of a tile
   are combined with the background through ``x_a = x_b + B H^T (H B H^T + R)^-1 (y - H x_b)``, with Gaussian
   background-error correlations of a per-variable length scale.
5. **Super-observations** (:func:`superobs`) -- a gridded source (a 0.25-degree analysis) is first averaged into
   coarser boxes, so its spatially correlated errors are not counted as independent evidence.

Everything works channel by channel on NatureV1's normalised ``(points, channels)`` state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cyclone.geometry import EARTH_RADIUS_KM

#: Physical limits for the gross check, in SI units, by variable base name.
LIMITS = {"temperature": (150.0, 340.0), "2m_temperature": (180.0, 340.0), "geopotential": (-5000.0, 220000.0),
          "u_component_of_wind": (-120.0, 120.0), "v_component_of_wind": (-120.0, 120.0),
          "10m_u_component_of_wind": (-80.0, 80.0), "10m_v_component_of_wind": (-80.0, 80.0),
          "mean_sea_level_pressure": (85000.0, 110000.0), "specific_humidity": (0.0, 0.04),
          "surface_pressure": (45000.0, 110000.0), "sea_surface_temperature": (269.0, 310.0)}
#: Background-error correlation length scale, km, by variable base name (default 400).
LENGTH_KM = {"geopotential": 600.0, "mean_sea_level_pressure": 500.0, "temperature": 400.0, "2m_temperature": 250.0,
             "u_component_of_wind": 350.0, "v_component_of_wind": 350.0, "specific_humidity": 250.0}


def unit_vectors(lat, lon) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def chord_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distance between unit vectors (rows of ``a`` against rows of ``b``), km."""
    dot = np.clip(a @ b.T, -1.0, 1.0)
    return EARTH_RADIUS_KM * np.arccos(dot)


def gaspari_cohn(distance: np.ndarray, radius: float) -> np.ndarray:
    """Gaspari-Cohn fifth-order compactly supported correlation, zero beyond ``2 * radius`` -- here ``radius`` is half."""
    r = np.abs(distance) / radius
    out = np.zeros_like(r)
    a = r <= 1.0
    b = (r > 1.0) & (r < 2.0)
    ra, rb = r[a], r[b]
    out[a] = -0.25 * ra ** 5 + 0.5 * ra ** 4 + 0.625 * ra ** 3 - 5.0 / 3.0 * ra ** 2 + 1.0
    out[b] = rb ** 5 / 12.0 - 0.5 * rb ** 4 + 0.625 * rb ** 3 + 5.0 / 3.0 * rb ** 2 - 5.0 * rb + 4.0 - 2.0 / (3.0 * rb)
    return out


@dataclass
class Observations:
    """Observations of one channel: positions, values and error standard deviations (all in the same units as
    the state they are compared with)."""

    lat: np.ndarray
    lon: np.ndarray
    value: np.ndarray
    sigma: np.ndarray
    source: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="<U16"))

    def __post_init__(self):
        self.lat, self.lon = np.asarray(self.lat, np.float64), np.asarray(self.lon, np.float64)
        self.value, self.sigma = np.asarray(self.value, np.float64), np.broadcast_to(np.asarray(self.sigma, np.float64), self.lat.shape).copy()
        if self.source.size != self.lat.size:
            self.source = np.full(self.lat.size, "obs", dtype="<U16")

    def subset(self, keep: np.ndarray) -> "Observations":
        return Observations(self.lat[keep], self.lon[keep], self.value[keep], self.sigma[keep], self.source[keep])

    def __len__(self) -> int:
        return self.lat.size


def interpolate_background(grid_lat: np.ndarray, grid_lon: np.ndarray, field: np.ndarray, lat, lon) -> np.ndarray:
    """Bilinear background ``H x_b`` at observation points on a regular grid (lat ascending, lon 0..360)."""
    step_lat = grid_lat[1] - grid_lat[0]
    step_lon = grid_lon[1] - grid_lon[0]
    grid = field.reshape(grid_lat.size, grid_lon.size)
    fi = np.clip((np.asarray(lat) - grid_lat[0]) / step_lat, 0, grid_lat.size - 1.000001)
    fj = (np.asarray(lon) % 360.0 - grid_lon[0]) / step_lon
    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int) % grid_lon.size
    j1 = (j0 + 1) % grid_lon.size
    di, dj = fi - i0, fj - np.floor(fj)
    return ((1 - di) * (1 - dj) * grid[i0, j0] + (1 - di) * dj * grid[i0, j1] + di * (1 - dj) * grid[i0 + 1, j0]
            + di * dj * grid[i0 + 1, j1])


class BiasCorrector:
    """Running mean of innovations per latitude band (an exponentially weighted variational bias correction)."""

    def __init__(self, bands_deg: float = 15.0, memory: float = 0.05) -> None:
        self.edges = np.arange(-90.0, 90.0 + bands_deg, bands_deg)
        self.memory = memory
        self.bias: dict[str, np.ndarray] = {}
        self.updates: dict[str, np.ndarray] = {}

    def correct(self, source: str, obs: Observations) -> Observations:
        b = self.bias.get(source)
        if b is None:
            return obs
        band = np.clip(np.searchsorted(self.edges, obs.lat) - 1, 0, len(self.edges) - 2)
        return Observations(obs.lat, obs.lon, obs.value - b[band], obs.sigma, obs.source)

    def update(self, source: str, obs: Observations, innovation: np.ndarray) -> None:
        band = np.clip(np.searchsorted(self.edges, obs.lat) - 1, 0, len(self.edges) - 2)
        current = self.bias.get(source, np.zeros(len(self.edges) - 1))
        seen = self.updates.get(source, np.zeros(len(self.edges) - 1))
        for k in range(len(current)):
            sel = band == k
            if sel.sum() >= 5:
                # Warm start: the first updates are a plain running mean, then the memory takes over.
                rate = max(self.memory, 1.0 / (seen[k] + 1.0))
                # Innovations measured against bias-corrected values: add the current estimate back.
                current[k] = (1 - rate) * current[k] + rate * (current[k] + float(np.median(innovation[sel])))
                seen[k] += 1
        self.bias[source], self.updates[source] = current, seen


def quality_control(obs: Observations, background_at_obs: np.ndarray, background_sigma_at_obs: np.ndarray,
                    limits: tuple[float, float] | None = None, k_background: float = 4.0, k_buddy: float = 3.0,
                    buddy_km: float = 300.0) -> tuple[np.ndarray, dict]:
    """``(keep, counts)``: gross, background and buddy checks; ``limits`` in the observations' units."""
    keep = np.isfinite(obs.value)
    counts = {"total": int(len(obs))}
    if limits is not None:
        gross = (obs.value < limits[0]) | (obs.value > limits[1])
        keep &= ~gross
        counts["gross"] = int(gross.sum())
    innovation = obs.value - background_at_obs
    scale = np.sqrt(obs.sigma ** 2 + background_sigma_at_obs ** 2)
    background = np.abs(innovation) > k_background * scale
    keep &= ~background
    counts["background"] = int((background & np.isfinite(obs.value)).sum())
    buddy_reject = np.zeros(len(obs), bool)
    idx = np.nonzero(keep)[0]
    if idx.size > 3:
        v = unit_vectors(obs.lat[idx], obs.lon[idx])
        for start in range(0, idx.size, 2048):
            block = idx[start:start + 2048]
            d = chord_km(unit_vectors(obs.lat[block], obs.lon[block]), v)
            near = (d < buddy_km) & (d > 0)
            for row, i in enumerate(block):
                neighbours = idx[near[row]]
                if neighbours.size < 3:
                    continue
                med = np.median(innovation[neighbours])
                spread = 1.4826 * np.median(np.abs(innovation[neighbours] - med)) + 0.5 * scale[i]
                if abs(innovation[i] - med) > k_buddy * spread:
                    buddy_reject[i] = True
    keep &= ~buddy_reject
    counts["buddy"] = int(buddy_reject.sum())
    counts["kept"] = int(keep.sum())
    return keep, counts


def superobs(obs: Observations, box_deg: float = 1.5) -> Observations:
    """Average observations into ``box_deg`` boxes (error shrinks only as the square root of a quarter of the
    count, a hedge against their correlated errors)."""
    i = np.floor((obs.lat + 90.0) / box_deg).astype(int)
    j = np.floor((obs.lon % 360.0) / box_deg).astype(int)
    key = i * 100000 + j
    unique, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    lat = np.bincount(inverse, obs.lat) / counts
    lon = np.bincount(inverse, np.where(obs.lon % 360.0 > 359.0, obs.lon % 360.0 - 360.0, obs.lon % 360.0)) / counts
    value = np.bincount(inverse, obs.value) / counts
    sigma = np.sqrt(np.bincount(inverse, obs.sigma ** 2) / counts) / np.sqrt(np.maximum(counts / 4.0, 1.0))
    return Observations(lat, lon % 360.0, value, sigma)


def analyse(grid_lat: np.ndarray, grid_lon: np.ndarray, background: np.ndarray, background_sigma: np.ndarray,
            obs: Observations, length_km: float = 400.0, localisation_km: float = 1200.0, tile: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    Local optimal interpolation of one channel. ``background``/``background_sigma``: ``(P,)`` on the regular grid.

    Returns ``(analysis (P,), analysis_sigma (P,))``.
    """
    analysis = background.astype(np.float64).copy()
    a_sigma = background_sigma.astype(np.float64).copy()
    if len(obs) == 0:
        return analysis, a_sigma
    lat_g, lon_g = np.meshgrid(grid_lat, grid_lon, indexing="ij")
    hx = interpolate_background(grid_lat, grid_lon, background, obs.lat, obs.lon)
    sb_obs = interpolate_background(grid_lat, grid_lon, background_sigma, obs.lat, obs.lon)
    innovation = obs.value - hx
    ov = unit_vectors(obs.lat, obs.lon)
    h, w = lat_g.shape
    for i0 in range(0, h, tile):
        for j0 in range(0, w, tile):
            rows = slice(i0, min(i0 + tile, h))
            cols = slice(j0, min(j0 + tile, w))
            tlat, tlon = lat_g[rows, cols].ravel(), lon_g[rows, cols].ravel()
            tv = unit_vectors(tlat, tlon)
            centre = unit_vectors(tlat.mean(), np.degrees(np.arctan2(np.sin(np.radians(tlon)).mean(), np.cos(np.radians(tlon)).mean())))
            dc = chord_km(centre[None], ov)[0]
            near = dc < localisation_km + tile * 170.0
            if not near.any():
                continue
            o = np.nonzero(near)[0]
            d_oo = chord_km(ov[o], ov[o])
            d_to = chord_km(tv, ov[o])
            loc_oo = gaspari_cohn(d_oo, localisation_km / 2.0)
            loc_to = gaspari_cohn(d_to, localisation_km / 2.0)
            sb_o = sb_obs[o]
            point_index = (np.arange(h)[rows][:, None] * w + np.arange(w)[cols][None, :]).ravel()
            sb_t = background_sigma[point_index]
            b_oo = sb_o[:, None] * sb_o[None, :] * np.exp(-0.5 * (d_oo / length_km) ** 2) * loc_oo
            b_to = sb_t[:, None] * sb_o[None, :] * np.exp(-0.5 * (d_to / length_km) ** 2) * loc_to
            s = b_oo + np.diag(obs.sigma[o] ** 2)
            try:
                gain = np.linalg.solve(s, b_to.T).T                       # (T, O)
            except np.linalg.LinAlgError:
                continue
            analysis[point_index] = background[point_index] + gain @ innovation[o]
            reduction = np.einsum("to,to->t", gain, b_to)
            a_sigma[point_index] = np.sqrt(np.maximum(sb_t ** 2 - reduction, 0.05 * sb_t ** 2))
    return analysis, a_sigma


class Assimilator:
    """
    The cycle: bias-correct each source, quality-control against the background, analyse every channel.

    ``state`` and ``sigma`` are ``(P, C)`` in physical units for the channels being analysed (decode NatureV1's
    normalised state first with its normaliser); ``observations`` maps a channel name to a list of
    ``(source name, Observations)``.
    """

    def __init__(self, channels: list[str], grid_lat=np.linspace(-90, 90, 121), grid_lon=np.arange(240) * 1.5,
                 bias: BiasCorrector | None = None, anchors=("radiosonde", "buoy", "aircraft")) -> None:
        self.channels, self.grid_lat, self.grid_lon = list(channels), np.asarray(grid_lat), np.asarray(grid_lon)
        self.bias = bias or BiasCorrector()
        # Sources trusted to be unbiased: they anchor the system and are never bias-corrected, so a biased
        # background cannot drag the corrections along with it.
        self.anchors = set(anchors)

    def cycle(self, state: np.ndarray, sigma: np.ndarray, observations: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        analysis, a_sigma = state.copy(), sigma.copy()
        report = {}
        for name, sources in observations.items():
            if name not in self.channels:
                continue
            c = self.channels.index(name)
            base = name.split("@")[0]
            merged = []
            for source, obs in sources:
                anchored = source in self.anchors
                if not anchored:
                    obs = self.bias.correct(f"{source}|{name}", obs)
                hx = interpolate_background(self.grid_lat, self.grid_lon, state[:, c], obs.lat, obs.lon)
                sb = interpolate_background(self.grid_lat, self.grid_lon, sigma[:, c], obs.lat, obs.lon)
                keep, counts = quality_control(obs, hx, sb, LIMITS.get(base))
                if not anchored:
                    self.bias.update(f"{source}|{name}", obs.subset(keep), (obs.value - hx)[keep])
                report[f"{source}|{name}"] = counts
                merged.append(obs.subset(keep))
            if not merged:
                continue
            combined = Observations(np.concatenate([m.lat for m in merged]), np.concatenate([m.lon for m in merged]),
                                    np.concatenate([m.value for m in merged]), np.concatenate([m.sigma for m in merged]))
            analysis[:, c], a_sigma[:, c] = analyse(self.grid_lat, self.grid_lon, state[:, c], sigma[:, c], combined,
                                                    LENGTH_KM.get(base, 400.0))
        return analysis, a_sigma, report


__all__ = ["Assimilator", "BiasCorrector", "LENGTH_KM", "LIMITS", "Observations", "analyse", "gaspari_cohn",
           "interpolate_background", "quality_control", "superobs", "unit_vectors"]
