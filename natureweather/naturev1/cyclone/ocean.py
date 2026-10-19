# Copyright 2026 Nathan. Apache-2.0.
"""
The ocean fights back: the cold wake a storm stirs up, and how much of it the storm itself has to sit on.

The idea: a hurricane's winds mix cold thermocline water up to the surface. A storm moving at 5 m/s leaves the
cold water behind it; a storm that stalls -- Dorian over Grand Bahama, 1 mph for a day and a half -- sits on top
of its own wake, its fuel supply drops, and it weakens even with everything else favourable. Potential
intensity computed from the analysis's sea-surface temperature knows none of this. Here, after the
observational fits of Lloyd and Vecchi (2011) and Mei et al. (2012):

* **wake cooling** grows with intensity (wind stress goes as speed squared, mixing faster still) and falls
  with translation speed (less time over each patch of ocean), and is smaller over very warm water (a warmer
  surface usually means a deeper mixed layer, more water to stir before the cold arrives);
* **the share under the core**: a fast storm feels a small fraction of its wake, a stalled one nearly all;
* **residence time**: cooling under a stalled storm accumulates (e-folding ``RESIDENCE_H``), so the longer it
  sits, the colder its ocean.

:func:`effective_sst` gives the sea-surface temperature the core actually feels at each forecast hour, which
the intensity model passes to potential intensity (:mod:`naturev1.cyclone.intensity`).
"""

from __future__ import annotations

import numpy as np

#: Peak wake cooling for a 100 kt storm moving at 5 m/s over 28.5 C water, K.
COOLING_K = 1.4
#: Speed scale of the translation dependence, m/s, and its exponent.
SPEED_SCALE_MS = 5.0
SPEED_POWER = 0.8
#: E-folding time over which cooling under a slow storm accumulates, hours.
RESIDENCE_H = 18.0


#: Depth of the 26 C isotherm the cooling constants refer to (m), and how cooling falls with a deeper warm layer.
D26_REFERENCE_M = 50.0
D26_POWER = 0.5


# With the climatology present and ``d26_power`` 0 (the shipped value), the depth branch is taken with a factor of exactly 1:
# the 0.14 surface-temperature warm-water factor is replaced, and the 0.15 refit absorbed the difference.
def wake_cooling_k(vmax_kt, translation_ms, sst_k, d26_m=None, d26_power: float | None = None) -> np.ndarray:
    """
    Peak cooling of the wake, K: rises with intensity^1.5, falls with translation speed and warm water -- and, when the
    depth of the 26 C isotherm ``d26_m`` is known, with the depth of warm water (there is more of it to stir before
    the cold arrives: a storm over 100 m of warm water cools its surface about 0.7 times what it would over 50 m).
    Without ``d26_m`` the surface temperature stands in for it, as before.
    """
    v = np.clip(np.asarray(vmax_kt, np.float64), 0.0, 200.0) / 100.0
    u = np.maximum(np.asarray(translation_ms, np.float64), 0.3)
    speed = np.clip((SPEED_SCALE_MS / u) ** SPEED_POWER, 0.4, 4.0)
    if d26_m is None:
        warm = np.clip(1.0 - 0.25 * (np.asarray(sst_k, np.float64) - 273.15 - 28.5), 0.5, 1.3)
    else:
        depth = np.where(np.isfinite(d26_m), np.asarray(d26_m, np.float64), D26_REFERENCE_M)
        power = D26_POWER if d26_power is None else float(d26_power)
        warm = np.clip((D26_REFERENCE_M / np.maximum(depth, 5.0)) ** power, 0.35, 1.8)
    return COOLING_K * v ** 1.5 * speed * warm


def core_share(translation_ms) -> np.ndarray:
    """Fraction of the wake cooling felt under the core: ~1 for a stalled storm, ~0.25 at 7 m/s."""
    u = np.maximum(np.asarray(translation_ms, np.float64), 0.0)
    return 1.0 / (1.0 + u / 2.3)


def effective_sst(sst_k: np.ndarray, vmax_kt: np.ndarray, translation_ms: np.ndarray, hours: np.ndarray,
                  d26_m: np.ndarray | None = None, d26_power: float | None = None) -> np.ndarray:
    """
    Sea-surface temperature under the core along a forecast, K.

    All arguments are ``(T,)`` along the forecast (``hours`` from now). Slow stretches accumulate cooling with
    :data:`RESIDENCE_H`; moving off resets the residence (the storm reaches new water). With ``d26_m`` (the depth of
    the 26 C isotherm along the track, from :class:`OceanClimatology`) the cooling knows how deep the warm water is:
    Melissa crawled over the central Caribbean, where the 26 C water runs 100 m deep in October, and intensified --
    a surface-temperature-only wake would have had her cooling her own ocean.
    """
    sst_k = np.asarray(sst_k, np.float64)
    out = sst_k.copy()
    residence = 0.0
    hours = np.asarray(hours, np.float64)
    for k in range(sst_k.size):
        dt = hours[k] - hours[k - 1] if k else 0.0
        slow = translation_ms[k] < 3.0
        residence = residence + dt if slow else max(residence - 2.0 * dt, 0.0)
        build = 1.0 - np.exp(-(residence + 6.0) / RESIDENCE_H)
        depth = None if d26_m is None else d26_m[k]
        cooling = wake_cooling_k(vmax_kt[k], translation_ms[k], sst_k[k], depth, d26_power) * core_share(translation_ms[k]) * build
        out[k] = sst_k[k] - cooling
    return out


def anomaly_adjusted_depth(d26_m, sst_k, sst_clim_c, per_c: float) -> np.ndarray:
    """
    The climatological 26 C depth deepened (or shoaled) by this year's surface anomaly: ``D26 * (1 + per_c * dSST)``,
    clipped to 0.5-2 times. The climatology cannot know a record-warm year; the analysis's sea-surface temperature can,
    and a warmer surface over the same water column usually means a deeper warm layer. ``per_c`` is fitted on the
    tuning seasons (``coefficients.OCEAN["sst_anomaly_per_c"]``); 0 leaves the climatology alone.
    """
    d = np.asarray(d26_m, np.float64)
    if not per_c:
        return d
    anomaly = np.asarray(sst_k, np.float64) - 273.15 - np.asarray(sst_clim_c, np.float64)
    factor = np.clip(1.0 + per_c * np.nan_to_num(anomaly, nan=0.0), 0.5, 2.0)
    return np.where(np.isfinite(d), d * factor, d)


class OceanClimatology:
    """
    Monthly depth of the 26 C isotherm (m) and tropical-cyclone heat potential (kJ/cm2, the heat above 26 C down to
    that depth) on a 1-degree grid, computed from the World Ocean Atlas 2023 decadal-average temperature analysis
    (0-1500 m, 57 levels). ``ocean_climatology.npz`` ships with the package; :meth:`default` loads it once.

    Heat potential is what the SHIPS rapid-intensification index uses in place of surface temperature where it
    can, because a storm's fuel is the warm *layer*, not the skin: 28.5 C over 30 m of warm water and 28.5 C over
    120 m are different oceans to a hurricane sitting on them for a day.
    """

    _default = None

    def __init__(self, d26_m: np.ndarray, tchp_kj_cm2: np.ndarray, lat: np.ndarray, lon: np.ndarray, sst_c=None) -> None:
        self.d26 = np.asarray(d26_m, np.float32)                    # (12, 180, 360)
        self.tchp = np.asarray(tchp_kj_cm2, np.float32)
        self.sst = None if sst_c is None else np.asarray(sst_c, np.float32)      # the climatology's own surface, C
        self.lat = np.asarray(lat, np.float64)
        self.lon = np.asarray(lon, np.float64)

    @classmethod
    def default(cls) -> "OceanClimatology | None":
        """The packaged climatology (None when the package was installed without it)."""
        if cls._default is None:
            try:
                from importlib import resources

                with resources.files("naturev1.cyclone").joinpath("ocean_climatology.npz").open("rb") as handle:
                    data = np.load(handle)
                    cls._default = cls(data["d26_m"], data["tchp_kj_cm2"], data["lat"], data["lon"], data.get("sst_c"))
            except Exception:
                cls._default = False
        return cls._default or None

    def _month_weights(self, time) -> tuple[int, int, float]:
        """The two months around ``time`` and the weight of the second (monthly means sit mid-month)."""
        t = np.datetime64(time, "s")
        year = t.astype("datetime64[Y]").astype(int) + 1970
        day_of_year = (t - np.datetime64(f"{year}-01-01", "s")) / np.timedelta64(1, "D")
        pos = day_of_year / 365.25 * 12.0 - 0.5                     # 0 at mid-January
        m0 = int(np.floor(pos)) % 12
        return m0, (m0 + 1) % 12, float(pos - np.floor(pos))

    def sample(self, lat, lon, time) -> dict:
        """``{"d26_m", "tchp_kj_cm2", "sst_c"}`` at points (degrees), bilinear in space between the two months around
        ``time``; land points take the nearest ocean value within a degree or two (NaN beyond)."""
        la = np.atleast_1d(np.asarray(lat, np.float64))
        lo = (np.atleast_1d(np.asarray(lon, np.float64)) + 180.0) % 360.0 - 180.0
        m0, m1, w = self._month_weights(time)
        fi = np.clip((la - self.lat[0]) / (self.lat[1] - self.lat[0]), 0, self.lat.size - 1.001)
        fj = ((lo - self.lon[0]) / (self.lon[1] - self.lon[0])) % self.lon.size
        i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int) % self.lon.size
        di, dj = fi - i0, fj - j0
        i1, j1 = np.minimum(i0 + 1, self.lat.size - 1), (j0 + 1) % self.lon.size
        out = {}
        for name, grid in (("d26_m", self.d26), ("tchp_kj_cm2", self.tchp)) + ((("sst_c", self.sst),) if self.sst is not None else ()):
            g = (1.0 - w) * grid[m0] + w * grid[m1]
            corners = np.stack([g[i0, j0], g[i0, j1], g[i1, j0], g[i1, j1]])
            weights = np.stack([(1 - di) * (1 - dj), (1 - di) * dj, di * (1 - dj), di * dj])
            ok = np.isfinite(corners)
            total = (weights * ok).sum(0)
            value = np.where(total > 0, (np.where(ok, corners, 0.0) * weights).sum(0) / np.maximum(total, 1e-9), np.nan)
            missing = ~np.isfinite(value)
            if missing.any():                                        # coastal points: the nearest ocean cell within 2 degrees
                for k in np.nonzero(missing)[0]:
                    ii = slice(max(i0[k] - 2, 0), min(i0[k] + 3, self.lat.size))
                    jj = np.arange(j0[k] - 2, j0[k] + 3) % self.lon.size
                    patch = g[ii][:, jj]
                    value[k] = np.nanmean(patch) if np.isfinite(patch).any() else np.nan
            out[name] = value
        return out


__all__ = ["COOLING_K", "D26_POWER", "D26_REFERENCE_M", "OceanClimatology", "RESIDENCE_H", "anomaly_adjusted_depth", "core_share",
           "effective_sst", "wake_cooling_k"]
