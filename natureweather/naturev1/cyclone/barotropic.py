# Copyright 2026 Nathan. Apache-2.0.
"""
A limited-area barotropic vorticity model: the storm and its environment evolving together.

The idea (NHC's LBAR, Vic Ooyama's barotropic legacy): a frozen steering flow is wrong by day three, because
ridges and troughs move. The deep-layer flow is, to a good first approximation, barotropic, so it can be
*forecast* by conserving absolute vorticity -- cheaply, without any of the physics a full model needs. And a
vortex embedded in that flow develops its own beta gyres and drifts with them; nothing about beta drift has
to be prescribed. This model:

* lives on a square grid in the storm-centred azimuthal-equidistant plane (exact distances from the storm,
  no pole problem), with the true Coriolis parameter at every point (not a beta-plane approximation);
* starts from the analysis's deep-layer environmental wind (vortex removed, winds rotated into the grid's
  frame), plus a bogus vortex whose outer circulation matches the analysis's (:mod:`naturev1.cyclone.bogus`);
* steps ``dzeta/dt = -J(psi, zeta + f) - nu4 del^4 zeta`` with Arakawa's energy- and enstrophy-conserving
  Jacobian and a third-order strong-stability-preserving Runge-Kutta scheme;
* inverts vorticity for the streamfunction exactly, every stage, with a fast sine transform (Dirichlet
  boundary values from the analysis flow integrated around the domain's edge, held fixed: the large scale
  outside the box does not change -- the limitation LBAR also had);
* relaxes a sponge at the edge back to the analysis, so waves leave instead of reflecting;
* tracks the vortex as the maximum of cyclonic vorticity near its previous position, refined to sub-grid
  precision by a quadratic fit.
"""

from __future__ import annotations

import numpy as np

from .bogus import alpha_for_size, bogus_vorticity
from .fields import Environment
from .geometry import OMEGA, bearing_deg, from_local, hemisphere

GRID_KM = 80.0
CELLS = 73                        # 5760 km across
STEP_S = 900.0
SPONGE_CELLS = 6
SPONGE_TAU_S = 3 * 3600.0
#: Grid-scale damping time of the hyperdiffusion, s.
DAMPING_S = 6 * 3600.0
#: Deep-layer mean used for the environment (top, bottom) hPa, and the bogus's share of the 850 hPa outer wind.
LAYER = (300.0, 850.0)
DEEP_FRACTION = 0.8


def dst1(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Type-I discrete sine transform along ``axis`` via an odd extension and the FFT (no scipy needed)."""
    a = np.moveaxis(a, axis, -1)
    n = a.shape[-1]
    zero = np.zeros(a.shape[:-1] + (1,))
    extended = np.concatenate([zero, a, zero, -a[..., ::-1]], -1)
    spectrum = np.fft.rfft(extended, axis=-1)
    out = -spectrum[..., 1:n + 1].imag / 2.0
    return np.moveaxis(out, -1, axis)


class PoissonSolver:
    """Exact solution of the 5-point ``del^2 psi = zeta`` with Dirichlet boundary values on a square grid."""

    def __init__(self, cells: int, spacing_m: float) -> None:
        self.n, self.h = cells, spacing_m
        m = cells - 2
        k = np.arange(1, m + 1)
        lam = (2.0 * np.cos(np.pi * k / (m + 1)) - 2.0) / spacing_m ** 2
        self.eigen = lam[:, None] + lam[None, :]
        self.norm = (2.0 / (m + 1)) ** 2

    def solve(self, zeta: np.ndarray, boundary: np.ndarray) -> np.ndarray:
        """``zeta`` and ``boundary`` are full ``(n, n)``; boundary values are read from the edge of ``boundary``."""
        h2 = self.h ** 2
        rhs = zeta[1:-1, 1:-1].copy()
        rhs[0, :] -= boundary[0, 1:-1] / h2
        rhs[-1, :] -= boundary[-1, 1:-1] / h2
        rhs[:, 0] -= boundary[1:-1, 0] / h2
        rhs[:, -1] -= boundary[1:-1, -1] / h2
        spec = dst1(dst1(rhs, 0), 1) / self.eigen
        inner = dst1(dst1(spec, 0), 1) * self.norm
        psi = boundary.copy()
        psi[1:-1, 1:-1] = inner
        return psi


def arakawa_jacobian(psi: np.ndarray, q: np.ndarray, h: float) -> np.ndarray:
    """J(psi, q) = psi_x q_y - psi_y q_x on interior points (Arakawa 1966), rows = y, columns = x."""
    p, z = psi, q
    pe, pw, pn, ps = p[1:-1, 2:], p[1:-1, :-2], p[2:, 1:-1], p[:-2, 1:-1]
    qe, qw, qn, qs = z[1:-1, 2:], z[1:-1, :-2], z[2:, 1:-1], z[:-2, 1:-1]
    pne, pnw, pse, psw = p[2:, 2:], p[2:, :-2], p[:-2, 2:], p[:-2, :-2]
    qne, qnw, qse, qsw = z[2:, 2:], z[2:, :-2], z[:-2, 2:], z[:-2, :-2]
    j1 = (pe - pw) * (qn - qs) - (pn - ps) * (qe - qw)
    j2 = pe * (qne - qse) - pw * (qnw - qsw) - pn * (qne - qnw) + ps * (qse - qsw)
    j3 = qn * (pne - pnw) - qs * (pse - psw) - qe * (pne - pse) + qw * (pnw - psw)
    out = np.zeros_like(psi)
    out[1:-1, 1:-1] = (j1 + j2 + j3) / (12.0 * h * h)
    return out


def laplacian(f: np.ndarray, h: float) -> np.ndarray:
    out = np.zeros_like(f)
    out[1:-1, 1:-1] = (f[1:-1, 2:] + f[1:-1, :-2] + f[2:, 1:-1] + f[:-2, 1:-1] - 4 * f[1:-1, 1:-1]) / (h * h)
    return out


class BarotropicModel:
    """
    One storm's barotropic forecast. Build from an analysis (vortex already removed), then :meth:`run`.

    Args:
        env: the environmental analysis (``remove_vortex(...).environment``).
        lat0, lon0: the storm's position.
        outer_ms: the outer strength of its vortex at 850 hPa (:func:`naturev1.cyclone.beta.outer_strength`).
        r34_nmi: best-track gale radius, for the profile's outer decay (NaN for the default).
    """

    def __init__(self, env: Environment, lat0: float, lon0: float, outer_ms: float, r34_nmi: float = float("nan"),
                 cells: int = CELLS, grid_km: float = GRID_KM, rmax_km: float | None = None) -> None:
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self.n, self.h = cells, grid_km * 1000.0
        half = (cells - 1) / 2.0
        axis = (np.arange(cells) - half) * grid_km
        self.x, self.y = np.meshgrid(axis, axis)                          # km; rows y (north), columns x (east)
        lat, lon = from_local(self.lat0, self.lon0, self.x, self.y)
        self.lat, self.lon = lat, lon
        self.f = 2.0 * OMEGA * np.sin(np.radians(lat))
        # The grid's own x-axis direction at each point, as a geographic bearing.
        lat_e, lon_e = from_local(self.lat0, self.lon0, self.x + 1.0, self.y)
        theta = np.radians(bearing_deg(lat, lon, lat_e, lon_e))
        ex_e, ex_n = np.sin(theta), np.cos(theta)
        u = env.sample(env.layer_mean("u", *LAYER), lat, lon)
        v = env.sample(env.layer_mean("v", *LAYER), lat, lon)
        ux = u * ex_e + v * ex_n
        uy = -u * ex_n + v * ex_e
        self.sign = float(hemisphere(self.lat0))
        # Environmental vorticity on the grid, and the boundary streamfunction from the edge flow.
        zeta_env = np.zeros_like(ux)
        zeta_env[1:-1, 1:-1] = ((uy[1:-1, 2:] - uy[1:-1, :-2]) - (ux[2:, 1:-1] - ux[:-2, 1:-1])) / (2 * self.h)
        zeta_env[0, :], zeta_env[-1, :], zeta_env[:, 0], zeta_env[:, -1] = (zeta_env[1, :], zeta_env[-2, :],
                                                                           zeta_env[:, 1], zeta_env[:, -2])
        self.boundary = self._edge_streamfunction(ux, uy)
        r = np.hypot(self.x, self.y)
        rmax = max(rmax_km or 0.0, 2.5 * grid_km)
        self.zeta_env = zeta_env
        self.zeta0 = zeta_env + bogus_vorticity(r, DEEP_FRACTION * outer_ms, rmax, alpha_for_size(r34_nmi), self.sign)
        self.solver = PoissonSolver(cells, self.h)
        k = np.pi / self.h
        self.nu4 = 1.0 / (DAMPING_S * k ** 4)
        distance = np.minimum.reduce([np.arange(cells)[None, :] + 0 * self.x, (cells - 1 - np.arange(cells))[None, :] + 0 * self.x,
                                      np.arange(cells)[:, None] + 0 * self.x, (cells - 1 - np.arange(cells))[:, None] + 0 * self.x])
        self.sponge = np.where(distance < SPONGE_CELLS, (1.0 - distance / SPONGE_CELLS) ** 2 / SPONGE_TAU_S, 0.0)

    def _edge_streamfunction(self, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        """psi on the edge from d psi = v dx - u dy integrated counter-clockwise; net divergence spread evenly."""
        n, h = self.n, self.h
        loop = ([(0, j) for j in range(n)] + [(i, n - 1) for i in range(1, n)]
                + [(n - 1, j) for j in range(n - 2, -1, -1)] + [(i, 0) for i in range(n - 2, 0, -1)])
        values = [0.0]
        for (i0, j0), (i1, j1) in zip(loop[:-1], loop[1:]):
            um = 0.5 * (ux[i0, j0] + ux[i1, j1])
            vm = 0.5 * (uy[i0, j0] + uy[i1, j1])
            dx, dy = (j1 - j0) * h, (i1 - i0) * h
            values.append(values[-1] + vm * dx - um * dy)
        i0, j0 = loop[-1]
        i1, j1 = loop[0]
        closing = values[-1] + 0.5 * (uy[i0, j0] + uy[i1, j1]) * (j1 - j0) * h - 0.5 * (ux[i0, j0] + ux[i1, j1]) * (i1 - i0) * h
        boundary = np.zeros((n, n))
        total = len(loop)
        for k, (i, j) in enumerate(loop):
            boundary[i, j] = values[k] - closing * k / total
        return boundary

    def _tendency(self, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi = self.solver.solve(zeta, self.boundary)
        tend = -arakawa_jacobian(psi, zeta + self.f, self.h)
        tend -= self.nu4 * laplacian(laplacian(zeta, self.h), self.h)
        tend -= self.sponge * (zeta - self.zeta_env)
        tend[0, :] = tend[-1, :] = tend[:, 0] = tend[:, -1] = 0.0
        return tend, psi

    def _locate(self, zeta: np.ndarray, near: tuple[float, float], search_km: float = 450.0) -> tuple[float, float]:
        field = self.sign * zeta
        smooth = field.copy()
        smooth[1:-1, 1:-1] = (field[1:-1, 1:-1] * 4 + field[1:-1, 2:] + field[1:-1, :-2] + field[2:, 1:-1]
                              + field[:-2, 1:-1]) / 8.0
        mask = np.hypot(self.x - near[0], self.y - near[1]) < search_km
        mask[:2, :] = mask[-2:, :] = mask[:, :2] = mask[:, -2:] = False
        if not mask.any():
            return near
        candidate = np.where(mask, smooth, -np.inf)
        i, j = np.unravel_index(int(np.argmax(candidate)), candidate.shape)
        c, e, w, nn, s = smooth[i, j], smooth[i, j + 1], smooth[i, j - 1], smooth[i + 1, j], smooth[i - 1, j]
        dx = 0.5 * (w - e) / (w - 2 * c + e) if (w - 2 * c + e) < 0 else 0.0
        dy = 0.5 * (s - nn) / (s - 2 * c + nn) if (s - 2 * c + nn) < 0 else 0.0
        step = self.h / 1000.0
        return float(self.x[i, j] + np.clip(dx, -0.5, 0.5) * step), float(self.y[i, j] + np.clip(dy, -0.5, 0.5) * step)

    def run(self, leads, return_fields: bool = False) -> dict:
        """Integrate to the last lead; returns ``{"track": (L, 2) lat/lon, "xy": (L, 2) km, "speed_ms": (L,)}``."""
        leads = [float(h) for h in leads]
        zeta = self.zeta0.copy()
        position = (0.0, 0.0)
        out_xy, fields = [], []
        t, k = 0.0, 0
        end = max(leads)
        dt = STEP_S
        while k < len(leads) and leads[k] <= 0:
            out_xy.append(position)
            k += 1
        steps_per_hour = 3600.0 / dt
        step = 0
        while t < end - 1e-9:
            k1, _ = self._tendency(zeta)
            z1 = zeta + dt * k1
            k2, _ = self._tendency(z1)
            z2 = 0.75 * zeta + 0.25 * (z1 + dt * k2)
            k3, _ = self._tendency(z2)
            zeta = zeta / 3.0 + 2.0 / 3.0 * (z2 + dt * k3)
            step += 1
            t = step / steps_per_hour
            if step % int(steps_per_hour) == 0:
                position = self._locate(zeta, position)
            while k < len(leads) and leads[k] <= t + 1e-9:
                position = self._locate(zeta, position)
                out_xy.append(position)
                if return_fields:
                    fields.append(zeta.copy())
                k += 1
        xy = np.asarray(out_xy)
        lat, lon = from_local(self.lat0, self.lon0, xy[:, 0], xy[:, 1])
        result = {"track": np.stack([lat, lon], -1), "xy": xy}
        if return_fields:
            result["zeta"] = fields
        return result


def barotropic_track(env_clean: Environment, case, outer_ms: float, leads) -> np.ndarray:
    """``(L, 2)`` lat/lon from one barotropic forecast of ``case``."""
    model = BarotropicModel(env_clean, case.lat, case.lon, outer_ms, getattr(case, "r34_nmi", float("nan")))
    return model.run(leads)["track"]


__all__ = ["BarotropicModel", "PoissonSolver", "arakawa_jacobian", "barotropic_track", "dst1", "laplacian"]
