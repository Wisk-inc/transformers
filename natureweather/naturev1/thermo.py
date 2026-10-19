# Copyright 2026 Nathan. Apache-2.0.
"""
Tropical-cyclone thermodynamics from the model's own atmosphere: the ceiling, the brakes and the fuel.

The intensity heads were told the ocean temperature and the shear and left to work out the rest from a
pooled vector. Operational intensity models are told much more, and every one of the quantities here is
something a forecaster's statistical-dynamical guidance reads explicitly:

* **Potential intensity** (Bister and Emanuel 2002, the algorithm behind ``tcpyPI``): the strongest wind
  the ocean and the column above it can sustain, from the sea-surface temperature, the surface pressure
  and the temperature and humidity at every pressure level -- computed by lifting parcels through the
  model's own 13-level column, iterating the eyewall pressure to self-consistency, with dissipative
  heating. It sets the ceiling. A storm 80 kt below it can explode; one at it cannot.
* **Ventilation** (Tang and Emanuel 2012): shear times the mid-level entropy deficit, over the potential
  intensity -- the dry air that shear drives into the core. The single best thermodynamic brake on
  intensification that the literature has.
* **Genesis potential** (Emanuel and Nolan 2004): absolute vorticity, mid-level humidity, potential
  intensity and shear in one index -- where storms can form, everywhere on the globe at once.
* **Cyclone phase space** (Hart 2003): thermal asymmetry across the track and the thermal-wind warm- or
  cold-core signature below and above 600 hPa -- tropical, transitioning or extratropical, measured.
* **The SHIPS predictors** the backbone's pooled vector smears: upper-level divergence (outflow), low-level
  vorticity, mid- and low-level humidity, upper-level temperature, boundary-layer equivalent potential
  temperature and CAPE, water vapour, rain in the core, land under the storm.

Everything works on the normalized model state ``(B, P, C)`` on a regular latitude-longitude grid, with the
dataset's channel names and normalizer, so it reads the analysis now or any step of a rollout.
"""

from __future__ import annotations

import math

import torch

#: Dry-air gas constant, water-vapour gas constant (J kg-1 K-1), heat capacity of dry air at constant pressure,
#: latent heat of vaporisation at 0 C, gravity, ratio of gas constants.
RD, RV, CP, LV, GRAVITY = 287.04, 461.5, 1005.7, 2.501e6, 9.80665
EPS = RD / RV
#: Exchange-coefficient ratio Ck/Cd and the gradient-to-surface wind reduction used with potential intensity.
CK_CD = 0.9
SURFACE_REDUCTION = 0.8
EARTH_RADIUS_M = 6.371e6
OMEGA = 7.292e-5

#: The physics vector handed to the storm heads, in this order, each scaled to order one.
STORM_PHYSICS_FEATURES = (
    "potential_intensity", "intensity_gap", "outflow_temperature", "ventilation", "entropy_deficit", "cape",
    "genesis_potential", "phase_b", "phase_vtl", "phase_vtu", "divergence_200", "vorticity_850", "rh_mid",
    "rh_low", "temperature_200", "theta_e_boundary", "water_vapour", "core_rain", "land_share",
    "translation_speed", "steering_shear_u", "steering_shear_v", "observed",
)


# ------------------------------------------------------------------------------------- moist air --

def saturation_vapour_pressure(temperature_k: torch.Tensor) -> torch.Tensor:
    """hPa, over liquid water (Bolton 1980)."""
    celsius = temperature_k - 273.15
    return 6.112 * torch.exp(17.67 * celsius / (celsius + 243.5))


def saturation_mixing_ratio(temperature_k: torch.Tensor, pressure_hpa: torch.Tensor) -> torch.Tensor:
    es = saturation_vapour_pressure(temperature_k)
    return EPS * es / (pressure_hpa - es).clamp_min(1.0)


def mixing_ratio(specific_humidity: torch.Tensor) -> torch.Tensor:
    q = specific_humidity.clamp(0.0, 0.1)
    return q / (1.0 - q)


def vapour_pressure(mixing: torch.Tensor, pressure_hpa: torch.Tensor) -> torch.Tensor:
    return pressure_hpa * mixing / (EPS + mixing)


def relative_humidity(temperature_k, specific_humidity, pressure_hpa) -> torch.Tensor:
    """Fraction, clamped to [0, 1.2] (supersaturation in an analysis is noise, not physics)."""
    e = vapour_pressure(mixing_ratio(specific_humidity), pressure_hpa)
    return (e / saturation_vapour_pressure(temperature_k)).clamp(0.0, 1.2)


def virtual_temperature(temperature_k, mixing) -> torch.Tensor:
    return temperature_k * (1.0 + mixing / EPS) / (1.0 + mixing)


def lcl_temperature(temperature_k, mixing, pressure_hpa) -> torch.Tensor:
    """Temperature at the lifting condensation level (Bolton 1980, eq. 21)."""
    e = vapour_pressure(mixing, pressure_hpa).clamp_min(1e-6)
    return 2840.0 / (3.5 * torch.log(temperature_k) - torch.log(e) - 4.805) + 55.0


def equivalent_potential_temperature(temperature_k, mixing, pressure_hpa) -> torch.Tensor:
    """Pseudo-equivalent potential temperature (Bolton 1980, eq. 43), K."""
    grams = mixing * 1000.0
    lcl = lcl_temperature(temperature_k, mixing, pressure_hpa)
    theta = temperature_k * (1000.0 / pressure_hpa) ** (0.2854 * (1.0 - 0.28e-3 * grams))
    return theta * torch.exp((3.376 / lcl - 0.00254) * grams * (1.0 + 0.81e-3 * grams))


def saturation_theta_e(temperature_k, pressure_hpa) -> torch.Tensor:
    mixing = saturation_mixing_ratio(temperature_k, pressure_hpa)
    grams = mixing * 1000.0
    theta = temperature_k * (1000.0 / pressure_hpa) ** (0.2854 * (1.0 - 0.28e-3 * grams))
    return theta * torch.exp((3.376 / temperature_k - 0.00254) * grams * (1.0 + 0.81e-3 * grams))


def moist_entropy(temperature_k, mixing, pressure_hpa, saturated: bool = False) -> torch.Tensor:
    """Moist entropy per unit dry mass (Emanuel 1994, neglecting condensate), J kg-1 K-1."""
    if saturated:
        mixing = saturation_mixing_ratio(temperature_k, pressure_hpa)
        humidity = torch.ones_like(temperature_k)
    else:
        humidity = (vapour_pressure(mixing, pressure_hpa) / saturation_vapour_pressure(temperature_k)).clamp(1e-4, 1.0)
    dry_pressure = (pressure_hpa - vapour_pressure(mixing, pressure_hpa)).clamp_min(1.0)
    return (CP * torch.log(temperature_k) - RD * torch.log(dry_pressure) + LV * mixing / temperature_k
            - RV * mixing * torch.log(humidity))


def parcel_temperature(theta_e: torch.Tensor, pressure_hpa: torch.Tensor, iterations: int = 48) -> torch.Tensor:
    """The saturated temperature at ``pressure_hpa`` with pseudo-equivalent potential temperature ``theta_e``."""
    low = torch.full_like(theta_e, 140.0)
    high = torch.full_like(theta_e, 360.0)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        warmer = saturation_theta_e(middle, pressure_hpa) > theta_e
        high = torch.where(warmer, middle, high)
        low = torch.where(warmer, low, middle)
    return 0.5 * (low + high)


# ------------------------------------------------------------------------------- parcels and CAPE --

def cape(parcel_t: torch.Tensor, parcel_r: torch.Tensor, parcel_p: torch.Tensor, levels_hpa: torch.Tensor,
         env_t: torch.Tensor, env_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convective available potential energy of a parcel lifted pseudo-adiabatically through a column.

    Args:
        parcel_t, parcel_r, parcel_p: ``(B,)`` starting temperature (K), mixing ratio, pressure (hPa).
        levels_hpa: ``(L,)`` environment pressure levels, any order.
        env_t, env_r: ``(B, L)`` environment temperature and mixing ratio on those levels.

    Returns:
        ``(cape J/kg, outflow temperature K, level of neutral buoyancy hPa)``. CAPE integrates positive and
        negative buoyancy from the parcel's level up to the highest level where it is still buoyant (the
        zero crossing interpolated in log-pressure); without buoyancy anywhere it is zero, with NaN outflow.
    """
    order = torch.argsort(levels_hpa, descending=True)
    levels = levels_hpa[order].to(env_t.dtype)
    env_t, env_r = env_t[:, order], env_r[:, order]
    batch = env_t.shape[0]
    # The parcel's own level, then every environment level above it.
    log_levels = torch.log(levels).view(1, -1).expand(batch, -1)
    log_p0 = torch.log(parcel_p).view(-1, 1)
    # Environment at the parcel level, by log-pressure interpolation (or extrapolation below the lowest level).
    index = torch.searchsorted(-levels.contiguous(), -parcel_p.contiguous()).clamp(1, len(levels) - 1)
    lo, hi = index - 1, index
    w = ((log_p0[:, 0] - log_levels.gather(1, lo.view(-1, 1))[:, 0])
         / (log_levels.gather(1, hi.view(-1, 1))[:, 0] - log_levels.gather(1, lo.view(-1, 1))[:, 0]))
    env_t0 = env_t.gather(1, lo.view(-1, 1))[:, 0] * (1 - w) + env_t.gather(1, hi.view(-1, 1))[:, 0] * w
    env_r0 = env_r.gather(1, lo.view(-1, 1))[:, 0] * (1 - w) + env_r.gather(1, hi.view(-1, 1))[:, 0] * w
    pressures = torch.cat([parcel_p.view(-1, 1), levels.view(1, -1).expand(batch, -1)], 1)
    temperatures = torch.cat([env_t0.view(-1, 1), env_t], 1)
    mixings = torch.cat([env_r0.clamp_min(0).view(-1, 1), env_r], 1)
    above = torch.cat([torch.ones(batch, 1, dtype=torch.bool, device=env_t.device),
                       levels.view(1, -1) < parcel_p.view(-1, 1)], 1)

    lcl_t = lcl_temperature(parcel_t, parcel_r, parcel_p)
    lcl_p = parcel_p * (lcl_t / parcel_t) ** (1.0 / 0.2854)
    theta_e = equivalent_potential_temperature(parcel_t, parcel_r, parcel_p)
    dry_t = parcel_t.view(-1, 1) * (pressures / parcel_p.view(-1, 1)) ** 0.2854
    moist_t = parcel_temperature(theta_e.view(-1, 1).expand_as(pressures), pressures)
    unsaturated = pressures >= lcl_p.view(-1, 1)
    lifted_t = torch.where(unsaturated, dry_t, moist_t)
    lifted_r = torch.where(unsaturated, parcel_r.view(-1, 1).expand_as(pressures),
                           saturation_mixing_ratio(moist_t, pressures))
    buoyancy = RD * (virtual_temperature(lifted_t, lifted_r) - virtual_temperature(temperatures, mixings))
    buoyancy = torch.where(above, buoyancy, torch.zeros_like(buoyancy))

    positive = (buoyancy > 0) & above
    count = pressures.shape[1]
    position = torch.arange(count, device=env_t.device).view(1, -1).expand(batch, -1)
    top = torch.where(positive, position, torch.full_like(position, -1)).max(1).values      # last buoyant point
    has = top >= 0
    log_p = torch.log(pressures)
    segment = 0.5 * (buoyancy[:, :-1] + buoyancy[:, 1:]) * (log_p[:, :-1] - log_p[:, 1:])
    inside = (position[:, 1:] <= top.view(-1, 1)) & above[:, 1:]
    total = (segment * inside).sum(1)
    # The partial segment from the last buoyant point to the zero crossing above it.
    nxt = (top + 1).clamp(max=count - 1)
    b_top = buoyancy.gather(1, top.clamp_min(0).view(-1, 1))[:, 0]
    b_next = buoyancy.gather(1, nxt.view(-1, 1))[:, 0]
    can = has & (top < count - 1) & (b_next < 0)
    frac = torch.where(can, b_top / (b_top - b_next).clamp_min(1e-9), torch.zeros_like(b_top))
    lp_top = log_p.gather(1, top.clamp_min(0).view(-1, 1))[:, 0]
    lp_next = log_p.gather(1, nxt.view(-1, 1))[:, 0]
    lp_lnb = lp_top + frac * (lp_next - lp_top)
    total = total + torch.where(can, 0.5 * b_top * (lp_top - lp_lnb), torch.zeros_like(total))
    t_top = temperatures.gather(1, top.clamp_min(0).view(-1, 1))[:, 0]
    t_next = temperatures.gather(1, nxt.view(-1, 1))[:, 0]
    outflow = t_top + frac * (t_next - t_top)
    nan = torch.full_like(outflow, float("nan"))
    return torch.where(has, total, torch.zeros_like(total)), torch.where(has, outflow, nan), torch.where(has, lp_lnb.exp(), nan)


def potential_intensity(sst_k: torch.Tensor, msl_hpa: torch.Tensor, levels_hpa: torch.Tensor, env_t: torch.Tensor,
                        env_q: torch.Tensor, iterations: int = 40, ck_cd: float = CK_CD,
                        reduction: float = SURFACE_REDUCTION) -> dict[str, torch.Tensor]:
    """
    Potential intensity after Bister and Emanuel (2002), in the CAPE formulation of Emanuel's ``pcmin``.

    The maximum gradient wind is ``V^2 = (Ck/Cd) (Ts / To) (CAPE*_m - CAPE_m)``: the CAPE of air saturated at
    the sea-surface temperature at the eyewall pressure, minus that of boundary-layer air there, times the
    dissipative-heating factor (sea surface over outflow temperature). The eyewall pressure itself depends
    on the answer, so it is iterated: hydrostatic in the boundary layer, ``p_m = p_s exp(-CAT / (Rd Tv))``.

    Args:
        sst_k: ``(B,)`` sea-surface temperature. msl_hpa: ``(B,)`` ambient surface pressure.
        levels_hpa: ``(L,)``. env_t: ``(B, L)`` K. env_q: ``(B, L)`` specific humidity, kg/kg.

    Returns:
        ``{"vmax_ms": surface wind (x reduction), "pmin_hpa", "outflow_k", "efficiency", "converged"}``;
        zero wind where no saturated parcel is buoyant (too cold an ocean).
    """
    dtype = torch.float64
    sst, msl = sst_k.to(dtype), msl_hpa.to(dtype)
    levels = levels_hpa.to(env_t.device, dtype)
    t, r = env_t.to(dtype), mixing_ratio(env_q.to(dtype))
    lowest = int(torch.argmax(levels))
    t_b, r_b, p_b = t[:, lowest], r[:, lowest], levels[lowest].expand_as(sst)
    cape_a, _, _ = cape(t_b, r_b, p_b, levels, t, r)
    pm = torch.minimum(msl - 30.0, torch.full_like(msl, 970.0))
    converged = torch.zeros_like(msl, dtype=torch.bool)
    for _ in range(iterations):
        cape_m, _, _ = cape(t_b, r_b, pm, levels, t, r)
        rs0 = saturation_mixing_ratio(sst, pm)
        cape_ms, outflow, _ = cape(sst, rs0, pm, levels, t, r)
        ratio = torch.where(torch.isfinite(outflow), sst / outflow, torch.ones_like(sst))
        tv_av = 0.5 * (virtual_temperature(t_b, r_b) + virtual_temperature(sst, rs0))
        cat = (cape_m - cape_a + 0.5 * ck_cd * ratio * (cape_ms - cape_m)).clamp_min(0.0)
        updated = msl * torch.exp(-cat / (RD * tv_av))
        converged = (updated - pm).abs() < 0.2
        pm = torch.where(converged, pm, 0.5 * (pm + updated)).clamp(600.0, 1050.0)
    energy = (cape_ms - cape_m).clamp_min(0.0)
    vmax = torch.sqrt(ck_cd * ratio * energy) * reduction
    usable = torch.isfinite(outflow) & torch.isfinite(vmax)
    return {"vmax_ms": torch.where(usable, vmax, torch.zeros_like(vmax)),
            "pmin_hpa": torch.where(usable, pm, msl),
            "outflow_k": outflow,
            "efficiency": torch.where(usable, (sst - outflow) / outflow, torch.zeros_like(sst)),
            "converged": converged}


def ventilation_index(shear_ms, entropy_deficit, potential_ms) -> torch.Tensor:
    """Tang and Emanuel (2012): shear x entropy deficit / potential intensity. Above ~0.1 intensification stalls."""
    return shear_ms * entropy_deficit / potential_ms.clamp_min(5.0)


def entropy_deficit(sst_k, msl_hpa, t_mid, r_mid, p_mid, t_bl, r_bl, p_bl) -> torch.Tensor:
    """
    The non-dimensional mid-level entropy deficit of Tang and Emanuel (2012):
    ``(s*_m - s_m) / (s*_sst - s_b)`` -- how much drier than saturation the middle troposphere is,
    measured against the air-sea disequilibrium that fuels the storm.
    """
    s_star_m = moist_entropy(t_mid, r_mid, p_mid, saturated=True)
    s_m = moist_entropy(t_mid, r_mid, p_mid)
    s_star_sst = moist_entropy(sst_k, None, msl_hpa, saturated=True)
    s_b = moist_entropy(t_bl, r_bl, p_bl)
    return ((s_star_m - s_m) / (s_star_sst - s_b).clamp_min(5.0)).clamp(0.0, 5.0)


def genesis_potential_index(absolute_vorticity_850, rh_600_percent, potential_ms, shear_ms) -> torch.Tensor:
    """Emanuel and Nolan (2004): |1e5 eta|^1.5 (H/50)^3 (Vpot/70)^3 (1 + 0.1 Vshear)^-2."""
    return ((1e5 * absolute_vorticity_850).abs() ** 1.5 * (rh_600_percent.clamp(0, 100) / 50.0) ** 3
            * (potential_ms.clamp_min(0) / 70.0) ** 3 * (1.0 + 0.1 * shear_ms.clamp_min(0)) ** -2)


# ---------------------------------------------------------------------------- fields on the grid --

class GridFields:
    """
    Physical fields from a normalized regular-grid state, with spherical derivatives and storm-relative means.

    Args:
        state: ``(B, P, C)`` normalized, ``P = rows * cols`` latitude-major.
        variables / normalizer / grid: as for :func:`naturev1.storm_environment`.
    """

    def __init__(self, state: torch.Tensor, variables, normalizer, grid) -> None:
        from .steering import grid_latlon

        self.state, self.variables, self.normalizer = state, list(variables), normalizer
        latitude, longitude = grid_latlon(grid)
        self.latitude, self.longitude = latitude.to(state.device), longitude.to(state.device)
        cols = int((latitude == latitude[0]).sum())
        self.rows, self.cols = latitude.numel() // cols, cols
        self.area = grid.weights.to(state.device, torch.float64)
        self._cache: dict[str, torch.Tensor] = {}

    def has(self, name: str) -> bool:
        return name in self.variables

    def levels(self, variable: str) -> list[int]:
        return sorted(int(v.split("@")[1]) for v in self.variables if v.startswith(variable + "@"))

    def field(self, name: str) -> torch.Tensor:
        """``(B, P)`` physical values (float64)."""
        if name not in self._cache:
            values = self.state[..., self.variables.index(name)].to(torch.float64)
            self._cache[name] = torch.as_tensor(self.normalizer.decode(values, name), dtype=torch.float64)
        return self._cache[name]

    def grid2d(self, values: torch.Tensor) -> torch.Tensor:
        return values.view(values.shape[0], self.rows, self.cols)

    def _metric(self):
        lat = self.latitude.view(self.rows, self.cols)[:, 0]
        lon = self.longitude.view(self.rows, self.cols)[0]
        dlat = torch.deg2rad(lat[1] - lat[0]) if self.rows > 1 else torch.tensor(1.0)
        dlon = torch.deg2rad(((lon[1] - lon[0] + 180) % 360) - 180) if self.cols > 1 else torch.tensor(1.0)
        cos = torch.cos(torch.deg2rad(lat)).clamp_min(1e-3)
        return dlat.to(torch.float64), dlon.to(torch.float64), cos.to(torch.float64)

    def _ddx(self, field2d: torch.Tensor) -> torch.Tensor:
        _, dlon, cos = self._metric()
        return (torch.roll(field2d, -1, -1) - torch.roll(field2d, 1, -1)) / (
            2 * dlon * EARTH_RADIUS_M * cos.view(1, -1, 1))

    def _ddy(self, field2d: torch.Tensor) -> torch.Tensor:
        dlat, _, _ = self._metric()
        out = torch.zeros_like(field2d)
        out[:, 1:-1] = (field2d[:, 2:] - field2d[:, :-2]) / (2 * dlat * EARTH_RADIUS_M)
        out[:, 0], out[:, -1] = out[:, 1], out[:, -2]
        return out

    def divergence(self, level: int) -> torch.Tensor:
        u = self.grid2d(self.field(f"u_component_of_wind@{level}"))
        v = self.grid2d(self.field(f"v_component_of_wind@{level}"))
        _, _, cos = self._metric()
        div = self._ddx(u) + self._ddy(v * cos.view(1, -1, 1)) / cos.view(1, -1, 1)
        return div.reshape(div.shape[0], -1)

    def vorticity(self, level: int) -> torch.Tensor:
        u = self.grid2d(self.field(f"u_component_of_wind@{level}"))
        v = self.grid2d(self.field(f"v_component_of_wind@{level}"))
        _, _, cos = self._metric()
        vort = self._ddx(v) - self._ddy(u * cos.view(1, -1, 1)) / cos.view(1, -1, 1)
        return vort.reshape(vort.shape[0], -1)

    def coriolis(self) -> torch.Tensor:
        return 2 * OMEGA * torch.sin(torch.deg2rad(self.latitude)).to(torch.float64)

    def weights(self, centre: torch.Tensor, inner_km: float, outer_km: float) -> torch.Tensor:
        from .steering import great_circle_km

        centre = centre.to(self.state.device, torch.float64)
        distance = great_circle_km(centre[:, :1], centre[:, 1:], self.latitude[None], self.longitude[None])
        ring = (distance >= inner_km) & (distance <= outer_km)
        nearest = torch.zeros_like(ring).scatter_(1, distance.argmin(-1, keepdim=True), True)
        ring = torch.where(ring.sum(-1, keepdim=True) >= 1, ring, nearest)
        weights = ring.to(torch.float64) * self.area[None]
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)

    def mean(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (torch.nan_to_num(values) * weights).sum(-1)

    def distance_km(self, centre: torch.Tensor) -> torch.Tensor:
        from .steering import great_circle_km

        centre = centre.to(self.state.device, torch.float64)
        return great_circle_km(centre[:, :1], centre[:, 1:], self.latitude[None], self.longitude[None])


def _level_profiles(fields: GridFields, weights: torch.Tensor) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Area-mean temperature and specific humidity profiles on the levels that have both."""
    levels = [lv for lv in fields.levels("temperature") if fields.has(f"specific_humidity@{lv}") and lv >= 100]
    if not levels:
        return [], torch.zeros(0), torch.zeros(0)
    t = torch.stack([fields.mean(fields.field(f"temperature@{lv}"), weights) for lv in levels], -1)
    q = torch.stack([fields.mean(fields.field(f"specific_humidity@{lv}"), weights) for lv in levels], -1)
    return levels, t, q


def cyclone_phase(fields: GridFields, centre: torch.Tensor, motion: torch.Tensor | None = None,
                  radius_km: float = 500.0) -> dict[str, torch.Tensor]:
    """
    Hart (2003) cyclone phase space around each centre.

    ``B``: the 600-900 hPa thickness on the right of the track minus the left (hemisphere-signed), m --
    near zero for a symmetric tropical cyclone, above ~10 m for a frontal, transitioning one.
    ``-VTL`` / ``-VTU``: the thermal wind of the height perturbation between 900-600 and 600-300 hPa,
    positive for a warm core, negative for a cold core. 925 hPa stands in for 900.
    """
    batch = centre.shape[0]
    zero = torch.zeros(batch, dtype=torch.float64, device=fields.state.device)
    heights = {lv: fields.field(f"geopotential@{lv}") / GRAVITY for lv in fields.levels("geopotential")}
    distance = fields.distance_km(centre)
    inside = distance <= radius_km
    out = {"phase_b": zero.clone(), "phase_vtl": zero.clone(), "phase_vtu": zero.clone(), "phase_ok": zero.clone()}
    lower = [lv for lv in (925, 850, 700, 600) if lv in heights]
    upper = [lv for lv in (600, 500, 400, 300) if lv in heights]
    if len(lower) < 2 or len(upper) < 2:
        return out

    def thermal_wind(levels):
        dz = torch.stack([torch.where(inside, heights[lv], torch.full_like(heights[lv], -math.inf)).max(-1).values
                          - torch.where(inside, heights[lv], torch.full_like(heights[lv], math.inf)).min(-1).values
                          for lv in levels], -1)
        x = torch.log(torch.tensor(levels, dtype=torch.float64, device=dz.device))
        x = x - x.mean()
        return ((dz - dz.mean(-1, keepdim=True)) * x).sum(-1) / (x ** 2).sum()

    # Hart's -V_T is the slope of the height perturbation against log-pressure: a warm core's perturbation
    # shrinks with height (grows with ln p), so warm is positive and cold negative.
    out["phase_vtl"] = thermal_wind(lower)
    out["phase_vtu"] = thermal_wind(upper)
    if 600 in heights and lower[0] in heights:
        thickness = heights[600] - heights[lower[0]]
        centre64 = centre.to(fields.state.device, torch.float64)
        if motion is None:
            heading = torch.zeros(batch, dtype=torch.float64, device=fields.state.device)     # due north
        else:
            heading = torch.atan2(motion[:, 1].to(torch.float64), motion[:, 0].to(torch.float64))
        north = (fields.latitude[None] - centre64[:, :1])
        east = ((fields.longitude[None] - centre64[:, 1:] + 180) % 360 - 180) * torch.cos(torch.deg2rad(centre64[:, :1]))
        across = east * torch.cos(heading).view(-1, 1) - north * torch.sin(heading).view(-1, 1)    # >0 right of track
        sign = torch.where(centre64[:, 0] >= 0, 1.0, -1.0).view(-1, 1)
        right, left = inside & (across * sign > 0), inside & (across * sign < 0)
        mean_r = (thickness * right).sum(-1) / right.sum(-1).clamp_min(1)
        mean_l = (thickness * left).sum(-1) / left.sum(-1).clamp_min(1)
        out["phase_b"] = mean_r - mean_l
    out["phase_ok"] = torch.ones_like(zero)
    return out


@torch.no_grad()
def storm_physics(state: torch.Tensor, variables, normalizer, grid, storm_center: torch.Tensor,
                  storm_state: torch.Tensor | None = None, chunk: int = 128) -> torch.Tensor:
    """
    The thermodynamic and dynamic predictors around each storm, scaled to order one, in
    :data:`STORM_PHYSICS_FEATURES` order. ``observed`` is 0 when the state has no pressure-level winds,
    temperature or humidity (then everything else is 0 too).

    Units before scaling: potential intensity and the gap below it in m/s (/50); outflow temperature K
    (-200, /15); ventilation (x10); entropy deficit; CAPE J/kg (/2000); genesis potential (log1p);
    phase B m (/20), -VTL and -VTU m (/100); 200 hPa divergence and 850 hPa vorticity 1e-5 s-1; relative
    humidity fraction; 200 hPa temperature C (+55, /5); boundary-layer theta-e K (-350, /10); water vapour
    kg/m2 (/50); rain mm/6 h in the core (/10); land share; translation speed m/s (/10); deep-minus-shallow
    steering m/s (/10).
    """
    from .steering import _Reader, _steering_wind

    rows = []
    for start in range(0, state.shape[0], chunk):
        sub = state[start : start + chunk]
        centre = storm_center[start : start + chunk].to(state.device, torch.float64)
        fields = GridFields(sub, variables, normalizer, grid)
        batch = sub.shape[0]
        out = torch.zeros(batch, len(STORM_PHYSICS_FEATURES), dtype=torch.float64, device=state.device)
        core = fields.weights(centre, 0.0, 500.0)
        ring = fields.weights(centre, 200.0, 800.0)
        wide = fields.weights(centre, 0.0, 1000.0)
        levels, t_prof, q_prof = _level_profiles(fields, core)
        has_winds = fields.has("u_component_of_wind@200") and fields.has("u_component_of_wind@850")
        if not levels or not has_winds:
            rows.append(out.float())
            continue
        msl = (fields.mean(fields.field("mean_sea_level_pressure"), ring) / 100.0 if fields.has("mean_sea_level_pressure")
               else torch.full((batch,), 1012.0, dtype=torch.float64, device=state.device))
        sst = None
        if fields.has("sea_surface_temperature"):
            raw = fields.field("sea_surface_temperature")
            ocean = (raw > 250.0).to(torch.float64) * core
            share = ocean.sum(-1)
            sst = (raw * ocean).sum(-1) / share.clamp_min(1e-30)
            sst = torch.where(share > 0.2, sst, torch.full_like(sst, float("nan")))
        if sst is None or not bool(torch.isfinite(sst).any()):
            fallback = fields.mean(fields.field("2m_temperature"), core) if fields.has("2m_temperature") else t_prof[:, 0]
            sst = fallback if sst is None else torch.where(torch.isfinite(sst), sst, fallback)
        sst = torch.where(torch.isfinite(sst), sst, t_prof[:, 0])
        level_tensor = torch.tensor(levels, dtype=torch.float64, device=state.device)
        pi = potential_intensity(sst, msl, level_tensor, t_prof, q_prof)
        wind_now = None
        if storm_state is not None:
            st = torch.nan_to_num(storm_state[start : start + chunk].to(state.device, torch.float64))
            from .model import PEAK_WIND_ANCHOR_KT

            wind_now = torch.where(st[:, 5] > 0.5, (st[:, 0] * PEAK_WIND_ANCHOR_KT[1] + PEAK_WIND_ANCHOR_KT[0]) * 0.514444,
                                   torch.full_like(st[:, 0], 25.0))
            motion_deg = st[:, 2:4]
        else:
            motion_deg = torch.zeros(batch, 2, dtype=torch.float64, device=state.device)
        reader = _Reader(sub, variables, normalizer, grid)
        top = 200 if fields.has("u_component_of_wind@200") else max(fields.levels("u_component_of_wind"))
        shear_u = fields.mean(fields.field(f"u_component_of_wind@{top}") - fields.field("u_component_of_wind@850"), ring)
        shear_v = fields.mean(fields.field(f"v_component_of_wind@{top}") - fields.field("v_component_of_wind@850"), ring)
        shear = torch.hypot(shear_u, shear_v)
        mid = 600 if 600 in levels else min(levels, key=lambda lv: abs(lv - 600))
        bl = max(levels)
        k_mid, k_bl = levels.index(mid), levels.index(bl)
        chi = entropy_deficit(sst, msl, t_prof[:, k_mid], mixing_ratio(q_prof[:, k_mid]),
                              torch.full_like(sst, float(mid)), t_prof[:, k_bl], mixing_ratio(q_prof[:, k_bl]),
                              torch.full_like(sst, float(bl)))
        vent = ventilation_index(shear, chi, pi["vmax_ms"])
        cape_env, _, _ = cape(t_prof[:, k_bl], mixing_ratio(q_prof[:, k_bl]), torch.full_like(sst, float(bl)),
                              level_tensor, t_prof, mixing_ratio(q_prof))
        vort850 = fields.mean(fields.vorticity(850), wide)
        eta = vort850 + 2 * OMEGA * torch.sin(torch.deg2rad(centre[:, 0]))
        cyclonic = torch.where(centre[:, 0] >= 0, 1.0, -1.0).to(vort850.dtype)     # cyclonic positive in both
        rh = {lv: relative_humidity(fields.field(f"temperature@{lv}"), fields.field(f"specific_humidity@{lv}"),
                                    torch.tensor(float(lv), dtype=torch.float64, device=state.device))
              for lv in levels if lv in (850, 700, 600, 500)}
        rh600 = 100.0 * (fields.mean(rh[600], ring) if 600 in rh
                         else relative_humidity(t_prof[:, k_mid], q_prof[:, k_mid], torch.full_like(sst, float(mid))))
        gpi = genesis_potential_index(eta, rh600, pi["vmax_ms"], shear)
        phase = cyclone_phase(fields, centre, motion_deg)
        div200 = fields.mean(fields.divergence(top), wide)
        rh_mid = torch.stack([fields.mean(rh[lv], ring) for lv in (700, 600, 500) if lv in rh], -1).mean(-1) \
            if any(lv in rh for lv in (700, 600, 500)) else torch.zeros_like(shear)
        rh_low = torch.stack([fields.mean(rh[lv], ring) for lv in (850, 700) if lv in rh], -1).mean(-1) \
            if any(lv in rh for lv in (850, 700)) else torch.zeros_like(shear)
        t200 = fields.mean(fields.field(f"temperature@{top}"), ring) if fields.has(f"temperature@{top}") else torch.zeros_like(shear)
        theta_e = equivalent_potential_temperature(t_prof[:, k_bl], mixing_ratio(q_prof[:, k_bl]), torch.full_like(sst, float(bl)))
        tcwv = fields.mean(fields.field("total_column_water_vapour"), core) if fields.has("total_column_water_vapour") else torch.zeros_like(shear)
        rain = (fields.mean(fields.field("total_precipitation_6hr"), fields.weights(centre, 0.0, 300.0)) * 1000.0
                if fields.has("total_precipitation_6hr") else torch.zeros_like(shear))
        land = (fields.mean(fields.field("land_sea_mask"), fields.weights(centre, 0.0, 300.0))
                if fields.has("land_sea_mask") else torch.zeros_like(shear))
        lat = centre[:, 0]
        speed_ms = torch.hypot(motion_deg[:, 0] * 111195.0, motion_deg[:, 1] * 111195.0 * torch.cos(torch.deg2rad(lat))) / (12 * 3600.0)
        deep_u, deep_v, _ = _steering_wind(reader, ring)
        shallow = [lv for lv in (850, 700, 500) if reader.has(f"u_component_of_wind@{lv}") and reader.has(f"v_component_of_wind@{lv}")]
        su = sum(reader.mean(f"u_component_of_wind@{lv}", ring) for lv in shallow) / max(len(shallow), 1)
        sv = sum(reader.mean(f"v_component_of_wind@{lv}", ring) for lv in shallow) / max(len(shallow), 1)
        gap = pi["vmax_ms"] - (wind_now if wind_now is not None else torch.full_like(shear, 25.0))
        columns = [pi["vmax_ms"] / 50.0, gap / 50.0, (torch.nan_to_num(pi["outflow_k"], nan=200.0) - 200.0) / 15.0,
                   vent * 10.0, chi, cape_env / 2000.0, torch.log1p(gpi), phase["phase_b"] / 20.0,
                   phase["phase_vtl"] / 100.0, phase["phase_vtu"] / 100.0, div200 * 1e5, vort850 * cyclonic * 1e5, rh_mid, rh_low,
                   (t200 - 273.15 + 55.0) / 5.0, (theta_e - 350.0) / 10.0, tcwv / 50.0, rain / 10.0, land,
                   speed_ms / 10.0, (deep_u - su) / 10.0, (deep_v - sv) / 10.0, torch.ones_like(shear)]
        out = torch.stack([torch.nan_to_num(c.to(torch.float64), nan=0.0, posinf=0.0, neginf=0.0) for c in columns], -1)
        rows.append(out.clamp(-20.0, 20.0).float())
    return torch.cat(rows)


@torch.no_grad()
def genesis_map(state: torch.Tensor, variables, normalizer, grid) -> torch.Tensor:
    """
    The genesis potential index at every grid point of one state ``(P, C)`` -> ``(P,)``. Potential intensity
    is computed column by column, so this costs a few seconds on a CPU at 1.5 degrees.
    """
    fields = GridFields(state.unsqueeze(0), variables, normalizer, grid)
    levels = [lv for lv in fields.levels("temperature") if fields.has(f"specific_humidity@{lv}") and lv >= 100]
    t = torch.stack([fields.field(f"temperature@{lv}")[0] for lv in levels], -1)
    q = torch.stack([fields.field(f"specific_humidity@{lv}")[0] for lv in levels], -1)
    sst = fields.field("sea_surface_temperature")[0] if fields.has("sea_surface_temperature") else fields.field("2m_temperature")[0]
    sst = torch.where(sst > 250.0, sst, fields.field("2m_temperature")[0] if fields.has("2m_temperature") else sst)
    msl = fields.field("mean_sea_level_pressure")[0] / 100.0
    pi = potential_intensity(sst, msl, torch.tensor(levels, dtype=torch.float64), t, q)["vmax_ms"]
    eta = fields.vorticity(850)[0] + fields.coriolis()
    rh600 = relative_humidity(fields.field("temperature@600")[0], fields.field("specific_humidity@600")[0],
                              torch.tensor(600.0, dtype=torch.float64)) * 100
    shear = torch.hypot(fields.field("u_component_of_wind@200")[0] - fields.field("u_component_of_wind@850")[0],
                        fields.field("v_component_of_wind@200")[0] - fields.field("v_component_of_wind@850")[0])
    gpi = genesis_potential_index(eta, rh600, pi, shear)
    land = fields.field("land_sea_mask")[0] > 0.5 if fields.has("land_sea_mask") else torch.zeros_like(gpi, dtype=torch.bool)
    return torch.where(land | (sst < 299.0), torch.zeros_like(gpi), gpi).float()


# ------------------------------------------------------------------------------- NatureV2: ocean --

#: NatureV2 (0.17): what the ocean under the storm can give it, and how big the storm is now -- what the 0.15
#: verification found the rapid-intensification guidance gains from beyond the atmosphere (the World Ocean Atlas 26 C
#: depth and heat potential with the gale radius lifted the held-out Brier skill of P(RI) from 0.122 to 0.154). This
#: year's sea-surface anomaly against the climatology says whether the warm layer is likely deeper or shallower than
#: its average. Scaled to order one; zeros where unknown, with the two flags saying what was.
STORM_OCEAN_FEATURES = ("d26", "tchp", "sst_anomaly", "gale_radius", "gale_observed", "observed")


def storm_ocean(storm_center: torch.Tensor, valid_time, storm_environment: torch.Tensor | None = None,
                current_radii_nmi: torch.Tensor | None = None) -> torch.Tensor:
    """
    The ocean-and-size vector for each storm, ``(B, len(STORM_OCEAN_FEATURES))``.

    Args:
        storm_center: ``(B, 2)`` degrees.
        valid_time: the analysis time of each storm, seconds since 1970: ``(B,)`` or one value for all.
        storm_environment: ``(B, 8)`` from :func:`naturev1.storm_environment` -- its ``ocean_temperature`` is this
            year's sea surface under the core, compared with the climatology's for the anomaly. Optional.
        current_radii_nmi: the 34 / 50 / 64 kt wind radii now, ``(B, 3, 4)`` (or the 34 kt quadrants ``(B, 4)``),
            NaN where unobserved. Optional.

    Returns:
        D26 / 100 m, heat potential / 100 kJ cm-2, sea-surface anomaly / 2 K, mean 34 kt radius / 100 nmi, whether it
        was observed, and whether the climatology covered the point. The climatology is the packaged World Ocean
        Atlas (:class:`naturev1.cyclone.ocean.OceanClimatology`), read at each storm's position and month.
    """
    import numpy as np

    from .cyclone.ocean import OceanClimatology

    device = storm_center.device
    centre = storm_center.detach().to("cpu", torch.float64).numpy()
    count = centre.shape[0]
    out = np.zeros((count, len(STORM_OCEAN_FEATURES)), np.float64)
    seconds = torch.as_tensor(valid_time).detach().to("cpu", torch.float64).reshape(-1).numpy()
    if seconds.size == 1 and count != 1:
        seconds = np.full(count, seconds[0])
    climatology = OceanClimatology.default()
    known = np.isfinite(centre).all(-1) & np.isfinite(seconds[:count]) if count else np.zeros(0, bool)
    if climatology is not None and known.any():
        sst_now = None
        if storm_environment is not None:
            env = storm_environment.detach().to("cpu", torch.float64).numpy()
            sst_now = np.where(env[:, 7] > 0.5, env[:, 6] * 3.0 + 26.5, np.nan)          # degrees C
        for stamp in np.unique(seconds[known]):
            rows = np.nonzero(known & (seconds == stamp))[0]
            sample = climatology.sample(centre[rows, 0], centre[rows, 1], np.datetime64(int(stamp), "s"))
            d26, tchp = sample["d26_m"], sample["tchp_kj_cm2"]
            seen = np.isfinite(d26) & np.isfinite(tchp)
            out[rows, 0] = np.where(seen, np.clip(d26, 0.0, 300.0) / 100.0, 0.0)
            out[rows, 1] = np.where(seen, np.clip(tchp, 0.0, 300.0) / 100.0, 0.0)
            out[rows, 5] = seen.astype(np.float64)
            if sst_now is not None and "sst_c" in sample:
                anomaly = sst_now[rows] - sample["sst_c"]
                usable = seen & np.isfinite(anomaly) & (np.abs(anomaly) < 6.0)            # beyond 6 K it is not the sea
                out[rows, 2] = np.where(usable, anomaly / 2.0, 0.0)
    if current_radii_nmi is not None:
        radii = current_radii_nmi.detach().to("cpu", torch.float64).numpy()
        gale = radii[:, 0, :] if radii.ndim == 3 else radii
        gale = np.where(np.isfinite(gale) & (gale > 0), gale, np.nan)
        observed = np.isfinite(gale).any(-1)
        mean = np.where(observed, np.nanmean(np.where(observed[:, None], gale, 0.0), -1), 0.0) if gale.size else np.zeros(count)
        out[:, 3] = np.where(observed, np.clip(mean, 0.0, 600.0) / 100.0, 0.0)
        out[:, 4] = observed.astype(np.float64)
    return torch.from_numpy(out).to(device, torch.float32)


__all__ = ["CK_CD", "STORM_PHYSICS_FEATURES", "GridFields", "cape", "cyclone_phase", "entropy_deficit",
           "equivalent_potential_temperature", "genesis_map", "genesis_potential_index", "moist_entropy",
           "parcel_temperature", "potential_intensity", "relative_humidity", "saturation_mixing_ratio",
           "storm_ocean", "storm_physics", "STORM_OCEAN_FEATURES", "ventilation_index"]
