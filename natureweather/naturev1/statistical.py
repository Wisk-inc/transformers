# Copyright 2026 Nathan. Apache-2.0.
"""
Statistical-dynamical guidance inside the model, and the corrected consensus that combines everything.

The National Hurricane Center's best intensity forecasts are not any single model: they are the corrected
consensus of several -- dynamical models, the statistical-dynamical SHIPS and LGEM, and climatology and
persistence -- each bias-corrected and weighted by how well it has done. Track works the same way, and the
spread of the members says how far to trust the result (the Goerss Predicted Consensus Error). This module
puts that whole apparatus next to the network, fitted on the same banks:

* **SHIPS-style regression**: the intensity change at every lead, ridge-regressed on the predictors of
  :mod:`naturev1.thermo` (potential intensity and the gap below it, ventilation, shear, humidity,
  divergence, vorticity, phase space, ...) plus the storm's own state and the satellite network's reading.
* **Logistic growth** (after DeMaria 2009, LGEM): ``dV/dt = kappa V - beta (V / Vmpi)^n V``, integrated
  hour by hour with the thermodynamic potential intensity as the ceiling; the growth rate ``kappa`` is a
  learned linear function of the predictors, ``beta`` learned, ``n = 2.5``.
* **Climatology and persistence**: the mean change for storms of the same strength and recent trend,
  measured on the training banks -- the baseline every intensity forecast has to beat (SHIFOR's role).
* **Corrected consensus**: per-lead weights on the simplex over the members each storm has, plus a bias
  term, fitted on held-out seasons -- for intensity, and a bias correction for the track consensus in the
  poleward frame.
* **Predicted consensus error**: how far the consensus is expected to be wrong, per lead, as a linear function
  of how much the members disagree, scaled to contain a chosen share of held-out errors -- a cone whose width
  follows the situation, not climatology.

All fitted parameters are buffers of :class:`StatisticalModels`, saved with the weights; every one starts
unfitted, and an unfitted model contributes nothing.
"""

from __future__ import annotations

import math

import torch
from torch import nn

KT = 0.514444
#: The intensity consensus members, in buffer order.
INTENSITY_MEMBERS = ("heads", "eyewall", "regression", "logistic", "climatology", "satellite", "rollout")
#: What the RI consensus stacks: every member's 24 h / +30 kt logit.
RI_CONSENSUS_INPUTS = ("ri2", "ri_head", "satellite", "regression")
#: Intensity bins (kt) and 12-hour trend bins (kt) of the climatology-persistence table.
WIND_BINS = (35.0, 50.0, 65.0, 85.0, 110.0)
TREND_BINS = (-5.0, 5.0)
LOGISTIC_EXPONENT = 2.5


def _bucket(values: torch.Tensor, edges) -> torch.Tensor:
    return torch.bucketize(values, torch.tensor(edges, dtype=values.dtype, device=values.device))


def intensity_predictors(storm_state=None, storm_environment=None, storm_physics=None, storm_center=None,
                         satnet: dict | None = None, batch: int | None = None, device=None) -> torch.Tensor:
    """
    The predictor vector of the statistical models, ``(B, P)``: current wind and its square, 12-hour trend,
    |latitude|, the eight environment features, the physics features and the satellite network's change
    estimates. Missing groups are zeros (each group carries its own presence flag).
    """
    from .model import PEAK_WIND_ANCHOR_KT, STORM_ENVIRONMENT_FEATURES
    from .thermo import STORM_PHYSICS_FEATURES

    reference = next(x for x in (storm_state, storm_environment, storm_physics, storm_center) if x is not None)
    batch = batch or reference.shape[0]
    device = device or reference.device
    columns = []
    if storm_state is not None:
        st = torch.nan_to_num(storm_state.to(device).float())
        wind = (st[:, 0] * PEAK_WIND_ANCHOR_KT[1] + PEAK_WIND_ANCHOR_KT[0]) / 100.0
        columns += [wind, wind ** 2, st[:, 4], st[:, 5]]
    else:
        columns += [torch.zeros(batch, device=device)] * 4
    lat = torch.zeros(batch, device=device) if storm_center is None else torch.nan_to_num(storm_center.to(device).float()[:, 0]).abs() / 30.0
    columns.append(lat)
    env = (torch.zeros(batch, len(STORM_ENVIRONMENT_FEATURES), device=device) if storm_environment is None
           else torch.nan_to_num(storm_environment.to(device).float()))
    phys = (torch.zeros(batch, len(STORM_PHYSICS_FEATURES), device=device) if storm_physics is None
            else torch.nan_to_num(storm_physics.to(device).float()))
    sat = torch.zeros(batch, 6, device=device)
    if satnet is not None and "wind_change_kt" in satnet:
        present = satnet.get("present")
        flag = torch.ones(batch, device=device) if present is None else present.to(device).float().view(-1)
        sat = torch.cat([flag.view(-1, 1), satnet["wind_change_kt"].to(device).float() / 15.0 * flag.view(-1, 1)], -1)
    return torch.cat([torch.stack(columns, -1), env, phys, sat], -1)


def predictor_count() -> int:
    from .model import STORM_ENVIRONMENT_FEATURES
    from .thermo import STORM_PHYSICS_FEATURES

    return 5 + len(STORM_ENVIRONMENT_FEATURES) + len(STORM_PHYSICS_FEATURES) + 6


class StatisticalModels(nn.Module):
    """Every fitted statistical model and consensus parameter, as buffers. All zero (unfitted) at first."""

    def __init__(self, lead_times_hours) -> None:
        super().__init__()
        leads = len(lead_times_hours)
        self.lead_times_hours = tuple(int(h) for h in lead_times_hours)
        count = predictor_count()
        self.register_buffer("regression", torch.zeros(leads, count + 1))               # dV per lead, kt
        self.register_buffer("regression_scale", torch.zeros(count))                    # predictor standardisation
        self.register_buffer("regression_shift", torch.zeros(count))
        self.register_buffer("logistic", torch.zeros(count + 2))                        # kappa weights + bias, beta
        self.register_buffer("climatology", torch.zeros(len(WIND_BINS) + 1, len(TREND_BINS) + 1, leads))
        self.register_buffer("intensity_weights", torch.zeros(leads, len(INTENSITY_MEMBERS)))
        self.register_buffer("intensity_bias", torch.zeros(leads))
        self.register_buffer("track_bias", torch.zeros(leads, 2))                       # poleward frame, degrees
        self.register_buffer("consensus_error", torch.zeros(leads, 3))                  # a, b (km), scale
        # The rollout's resolved intensity mapped to best-track intensity: wind = a + b raw wind + c deficit.
        self.register_buffer("rollout_map", torch.zeros(leads, 3))
        self.register_buffer("regression_sd", torch.zeros(leads))                       # residual spread, kt
        # Stacked RI consensus (24 h, +30 kt): weights on [bias, RI 2.0, RI head, satellite net, regression]
        # logits and their presence flags.
        self.register_buffer("ri_consensus", torch.zeros(2 * len(RI_CONSENSUS_INPUTS) + 1))
        # regression, logistic, climatology, intensity consensus, track bias, consensus error, rollout map,
        # RI consensus
        self.register_buffer("fitted", torch.zeros(8, dtype=torch.int64))

    # ---------------------------------------------------------------------------------- members --

    def standardise(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.regression_scale.to(x.device)
        return (x - self.regression_shift.to(x.device)) / torch.where(scale > 0, scale, torch.ones_like(scale))

    def regression_member(self, x: torch.Tensor, wind_now_kt: torch.Tensor) -> torch.Tensor:
        """``(B, L)`` kt, NaN when unfitted."""
        if int(self.fitted[0]) == 0:
            return torch.full((x.shape[0], len(self.lead_times_hours)), float("nan"), device=x.device)
        z = self.standardise(x)
        design = torch.cat([torch.ones_like(z[:, :1]), z], -1)
        change = design @ self.regression.to(x.device).T
        return (wind_now_kt.view(-1, 1) + change).clamp(10.0, 200.0)

    def logistic_member(self, x: torch.Tensor, wind_now_kt: torch.Tensor, potential_kt: torch.Tensor,
                        parameters: torch.Tensor | None = None) -> torch.Tensor:
        """Integrate the logistic growth equation hour by hour to every lead. ``(B, L)`` kt."""
        params = self.logistic.to(x.device) if parameters is None else parameters
        if parameters is None and int(self.fitted[1]) == 0:
            return torch.full((x.shape[0], len(self.lead_times_hours)), float("nan"), device=x.device)
        z = self.standardise(x)
        kappa = (z @ params[:-2] + params[-2]) / 24.0                                    # per hour
        beta = torch.nn.functional.softplus(params[-1]) / 24.0
        ceiling = potential_kt.clamp_min(20.0)
        v = wind_now_kt.clamp_min(10.0)
        out, hour = [], 0
        for lead in self.lead_times_hours:
            while hour < lead:
                v = v + (kappa * v - beta * (v / ceiling).clamp_min(0.0) ** LOGISTIC_EXPONENT * v)
                v = v.clamp(10.0, 220.0)
                hour += 1
            out.append(v)
        return torch.stack(out, -1)

    def climatology_member(self, wind_now_kt: torch.Tensor, trend_kt: torch.Tensor) -> torch.Tensor:
        if int(self.fitted[2]) == 0:
            return torch.full((wind_now_kt.shape[0], len(self.lead_times_hours)), float("nan"), device=wind_now_kt.device)
        table = self.climatology.to(wind_now_kt.device)
        change = table[_bucket(wind_now_kt, WIND_BINS), _bucket(trend_kt, TREND_BINS)]
        return (wind_now_kt.view(-1, 1) + change).clamp(10.0, 200.0)

    def rollout_member(self, raw_wind_kt: torch.Tensor | None, raw_pressure_hpa: torch.Tensor | None,
                       batch: int, device) -> torch.Tensor:
        """A 1.5-degree rollout resolves a weak, broad version of the storm; the fitted map restores the scale."""
        nan = torch.full((batch, len(self.lead_times_hours)), float("nan"), device=device)
        if raw_wind_kt is None or int(self.fitted[6]) == 0:
            return nan
        table = self.rollout_map.to(device)
        deficit = (1010.0 - raw_pressure_hpa.to(device).float()) if raw_pressure_hpa is not None else torch.zeros_like(raw_wind_kt)
        return (table[:, 0] + table[:, 1] * raw_wind_kt.to(device).float() + table[:, 2] * deficit).clamp(10.0, 200.0)

    def consensus(self, members: torch.Tensor) -> torch.Tensor:
        """Bias-corrected weighted blend of ``(B, L, M)`` intensity members over the ones each storm has."""
        from .storms import consensus_track

        weights = self.intensity_weights.to(members.device)
        if int(self.fitted[3]) == 0:
            weights = torch.zeros_like(weights)
            weights[:, 0] = 1.0
        blended = consensus_track(members.unsqueeze(-1).expand(*members.shape, 2), weights)[..., 0]
        return blended + self.intensity_bias.to(members.device).view(1, -1) * float(int(self.fitted[3]) > 0)


def intensity_members(model, outputs: dict, predictors: torch.Tensor, storm_state, storm_environment,
                      storm_physics, satnet: dict | None = None, rollout_kt: torch.Tensor | None = None,
                      rollout_pressure: torch.Tensor | None = None) -> torch.Tensor:
    """``(B, L, len(INTENSITY_MEMBERS))`` wind forecasts in kt; NaN where a member is unavailable or unfitted."""
    from .model import PEAK_WIND_ANCHOR_KT, potential_intensity_kt

    stats: StatisticalModels = model.statistics
    batch, leads = outputs["intensity_mean"].shape[:2]
    device = outputs["intensity_mean"].device
    st = torch.nan_to_num(storm_state.to(device).float()) if storm_state is not None else torch.zeros(batch, 6, device=device)
    wind_now = torch.where(st[:, 5] > 0.5, st[:, 0] * PEAK_WIND_ANCHOR_KT[1] + PEAK_WIND_ANCHOR_KT[0],
                           torch.full_like(st[:, 0], 40.0))
    trend = st[:, 4] * 15.0
    if storm_physics is not None and float(storm_physics[:, -1].abs().sum()) > 0:
        potential = storm_physics.to(device).float()[:, 0] * 50.0 / KT
        if storm_environment is not None:
            potential = torch.where(potential > 1.0, potential, potential_intensity_kt(storm_environment.to(device)[:, 6]))
    elif storm_environment is not None:
        potential = potential_intensity_kt(storm_environment.to(device)[:, 6])
    else:
        potential = torch.full_like(wind_now, 140.0)
    nan = torch.full((batch, leads), float("nan"), device=device)
    satellite = nan
    if satnet is not None and "wind_kt" in satnet:
        from .ibtracs import CHANGE_HOURS

        hours = torch.tensor(CHANGE_HOURS, dtype=torch.float32, device=device)
        target = torch.tensor(model.config.lead_times_hours, dtype=torch.float32, device=device)
        change = satnet["wind_change_kt"].to(device).float()
        index = torch.searchsorted(hours, target).clamp(1, len(CHANGE_HOURS) - 1)
        w = ((target - hours[index - 1]) / (hours[index] - hours[index - 1])).clamp(0.0, 1.0)
        early = (target / hours[0]).clamp(max=1.0)
        interp = change[:, index - 1] * (1 - w) + change[:, index] * w
        interp = torch.where(target < hours[0], change[:, :1] * early, interp)
        satellite = satnet["wind_kt"].to(device).float().view(-1, 1) + interp
        present = satnet.get("present")
        if present is not None:
            satellite = torch.where(present.to(device).view(-1, 1).bool(), satellite, nan)
    rollout = stats.rollout_member(rollout_kt, rollout_pressure, batch, device)
    members = [outputs["intensity_mean"][..., 0].float() / KT, outputs["eyewall_peak_wind_kt"].float(),
               stats.regression_member(predictors, wind_now), stats.logistic_member(predictors, wind_now, potential),
               stats.climatology_member(wind_now, trend), satellite, rollout]
    return torch.stack(members, -1)


# --------------------------------------------------------------------------------------- fitting --

def _bank_inputs(model, bank: dict, device="cpu"):
    t = bank["targets"]
    satnet = None
    if bank.get("satnet") is not None:
        satnet = {k: v for k, v in bank["satnet"].items() if torch.is_tensor(v)}
    physics = t.get("storm_physics")
    x = intensity_predictors(t["storm_state"], t["storm_environment"], physics, t["storm_center"], satnet)
    return t, satnet, physics, x


@torch.no_grad()
def fit_regression(model, bank: dict, ridge: float = 10.0) -> str:
    """Per-lead ridge regression of the intensity change on the standardised predictors (training bank)."""
    from .storms import KT_TO_MS

    stats = model.statistics
    t, _, _, x = _bank_inputs(model, bank)
    shift, scale = x.mean(0), x.std(0)
    stats.regression_shift.copy_(shift)
    stats.regression_scale.copy_(torch.where(scale > 1e-6, scale, torch.zeros_like(scale)))
    z = stats.standardise(x).double()
    design = torch.cat([torch.ones_like(z[:, :1]), z], -1)
    now = t["current_wind_kt"].double()
    lines = []
    for lead in range(len(stats.lead_times_hours)):
        ok = (t["intensity_mask"][:, lead, 0] > 0) & torch.isfinite(now)
        if int(ok.sum()) < design.shape[1] * 3:
            continue
        y = t["intensity_target"][ok, lead, 0].double() / KT_TO_MS - now[ok]
        a = design[ok]
        penalty = torch.eye(a.shape[1], dtype=torch.float64) * ridge
        penalty[0, 0] = 0.0
        coef = torch.linalg.solve(a.T @ a + penalty, a.T @ y)
        stats.regression[lead] = coef.float()
        residual = (a @ coef - y)
        stats.regression_sd[lead] = float(residual.std())
        lines.append(f"+{stats.lead_times_hours[lead]}h MAE {float(residual.abs().mean()):.1f} kt")
    stats.fitted[0] = 1
    return "regression fitted: " + ", ".join(lines)


def fit_logistic(model, bank: dict, steps: int = 300, max_lead: int = 72, lr: float = 0.05) -> str:
    """Fit the logistic-growth parameters by minimising the wind error (kt) at every lead up to ``max_lead``."""
    from .model import potential_intensity_kt
    from .storms import KT_TO_MS

    stats = model.statistics
    if int(stats.fitted[0]) == 0:
        fit_regression(model, bank)                                  # for the predictor standardisation
    t, _, physics, x = _bank_inputs(model, bank)
    now = t["current_wind_kt"].float()
    keep = torch.isfinite(now) & (now > 0)
    potential = potential_intensity_kt(t["storm_environment"][:, 6])
    if physics is not None and float(physics[:, -1].abs().sum()) > 0:
        thermo = physics[:, 0] * 50.0 / KT
        potential = torch.where(thermo > 1.0, thermo, potential)
    leads = [i for i, h in enumerate(stats.lead_times_hours) if h <= max_lead]
    truth = t["intensity_target"][:, leads, 0] / KT_TO_MS
    mask = (t["intensity_mask"][:, leads, 0] > 0) & keep.view(-1, 1)
    landed = t.get("landfall_target")
    if landed is not None:
        mask = mask & (landed[:, leads] < 0.5)                       # over water: inland decay is separate
    params = torch.zeros(predictor_count() + 2, requires_grad=True)
    with torch.no_grad():
        params[-1] = math.log(math.expm1(1.0))                      # beta = 1 / day
    optimizer = torch.optim.Adam([params], lr=lr)
    saved_leads = stats.lead_times_hours
    stats.lead_times_hours = tuple(stats.lead_times_hours[i] for i in leads)
    try:
        with torch.enable_grad():
            for _ in range(steps):
                optimizer.zero_grad()
                predicted = stats.logistic_member(x[keep], now[keep], potential[keep], params)
                error = (predicted - torch.nan_to_num(truth[keep])) ** 2 * mask[keep]
                loss = error.sum() / mask[keep].sum().clamp_min(1)
                loss.backward()
                optimizer.step()
    finally:
        stats.lead_times_hours = saved_leads
    stats.logistic.copy_(params.detach())
    stats.fitted[1] = 1
    return f"logistic growth fitted: RMSE {float(loss.detach().sqrt()):.1f} kt over water to +{max_lead} h"


@torch.no_grad()
def fit_climatology(model, bank: dict, minimum: int = 20) -> str:
    """Mean intensity change by current-wind bin and 12-hour trend bin, per lead (training bank)."""
    from .storms import KT_TO_MS

    stats = model.statistics
    t = bank["targets"]
    now = t["current_wind_kt"].float()
    trend = torch.nan_to_num(t["storm_state"][:, 4].float()) * 15.0
    wind_bin, trend_bin = _bucket(now, WIND_BINS), _bucket(trend, TREND_BINS)
    table = torch.zeros_like(stats.climatology)
    for lead in range(len(stats.lead_times_hours)):
        ok = (t["intensity_mask"][:, lead, 0] > 0) & torch.isfinite(now)
        change = t["intensity_target"][:, lead, 0] / KT_TO_MS - now
        overall = float(change[ok].mean()) if ok.any() else 0.0
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                cell = ok & (wind_bin == i) & (trend_bin == j)
                table[i, j, lead] = float(change[cell].mean()) if int(cell.sum()) >= minimum else overall
    stats.climatology.copy_(table)
    stats.fitted[2] = 1
    return "climatology-persistence table fitted"


@torch.no_grad()
def fit_intensity_consensus(model, bank: dict, members: torch.Tensor | None = None) -> str:
    """
    Corrected intensity consensus on held-out storms: simplex weights over the available members per lead
    (the same projected least squares as the track consensus), then the mean remaining error as a bias.
    """
    from .storms import KT_TO_MS, _project_to_simplex

    stats = model.statistics
    t = bank["targets"]
    if members is None:
        members = bank_intensity_members(model, bank)
    truth = t["intensity_target"][..., 0] / KT_TO_MS
    valid = t["intensity_mask"][..., 0] > 0
    finite = torch.isfinite(members)
    weights = torch.zeros(members.shape[1], members.shape[2])
    bias = torch.zeros(members.shape[1])
    lines = []
    for lead in range(members.shape[1]):
        ok = valid[:, lead]
        present = [m for m in range(members.shape[2]) if ok.any() and float(finite[ok, lead, m].float().mean()) >= 0.9]
        if not present:
            continue
        rows = ok & finite[:, lead, present].all(-1)
        a = members[rows, lead][:, present].double()
        y = truth[rows, lead].double()
        if a.shape[0] < 3 * len(present):
            weights[lead, present] = 1.0 / len(present)
            continue
        # Weights and bias fitted jointly: centring both sides separates "which members track the storm's
        # ups and downs" (the weights) from "by how much they are all off" (the bias). Without it a member
        # that is right in every change but 10 kt high loses weight to one that is merely unbiased noise.
        a_c, y_c = a - a.mean(0, keepdim=True), y - y.mean()
        gram, moment = a_c.T @ a_c / a.shape[0], a_c.T @ y_c / a.shape[0]
        step = 1.0 / float(torch.linalg.eigvalsh(gram).max().clamp_min(1e-9))
        w = torch.full((len(present),), 1.0 / len(present), dtype=torch.float64)
        for _ in range(1000):
            w = _project_to_simplex(w - step * (gram @ w - moment))
        weights[lead, present] = w.float()
        bias[lead] = float((y - a @ w).mean())
        names = ", ".join(f"{INTENSITY_MEMBERS[m]} {100 * float(v):.0f}%" for m, v in zip(present, w) if v > 0.005)
        lines.append(f"+{stats.lead_times_hours[lead]}h: {names}, bias {float(bias[lead]):+.1f} kt")
    stats.intensity_weights.copy_(weights)
    stats.intensity_bias.copy_(bias)
    stats.fitted[3] = 1
    return "intensity consensus: " + "; ".join(lines)


@torch.no_grad()
def bank_intensity_members(model, bank: dict) -> torch.Tensor:
    from .storms import _head_outputs

    out = _head_outputs(model, bank)
    t, satnet, physics, x = _bank_inputs(model, bank)
    return intensity_members(model, out, x, t["storm_state"], t["storm_environment"], physics, satnet,
                             t.get("rollout_wind_kt"), t.get("rollout_pressure_hpa"))


@torch.no_grad()
def fit_rollout_map(model, bank: dict, ridge: float = 1.0) -> str:
    """Per lead, best-track wind regressed on the rollout's resolved wind and pressure deficit (training bank)."""
    from .storms import KT_TO_MS

    stats = model.statistics
    t = bank["targets"]
    if t.get("rollout_wind_kt") is None:
        return "rollout map: the bank has no rollout intensity"
    lines = []
    for lead in range(len(stats.lead_times_hours)):
        raw, pressure = t["rollout_wind_kt"][:, lead].double(), t["rollout_pressure_hpa"][:, lead].double()
        ok = (t["intensity_mask"][:, lead, 0] > 0) & torch.isfinite(raw) & torch.isfinite(pressure)
        if int(ok.sum()) < 30:
            continue
        design = torch.stack([torch.ones_like(raw[ok]), raw[ok], 1010.0 - pressure[ok]], -1)
        y = t["intensity_target"][ok, lead, 0].double() / KT_TO_MS
        penalty = torch.diag(torch.tensor([0.0, ridge, ridge], dtype=torch.float64))
        coef = torch.linalg.solve(design.T @ design + penalty, design.T @ y)
        stats.rollout_map[lead] = coef.float()
        lines.append(f"+{stats.lead_times_hours[lead]}h MAE {float((design @ coef - y).abs().mean()):.1f} kt")
    stats.fitted[6] = 1
    return "rollout intensity map: " + ", ".join(lines)


# --------------------------------------------------------------------- track bias and the cone --

def _poleward(displacement: torch.Tensor, latitude: torch.Tensor) -> torch.Tensor:
    sign = torch.where(latitude < 0, -1.0, 1.0).to(displacement.dtype)
    out = displacement.clone()
    out[..., 0] = out[..., 0] * sign.view(-1, *([1] * (displacement.dim() - 2)))
    return out


@torch.no_grad()
def fit_track_bias(model, members: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor, latitude: torch.Tensor,
                   shrink: float = 0.5) -> str:
    """
    The mean error of the track consensus per lead, in the poleward frame (degrees), shrunk toward zero by
    ``shrink`` -- a systematic slow or equatorward bias is real; the noise around it is not worth fitting.
    """
    from .storms import consensus_track

    stats = model.statistics
    blended = consensus_track(members, model.track_consensus.cpu().float())
    error = _poleward(truth - blended, latitude)
    bias = torch.zeros_like(stats.track_bias)
    for lead in range(error.shape[1]):
        ok = valid[:, lead] & torch.isfinite(error[:, lead]).all(-1)
        if int(ok.sum()) >= 30:
            bias[lead] = error[ok, lead].mean(0) * shrink
    stats.track_bias.copy_(bias)
    stats.fitted[4] = 1
    return "track bias (poleward, east; degrees): " + ", ".join(
        f"+{h}h ({float(bias[i, 0]):+.2f}, {float(bias[i, 1]):+.2f})" for i, h in enumerate(stats.lead_times_hours))


def corrected_track(model, members: torch.Tensor, latitude: torch.Tensor) -> torch.Tensor:
    """The track consensus with the fitted bias added back in each storm's own hemisphere."""
    from .storms import consensus_track

    blended = consensus_track(members, model.track_consensus.to(members.device).float())
    if int(model.statistics.fitted[4]) == 0:
        return blended
    bias = model.statistics.track_bias.to(members.device).float().unsqueeze(0).expand_as(blended)
    return blended + _poleward(bias, latitude.to(members.device))


def member_spread_km(members: torch.Tensor, blended: torch.Tensor, latitude: torch.Tensor) -> torch.Tensor:
    """``(N, L)`` mean distance of the available members from the consensus, km."""
    from .steering import displacement_km

    finite = torch.isfinite(members).all(-1)
    distance = displacement_km(torch.nan_to_num(members - blended.unsqueeze(2)).double(),
                               latitude.double().view(-1, 1, 1)).norm(dim=-1)
    return ((distance * finite).sum(-1) / finite.sum(-1).clamp_min(1)).float()


@torch.no_grad()
def fit_consensus_error(model, members: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor,
                        latitude: torch.Tensor, coverage: float = 0.67) -> str:
    """
    Goerss-style predicted consensus error: per lead, track error ~ a + b * member spread (least squares),
    then a scale so ``coverage`` of held-out errors fall inside ``scale * (a + b * spread)``.
    """
    from .steering import displacement_km

    stats = model.statistics
    blended = corrected_track(model, members, latitude)
    spread = member_spread_km(members, blended, latitude)
    error = displacement_km((truth - blended).double(), latitude.double().view(-1, 1)).norm(dim=-1).float()
    table = torch.zeros_like(stats.consensus_error)
    lines = []
    for lead in range(error.shape[1]):
        ok = valid[:, lead] & torch.isfinite(error[:, lead]) & torch.isfinite(spread[:, lead])
        if int(ok.sum()) < 30:
            continue
        x, y = spread[ok, lead].double(), error[ok, lead].double()
        design = torch.stack([torch.ones_like(x), x], -1)
        coef = torch.linalg.lstsq(design, y.view(-1, 1)).solution.view(-1)
        coef[1] = coef[1].clamp_min(0.0)
        predicted = (design @ coef).clamp_min(10.0)
        scale = float(torch.quantile(y / predicted, coverage))
        table[lead] = torch.tensor([float(coef[0]), float(coef[1]), scale])
        lines.append(f"+{stats.lead_times_hours[lead]}h {float(coef[0]):.0f} + {float(coef[1]):.2f} x spread")
    stats.consensus_error.copy_(table)
    stats.fitted[5] = 1
    return f"predicted consensus error ({100 * coverage:.0f}% radius): " + "; ".join(lines)


def consensus_radius_km(model, spread_km: torch.Tensor) -> torch.Tensor:
    """``(N, L)`` the situation-dependent cone radius; NaN when unfitted."""
    table = model.statistics.consensus_error.to(spread_km.device)
    if int(model.statistics.fitted[5]) == 0:
        return torch.full_like(spread_km, float("nan"))
    return ((table[:, 0] + table[:, 1] * spread_km).clamp_min(10.0) * table[:, 2]).float()


# ------------------------------------------------------------------------------ the RI consensus --

def ri_member_logits(model, outputs: dict, predictors: torch.Tensor, wind_now_kt: torch.Tensor,
                     satnet: dict | None = None) -> torch.Tensor:
    """``(B, 2 * len(RI_CONSENSUS_INPUTS))``: each member's 24 h / +30 kt logit, then its presence flag."""
    from .satnet import ri_threshold_list

    batch = predictors.shape[0]
    device = predictors.device
    logits = torch.zeros(batch, len(RI_CONSENSUS_INPUTS), device=device)
    present = torch.zeros_like(logits)
    column = ri_threshold_list().index((24, 30.0))
    if "ri2_logits" in outputs:
        logits[:, 0], present[:, 0] = outputs["ri2_logits"][:, column].float(), 1.0
    if "ri_logits" in outputs:
        logits[:, 1], present[:, 1] = outputs["ri_logits"][:, 1].float(), 1.0
    if satnet is not None and "ri_logits" in satnet:
        flag = satnet.get("present")
        flag = torch.ones(batch, device=device) if flag is None else flag.to(device).float()
        logits[:, 2], present[:, 2] = satnet["ri_logits"].to(device).float()[:, column] * flag, flag
    stats = model.statistics
    leads = list(stats.lead_times_hours)
    if int(stats.fitted[0]) and 24 in leads and float(stats.regression_sd[leads.index(24)]) > 0:
        lead = leads.index(24)
        predicted = stats.regression_member(predictors, wind_now_kt)[:, lead] - wind_now_kt
        z = (predicted - 30.0) / float(stats.regression_sd[lead])
        probability = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        logits[:, 3] = torch.logit(probability.clamp(1e-4, 1 - 1e-4))
        present[:, 3] = 1.0
    return torch.cat([logits.clamp(-12, 12), present], -1)


def ri_consensus_probability(model, inputs: torch.Tensor) -> torch.Tensor | None:
    """The stacked probability of +30 kt in 24 h, or None while unfitted."""
    stats = model.statistics
    if int(stats.fitted[7]) == 0:
        return None
    weights = stats.ri_consensus.to(inputs.device)
    return torch.sigmoid(weights[0] + inputs @ weights[1:])


def fit_ri_consensus(model, bank: dict, l2: float = 0.01) -> str:
    """Logistic stacking of the RI members on held-out storms (the published RI aids are combined the same way)."""
    from .storms import _head_outputs

    t, satnet, _, x = _bank_inputs(model, bank)
    out = _head_outputs(model, bank)
    inputs = ri_member_logits(model, out, x, t["current_wind_kt"].float(), satnet)
    target, mask = t["ri_target"][:, 1].float(), t["ri_mask"][:, 1] > 0
    if int(target[mask].sum()) < 5:
        return "RI consensus: fewer than five rapid intensifications to fit on"
    weights = torch.zeros(inputs.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights], lr=0.5, max_iter=200)
    data, y = inputs[mask].double(), target[mask].double()

    def closure():
        optimizer.zero_grad()
        logit = weights[0] + data @ weights[1:]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y) + l2 * (weights[1:] ** 2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)
    model.statistics.ri_consensus.copy_(weights.detach().float())
    model.statistics.fitted[7] = 1
    p = torch.sigmoid(weights.detach()[0] + data @ weights.detach()[1:])
    base = float(y.mean())
    brier = float(((p - y) ** 2).mean())
    return (f"RI consensus (+30 kt/24 h): Brier {brier:.4f}, skill {1 - brier / max(base * (1 - base), 1e-9):+.3f} "
            f"over the base rate {100 * base:.1f}% (on the storms it was fitted on)")


__all__ = ["INTENSITY_MEMBERS", "RI_CONSENSUS_INPUTS", "fit_ri_consensus", "ri_consensus_probability", "ri_member_logits", "StatisticalModels", "bank_intensity_members", "consensus_radius_km", "corrected_track",
           "fit_climatology", "fit_consensus_error", "fit_intensity_consensus", "fit_logistic", "fit_regression",
           "fit_rollout_map", "fit_track_bias", "intensity_members", "intensity_predictors", "member_spread_km", "predictor_count"]
