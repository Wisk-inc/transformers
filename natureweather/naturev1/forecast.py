# Copyright 2026 Nathan. Apache-2.0.
"""
Turn raw model outputs into a forecast a person can read and act on.

The model emits distributions. This turns them into the things people actually ask: where will it go,
how sure are you, will it hit land, when, will it rain here. Every number that comes out of here carries
its uncertainty, because a track without a cone is a guess wearing a suit.
"""

from __future__ import annotations

import datetime as dt
import math

import torch

from .model import RI_THRESHOLDS_KT, SURFACE_FIELDS, WEATHER_TYPES, WIND_RADII_THRESHOLDS_KT


def _ellipse(sigma_x: float, sigma_y: float, rho: float) -> tuple[float, float, float]:
    """
    Convert a 2-D Gaussian into the ellipse people draw: semi-major, semi-minor, bearing.

    Returns the 1-sigma axes in degrees and the bearing of the major axis in compass degrees, which is
    what a cone-of-uncertainty plot needs.
    """
    covariance_xy = rho * sigma_x * sigma_y
    trace = sigma_x**2 + sigma_y**2
    gap = math.sqrt(max((sigma_x**2 - sigma_y**2) ** 2 / 4 + covariance_xy**2, 0.0))
    major = math.sqrt(max(trace / 2 + gap, 0.0))
    minor = math.sqrt(max(trace / 2 - gap, 0.0))
    angle = 0.5 * math.atan2(2 * covariance_xy, sigma_x**2 - sigma_y**2)
    bearing = (90.0 - math.degrees(angle)) % 360.0
    return major, minor, bearing


def decode_track(
    outputs: dict,
    origin_lat: float,
    origin_lon: float,
    lead_times_hours: tuple[int, ...],
    issued: dt.datetime,
    batch_index: int = 0,
    max_scenarios: int | None = None,
) -> list[dict]:
    """
    Turn the mixture head into a ranked list of scenarios, each a full trajectory with a cone.

    Scenarios come back sorted by probability. Low-probability branches are kept rather than averaged
    away -- the whole reason for a mixture is that "62% offshore, 26% coast-hugging, 12% inland" is
    actionable and their average is not.
    """
    probabilities = outputs["mode_logits"][batch_index].softmax(-1)
    displacement = outputs["displacement"][batch_index]
    log_scale = outputs["log_scale"][batch_index]
    correlation = outputs["correlation"][batch_index]

    order = probabilities.argsort(descending=True)
    if max_scenarios:
        order = order[:max_scenarios]

    scenarios = []
    for rank, mode in enumerate(order.tolist()):
        points = []
        for lead_index, hours in enumerate(lead_times_hours):
            d_lat = float(displacement[mode, lead_index, 0])
            d_lon = float(displacement[mode, lead_index, 1])
            sigma_lat = float(log_scale[mode, lead_index, 0].exp())
            sigma_lon = float(log_scale[mode, lead_index, 1].exp())
            latitude = origin_lat + d_lat
            # The cone is drawn in kilometres. A degree of longitude is 111 km only at the equator --
            # at 30N it is 96 km -- so the east-west spread is scaled before the ellipse is formed,
            # rather than treating degrees of longitude and latitude as the same length.
            shrink = max(math.cos(math.radians(latitude)), 1e-3)
            major, minor, bearing = _ellipse(sigma_lon * shrink, sigma_lat, float(correlation[mode, lead_index]))
            # Degrees of longitude shrink with latitude; the displacement is in degrees, so no conversion
            # is needed here, but the cone's east-west extent in kilometres does depend on it.
            longitude = (origin_lon + d_lon + 180.0) % 360.0 - 180.0
            points.append(
                {
                    "lead_hours": hours,
                    "valid_time": (issued + dt.timedelta(hours=hours)).isoformat(),
                    "latitude": round(latitude, 3),
                    "longitude": round(longitude, 3),
                    "sigma_lat_deg": round(sigma_lat, 3),
                    "sigma_lon_deg": round(sigma_lon, 3),
                    "cone_semi_major_deg": round(major, 3),
                    "cone_semi_minor_deg": round(minor, 3),
                    "cone_bearing_deg": round(bearing, 1),
                    # One standard deviation of a 2-D Gaussian holds 39% of it, not the 68% it holds in
                    # one dimension; 2.448 standard deviations hold 95%.
                    "cone_radius_km_39": round(major * 111.0, 1),
                    "cone_radius_km_95": round(major * 111.0 * 2.448, 1),
                }
            )
        scenarios.append(
            {"rank": rank + 1, "probability": round(float(probabilities[mode]), 4), "track": points}
        )
    return scenarios


def decode_track_cone(
    outputs: dict,
    origin_lat: float,
    origin_lon: float,
    lead_times_hours: tuple[int, ...],
    issued: dt.datetime,
    batch_index: int = 0,
    samples: int = 4000,
) -> list[dict]:
    """
    The cone of uncertainty from the *whole* scenario mixture, not from one scenario.

    One scenario's ellipse is over-confident by construction -- the other scenarios carry the rest of
    the probability, and on held-out storms a single "95%" ellipse held the truth only 60-87% of the
    time. This draws ``samples`` positions from the full mixture and reports, per lead, the radius
    around the mixture's centre that holds 50% and 90% of them. On held-out storms the mixture's 90%
    region held the truth 87-92% of the time.
    """
    generator = torch.Generator().manual_seed(0)
    weights = outputs["mode_logits"][batch_index].softmax(-1).float().cpu()
    mean = outputs["displacement"][batch_index].float().cpu()              # (M, L, 2)
    scale = outputs["log_scale"][batch_index].float().cpu().exp()
    rho = outputs["correlation"][batch_index].float().cpu().clamp(-0.99, 0.99)
    pick = torch.multinomial(weights, samples, replacement=True, generator=generator)
    first, second = torch.randn(samples, mean.shape[1], generator=generator), torch.randn(
        samples, mean.shape[1], generator=generator)
    r = rho[pick]
    d_lat = mean[pick, :, 0] + scale[pick, :, 0] * first
    d_lon = mean[pick, :, 1] + scale[pick, :, 1] * (r * first + (1 - r**2).sqrt() * second)
    centre = (weights.view(-1, 1, 1) * mean).sum(0)                         # (L, 2)
    rows = []
    for lead, hours in enumerate(lead_times_hours):
        latitude = origin_lat + float(centre[lead, 0])
        shrink = max(math.cos(math.radians(latitude)), 1e-3)
        km = 111.0 * torch.sqrt((d_lat[:, lead] - centre[lead, 0]) ** 2
                                + ((d_lon[:, lead] - centre[lead, 1]) * shrink) ** 2)
        rows.append({"lead_hours": hours, "valid_time": (issued + dt.timedelta(hours=hours)).isoformat(),
                     "latitude": round(latitude, 3),
                     "longitude": round((origin_lon + float(centre[lead, 1]) + 180.0) % 360.0 - 180.0, 3),
                     "radius_km_50": round(float(km.quantile(0.5)), 1),
                     "radius_km_90": round(float(km.quantile(0.9)), 1)})
    return rows


def decode_consensus(
    outputs: dict,
    origin_lat: float,
    origin_lon: float,
    lead_times_hours: tuple[int, ...],
    issued: dt.datetime,
    weights: torch.Tensor,
    storm_state: torch.Tensor,
    batch_index: int = 0,
    rollout: dict | None = None,
) -> dict:
    """
    The multi-model track: every member's position at every lead, and their weighted blend.

    The members are those of :data:`naturev1.TRACK_MEMBERS` -- the learned head's most likely scenario
    and its mixture mean, the steering-flow track that came in as ``track_prior``, and extrapolation of
    the last twelve hours -- blended with per-lead weights fitted on held-out storms. The blend is the
    headline track; the members are kept beside it so that a forecaster can see when they disagree,
    which is itself information.
    """
    from .model import TRACK_MEMBERS
    from .steering import extrapolation_track

    from .storms import consensus_track

    probabilities = outputs["mode_logits"][batch_index].softmax(-1)
    displacement = outputs["displacement"][batch_index]
    missing = torch.full_like(outputs["track_prior"][batch_index], float("nan"))
    rollout = rollout or {}
    members = torch.stack([
        displacement[int(probabilities.argmax())],
        (displacement * probabilities.view(-1, 1, 1)).sum(0),
        outputs["track_prior"][batch_index],
        extrapolation_track(storm_state.view(1, -1), lead_times_hours)[0],
        torch.as_tensor(rollout.get("rollout_steering", missing), dtype=missing.dtype).reshape(missing.shape),
        torch.as_tensor(rollout.get("rollout_vortex", missing), dtype=missing.dtype).reshape(missing.shape),
        torch.as_tensor(rollout.get("analog_track", missing), dtype=missing.dtype).reshape(missing.shape),
        torch.as_tensor(rollout.get("rollout_ensemble", missing), dtype=missing.dtype).reshape(missing.shape),
    ], 1).float()                                                                  # (L, M, 2)
    width = members.shape[1]
    weights = weights.float()
    if weights.shape[-1] < width:                                                  # fitted before 0.12
        weights = torch.cat([weights, weights.new_zeros(weights.shape[0], width - weights.shape[-1])], -1)
    blended = consensus_track(members.unsqueeze(0), weights)[0]                     # (L, 2)

    def position(d_lat, d_lon):
        if not (math.isfinite(float(d_lat)) and math.isfinite(float(d_lon))):
            return None
        return {"latitude": round(origin_lat + float(d_lat), 3),
                "longitude": round((origin_lon + float(d_lon) + 180.0) % 360.0 - 180.0, 3)}

    points = []
    for lead, hours in enumerate(lead_times_hours):
        row = {"lead_hours": hours, "valid_time": (issued + dt.timedelta(hours=hours)).isoformat(),
               **(position(*blended[lead]) or {"latitude": None, "longitude": None}),
               "members": {name: position(*members[lead, m]) for m, name in enumerate(TRACK_MEMBERS)}}
        points.append(row)
    return {
        "track": points,
        "weights": {name: [round(float(w), 3) for w in weights[:, m]] for m, name in enumerate(TRACK_MEMBERS)},
        "note": "a weighted blend of the learned track, the steering-flow track, extrapolation and -- when "
                "given -- the tracks through the model's own rollout; weights per lead were fitted on held-out "
                "seasons, and a member that was not computed drops out of the blend",
    }


def decode_landfall(
    outputs: dict, lead_times_hours: tuple[int, ...], issued: dt.datetime, batch_index: int = 0,
    trained: dict | None = None,
) -> dict:
    """Landfall probability per lead time, plus the window it is most likely to happen in."""
    # "Landfall by this lead" can only grow with the lead. The head does not always say so (the red team
    # found it falling in a fifth of forecasts), so the running maximum is reported.
    probability = torch.cummax(outputs["landfall_logit"][batch_index].sigmoid(), dim=0).values
    spread = outputs["landfall_time_log_var"][batch_index].mul(0.5).exp()
    # No loss has ever supervised the timing spread, so without evidence of training it is left out
    # rather than printed as if it meant something.
    timing = not _untrained("landfall_timing", trained) and trained is not None
    by_lead = []
    for i, hours in enumerate(lead_times_hours):
        row = {"lead_hours": hours, "valid_time": (issued + dt.timedelta(hours=hours)).isoformat(),
               "probability": round(float(probability[i]), 4)}
        if timing:
            row["timing_sigma_hours"] = round(float(spread[i]), 2)
        by_lead.append(row)
    peak = int(probability.argmax())
    # Probability of landfall by the final lead time is the largest cumulative value, since the head is
    # trained as "by this lead" rather than "at this lead".
    bundle = {
        "by_lead": by_lead,
        "peak_probability": round(float(probability.max()), 4),
        "most_likely_lead_hours": lead_times_hours[peak],
        "most_likely_time": (issued + dt.timedelta(hours=lead_times_hours[peak])).isoformat(),
        "makes_landfall_likely": bool(probability.max() > 0.5),
    }
    if timing:
        bundle["timing_sigma_hours"] = round(float(spread[peak]), 2)
    return bundle


def decode_intensity(
    outputs: dict, lead_times_hours: tuple[int, ...], issued: dt.datetime, batch_index: int = 0
) -> list[dict]:
    """Peak wind and minimum central pressure per lead, with 90% intervals."""
    mean = outputs["intensity_mean"][batch_index]
    sigma = outputs["intensity_log_var"][batch_index].mul(0.5).exp()
    rows = []
    for i, hours in enumerate(lead_times_hours):
        wind, pressure = float(mean[i, 0]), float(mean[i, 1])
        wind_sigma, pressure_sigma = float(sigma[i, 0]), float(sigma[i, 1])
        rows.append(
            {
                "lead_hours": hours,
                "valid_time": (issued + dt.timedelta(hours=hours)).isoformat(),
                "max_wind_ms": round(wind, 1),
                "max_wind_90pct": [round(wind - 1.645 * wind_sigma, 1), round(wind + 1.645 * wind_sigma, 1)],
                "min_pressure_hpa": round(pressure, 1),
                "min_pressure_90pct": [
                    round(pressure - 1.645 * pressure_sigma, 1),
                    round(pressure + 1.645 * pressure_sigma, 1),
                ],
                "saffir_simpson": _category(wind),
            }
        )
    return rows


def decode_eyewall(
    outputs: dict, lead_times_hours, issued, batch_index: int = 0, trained: dict | None = None
) -> list[dict]:
    """
    The storm's core per lead: peak eyewall wind, the radius it sits at, and the wind footprint.

    Radius of maximum wind is reported with a wide interval on purpose. It is observed on under 5% of
    the best-track record, so the model has genuinely little to go on, and a head that reported it
    confidently would be lying about what the archive contains. The wind radii -- how far 34, 50 and 64
    kt winds reach into each quadrant -- are far better populated and describe the same structure from
    the outside in, which is why they are predicted jointly and are worth more attention when RMW is
    absent.
    """
    peak = outputs["eyewall_peak_wind_kt"][batch_index]
    peak_sigma = outputs["eyewall_peak_wind_log_var"][batch_index].mul(0.5).exp()
    rmw = outputs["eyewall_rmw_nmi"][batch_index]
    rmw_sigma = outputs["eyewall_rmw_log_var"][batch_index].mul(0.5).exp()
    # Stronger winds reach less far: the 50 kt radius cannot exceed the 34 kt one, nor 64 kt the 50 kt one,
    # and the radius of maximum wind sits inside the gale radius. The heads are trained separately and
    # broke these in about a fifth of held-out forecasts, so the structure is made consistent here.
    radii = torch.cummin(outputs["wind_radii_nmi"][batch_index].clamp_min(0.0), dim=1).values   # (L, 3, 4)
    gale = radii[:, 0].amax(-1)
    rmw = torch.where(gale > 0, torch.minimum(rmw, gale), rmw)

    entries = []
    for index, hours in enumerate(lead_times_hours):
        wind = float(peak[index])
        entry = {
            "lead_hours": hours,
            "valid_time": (issued + dt.timedelta(hours=hours)).isoformat(),
            "peak_wind_kt": round(wind, 1),
            "peak_wind_90pct": [round(wind - 1.645 * float(peak_sigma[index]), 1),
                                round(wind + 1.645 * float(peak_sigma[index]), 1)],
            "saffir_simpson": _category(wind / 1.94384),
            "rmw_nmi": round(float(rmw[index]), 1),
            "rmw_90pct": [round(max(float(rmw[index]) - 1.645 * float(rmw_sigma[index]), 0.0), 1),
                          round(float(rmw[index]) + 1.645 * float(rmw_sigma[index]), 1)],
            "wind_radii_nmi": {
                f"{int(threshold)}kt": {
                    quadrant: round(float(radii[index, group, corner]), 1)
                    for corner, quadrant in enumerate(("NE", "SE", "SW", "NW"))
                }
                for group, threshold in enumerate(WIND_RADII_THRESHOLDS_KT)
            },
        }
        if _untrained("eyewall_rmw", trained):
            for key in ("rmw_nmi", "rmw_90pct"):
                entry.pop(key)
            entry["rmw"] = _UNAVAILABLE
        if _untrained("wind_radii", trained):
            entry["wind_radii_nmi"] = _UNAVAILABLE
        entries.append(entry)
    return entries


def decode_rapid_intensification(
    outputs: dict, lead_times_hours, issued, batch_index: int = 0, trained: dict | None = None
) -> dict:
    """
    Will this storm intensify rapidly, by how much, and when.

    RI fires on about 4% of eligible track points at the 30-knot threshold. That base rate is why the
    probability is reported directly rather than as a yes/no: a "no" is right 96% of the time and worth
    nothing, while a calibrated 25% is the number a forecaster can act on. The expected 24-hour change
    comes with it, because "probably not rapid, but +20 kt" is a materially different warning from
    "probably not rapid, and steady".
    """
    probabilities = torch.sigmoid(outputs["ri_logits"][batch_index])
    change = float(outputs["ri_delta_wind_kt"][batch_index])
    change_sigma = float(outputs["ri_delta_log_var"][batch_index].mul(0.5).exp())
    onset = torch.softmax(outputs["ri_onset_logits"][batch_index], dim=-1)
    likeliest = int(onset.argmax())

    bundle = {
        f"probability_{int(threshold)}kt": round(float(probabilities[index]), 4)
        for index, threshold in enumerate(RI_THRESHOLDS_KT)
    }
    bundle.update({
        "window_hours": 24,
        "expected_change_kt": round(change, 1),
        "expected_change_90pct": [round(change - 1.645 * change_sigma, 1),
                                  round(change + 1.645 * change_sigma, 1)],
        "likeliest_onset_hours": lead_times_hours[likeliest],
        "onset_probability": round(float(onset[likeliest]), 4),
        "onset_by_lead": {int(hours): round(float(onset[index]), 4)
                          for index, hours in enumerate(lead_times_hours)},
    })
    if _untrained("ri_delta", trained):
        for key in ("expected_change_kt", "expected_change_90pct"):
            bundle.pop(key)
    # No dataset in this package supplies an onset label, so the onset head is never trained; it is
    # only reported when a training record says otherwise.
    if trained is None or _untrained("ri_onset", trained):
        for key in ("likeliest_onset_hours", "onset_probability", "onset_by_lead"):
            bundle.pop(key)
    return bundle


def _category(wind_ms: float) -> str:
    """Saffir-Simpson from 1-minute sustained wind in m/s."""
    knots = wind_ms * 1.94384
    for threshold, label in ((137, "Category 5"), (113, "Category 4"), (96, "Category 3"),
                             (83, "Category 2"), (64, "Category 1"), (34, "Tropical Storm")):
        if knots >= threshold:
            return label
    return "Tropical Depression"


#: Physical units of each surface field once decoded, and a factor to a friendlier unit.
FIELD_UNITS = {"t2m": ("K", 1.0), "mslp": ("hPa", 0.01), "u10": ("m/s", 1.0), "v10": ("m/s", 1.0),
               "precip_rate": ("mm/6h", 1000.0)}


def decode_point_forecast(
    outputs: dict,
    point_index: int,
    lead_times_hours: tuple[int, ...],
    issued: dt.datetime,
    batch_index: int = 0,
    normalizer=None,
    trained: dict | None = None,
) -> list[dict]:
    """
    "Will it rain or be sunny here" -- the full per-location forecast, at one output point.

    The heads predict in *normalized* units -- standard deviations from each variable's mean, after
    its transform. Printed raw, a 2 m temperature of 285 K comes out as "0.35", which reads as a
    number and means nothing. Pass the dataset's ``normalizer`` and every value comes back in physical
    units, with its 90% range decoded end to end (so a log-transformed field such as precipitation gets
    an honest, asymmetric range). Fields with no training target -- ERA5 carries no relative humidity
    or cloud fraction in this configuration -- are reported as unavailable rather than invented.

    Weather type comes back as a calibrated probability over categories rather than a single label, so
    "60% light rain, 30% overcast" stays visible instead of collapsing to "rain" -- when that head has
    been trained. Nothing in this package supplies weather-type labels yet, so with a training record
    it is marked unavailable.
    """
    from .era5 import ERA5_TO_SURFACE

    source = {surface: variable for variable, surface in ERA5_TO_SURFACE.items()}
    mean = outputs["field_mean"][batch_index, point_index].float()
    sigma = outputs["field_log_var"][batch_index, point_index].float().mul(0.5).exp()
    type_probabilities = outputs["weather_type_logits"][batch_index, point_index].softmax(-1)
    fields_trained = not _untrained("field", trained)
    types_trained = trained is None or not _untrained("weather_type", trained)

    rows = []
    for i, hours in enumerate(lead_times_hours):
        fields = {}
        for j, name in enumerate(SURFACE_FIELDS):
            value, spread = float(mean[i, j]), float(sigma[i, j])
            variable = source.get(name)
            if not fields_trained:
                fields[name] = _UNAVAILABLE
            elif normalizer is None:
                fields[name] = {
                    "value": round(value, 3), "sigma": round(spread, 3), "units": "normalized",
                    "range_90pct": [round(value - 1.645 * spread, 3), round(value + 1.645 * spread, 3)],
                }
            elif variable is None or variable not in normalizer.variables:
                fields[name] = {"available": False, "reason": "no training target for this field"}
            else:
                unit, factor = FIELD_UNITS.get(name, ("", 1.0))

                def decode(v, variable=variable, factor=factor):
                    return float(normalizer.decode(torch.tensor(v, dtype=torch.float64), variable)) * factor

                fields[name] = {
                    "value": round(decode(value), 3), "units": unit,
                    "range_90pct": [round(decode(value - 1.645 * spread), 3),
                                    round(decode(value + 1.645 * spread), 3)],
                }
        row = {"lead_hours": hours, "valid_time": (issued + dt.timedelta(hours=hours)).isoformat(),
               "fields": fields}
        if types_trained:
            ranked = sorted(
                ((WEATHER_TYPES[k], float(type_probabilities[i, k])) for k in range(len(WEATHER_TYPES))),
                key=lambda pair: -pair[1],
            )
            row.update({"weather": ranked[0][0], "weather_confidence": round(ranked[0][1], 4),
                        "weather_probabilities": {name: round(p, 4) for name, p in ranked}})
        else:
            row["weather"] = _UNAVAILABLE
        rows.append(row)
    return rows


def nearest_point(grid, latitude: float, longitude: float) -> int:
    """Index of the output-grid sample closest to a latitude/longitude, for a point forecast."""
    target = torch.tensor(
        [[math.radians(latitude), math.radians(longitude) % (2 * math.pi)]], dtype=grid.coords.dtype
    )
    return int((grid.points - grid.geometry.embed(target)).norm(dim=-1).argmin())


def build_forecast(
    outputs: dict,
    lead_times_hours: tuple[int, ...],
    issued: dt.datetime,
    storm_center: tuple[float, float] | None = None,
    output_grid=None,
    point_of_interest: tuple[float, float] | None = None,
    batch_index: int = 0,
    normalizer=None,
    trained: dict | None = None,
    inputs: tuple[str, ...] = (),
    consensus: dict | None = None,
    storm_physics=None,
    land_fn=None,
    model=None,
    hazard_samples: int = 400,
) -> dict:
    """
    Assemble the whole forecast bundle: track scenarios, landfall, intensity, local weather, ENSO -- and, from
    0.12, the storm's physics, the corrected intensity consensus, the situation-dependent cone, and the wind
    hazard at ``point_of_interest`` (probability of 34/50/64 kt by each lead, gale arrival, rain).

    Args:
        outputs: what :meth:`NatureV1.forward` returned.
        lead_times_hours: the model's lead times.
        issued: analysis time, UTC.
        storm_center: ``(lat, lon)`` of the current centre. Omit for a non-storm run and the track,
            landfall, intensity, eyewall and rapid-intensification sections are left out rather than
            invented.
        output_grid: the grid gridded fields were decoded onto, needed for a point forecast.
        point_of_interest: ``(lat, lon)`` to produce a local forecast for.
        normalizer: the dataset's normalizer, so gridded values come back in physical units.
        trained: :meth:`NatureV1.trained_heads`. Every section whose head has never had a training
            step is replaced by ``{"available": False, ...}`` instead of an initialisation dressed as a
            forecast. Strongly recommended; without it nothing can be flagged.
        inputs: which inputs the forecast was run from, e.g. ``("satellite",)``. If any of their
            encoders is untrained the whole bundle is marked untrustworthy -- an untrained input path
            turns every downstream number into noise, however trained the heads are.
        consensus: ``{"weights": model.track_consensus, "storm_state": (F,) tensor}`` to add the
            multi-model ``track_consensus`` section. Left out when the weights were never fitted, or
            when the outputs carry no steering track. Add ``"rollout": {"rollout_steering": (L, 2),
            "rollout_vortex": (L, 2)}`` (:func:`naturev1.tracker.rollout_member_tracks`) to bring the
            rollout members in; without them those members drop out of the blend.
        storm_physics: ``(F,)`` or ``(B, F)`` from :func:`naturev1.thermo.storm_physics`, for the ``physics``
            section (potential intensity, ventilation, phase -- tropical, hybrid or extratropical).
        land_fn: ``(lat, lon) -> land fraction`` arrays, for inland decay and roughness in the wind hazard.
        model: the model, for its fitted statistical parameters (the consensus cone).
    """
    bundle = {
        "issued": issued.isoformat(),
        "lead_times_hours": list(lead_times_hours),
        "trustworthy": True,
    }
    if trained is None:
        bundle["note"] = ("no training record supplied: every section is shown, including heads that "
                          "may never have been trained. Pass trained=model.trained_heads().")
    cold = [name for name in inputs if _untrained(f"{name}_encoder", trained)]
    if cold:
        bundle["trustworthy"] = False
        bundle["warning"] = (f"the {', '.join(cold)} input path has never been trained, so this bundle is "
                             "noise shaped like a forecast. Train on that input, or forecast from the "
                             "analysis the model was trained on.")

    score, threshold = outputs.get("feature_score"), outputs.get("feature_threshold")
    if score is not None and threshold is not None and float(score[batch_index]) > float(threshold[batch_index]):
        bundle["trustworthy"] = False
        bundle["warning"] = (f"the storm features are unlike anything in training (score {float(score[batch_index]):.1f} "
                             f"against a limit of {float(threshold[batch_index]):.1f}): the input is probably "
                             "corrupted or far outside the model's experience. Do not use this forecast.")
    if _untrained("enso", trained) or "enso_mean" not in outputs:
        bundle["enso"] = _UNAVAILABLE
    else:
        bundle["enso"] = {
            "nino34_index": round(float(outputs["enso_mean"][batch_index]), 3),
            "sigma": round(float(outputs["enso_log_var"][batch_index].mul(0.5).exp()), 3),
            "phase": _enso_phase(float(outputs["enso_mean"][batch_index])),
        }
    if storm_center is not None and not all(math.isfinite(float(v)) for v in storm_center):
        # Every storm product is relative to the centre; without one there is nothing honest to show.
        missing = {"available": False, "reason": "no valid storm centre was given"}
        bundle["storm_center"] = missing
        for key in ("track_scenarios", "landfall", "intensity", "eyewall", "rapid_intensification"):
            bundle[key] = missing
        storm_center = None
    if storm_center is not None:
        bundle["storm_center"] = {"latitude": storm_center[0], "longitude": storm_center[1]}
        if not _untrained("track", trained):
            bundle["track_cone"] = decode_track_cone(outputs, storm_center[0], storm_center[1],
                                                     lead_times_hours, issued, batch_index)
        bundle["track_scenarios"] = (_UNAVAILABLE if _untrained("track", trained) else decode_track(
            outputs, storm_center[0], storm_center[1], lead_times_hours, issued, batch_index))
        weights = None if consensus is None else consensus.get("weights")
        if (weights is not None and float(weights.abs().sum()) > 0 and "track_prior" in outputs
                and not _untrained("track", trained)):
            bundle["track_consensus"] = decode_consensus(
                outputs, storm_center[0], storm_center[1], lead_times_hours, issued, weights,
                consensus["storm_state"], batch_index, consensus.get("rollout"))
        bundle["landfall"] = (_UNAVAILABLE if _untrained("landfall", trained) else decode_landfall(
            outputs, lead_times_hours, issued, batch_index, trained))
        bundle["intensity"] = (_UNAVAILABLE if _untrained("intensity", trained) else decode_intensity(
            outputs, lead_times_hours, issued, batch_index))
        bundle["eyewall"] = (_UNAVAILABLE if _untrained("eyewall_peak", trained) else decode_eyewall(
            outputs, lead_times_hours, issued, batch_index, trained))
        bundle["rapid_intensification"] = (
            _UNAVAILABLE if _untrained("ri", trained)
            else decode_rapid_intensification(outputs, lead_times_hours, issued, batch_index, trained))
        if "intensity_consensus_kt" in outputs and not _untrained("intensity", trained):
            bundle["intensity_consensus"] = decode_intensity_consensus(outputs, lead_times_hours, issued, batch_index)
        if storm_physics is not None:
            physics = storm_physics if storm_physics.dim() == 1 else storm_physics[batch_index]
            bundle["physics"] = decode_physics(physics, outputs, batch_index)
        if model is not None and "track_consensus" in bundle and int(model.statistics.fitted[5]) > 0:
            bundle["consensus_cone"] = decode_consensus_cone(bundle["track_consensus"], model, storm_center)
        if point_of_interest is not None and not _untrained("track", trained) and not _untrained("intensity", trained):
            bundle["wind_hazard"] = decode_wind_hazard(outputs, storm_center, point_of_interest, lead_times_hours,
                                                       issued, batch_index, land_fn, trained, hazard_samples)
    if point_of_interest is not None and output_grid is not None:
        index = nearest_point(output_grid, *point_of_interest)
        bundle["point_forecast"] = {
            "requested": {"latitude": point_of_interest[0], "longitude": point_of_interest[1]},
            "forecast": decode_point_forecast(outputs, index, lead_times_hours, issued, batch_index,
                                              normalizer, trained),
        }
    return bundle


def decode_intensity_consensus(outputs: dict, lead_times_hours, issued: dt.datetime, batch_index: int = 0) -> dict:
    """The corrected intensity consensus per lead, with every member beside it (kt)."""
    from .statistical import INTENSITY_MEMBERS

    blended = outputs["intensity_consensus_kt"][batch_index]
    members = outputs.get("intensity_members_kt")
    rows = []
    for lead, hours in enumerate(lead_times_hours):
        row = {"lead_hours": hours, "valid_time": (issued + dt.timedelta(hours=hours)).isoformat(),
               "wind_kt": round(float(blended[lead]), 1), "category": _category(float(blended[lead]) * 0.514444)}
        if members is not None:
            row["members"] = {name: (round(float(members[batch_index, lead, m]), 1)
                                     if math.isfinite(float(members[batch_index, lead, m])) else None)
                              for m, name in enumerate(INTENSITY_MEMBERS)}
        rows.append(row)
    return {"by_lead": rows, "note": "bias-corrected, weighted blend of the network's heads, a SHIPS-style "
                                     "regression, a logistic-growth model, climatology-persistence, the satellite "
                                     "network and the rollout; weights per lead fitted on held-out seasons"}


def phase_label(b_m: float, vtl: float, vtu: float) -> str:
    """Hart (2003) phase from thermal asymmetry B (m) and the lower and upper thermal winds."""
    if b_m < 10.0 and vtl > 0 and vtu > 0:
        return "tropical (symmetric, deep warm core)"
    if b_m < 10.0 and vtl > 0:
        return "shallow warm core (subtropical-like)"
    if b_m >= 10.0 and vtl > 0:
        return "transitioning (asymmetric, warm core below)"
    if b_m >= 10.0 and vtl <= 0:
        return "extratropical (asymmetric, cold core)"
    return "cold core, symmetric (occluded or upper low)"


def decode_physics(physics, outputs: dict, batch_index: int = 0) -> dict:
    """The storm's thermodynamics in a forecaster's units, from the scaled physics vector."""
    from .thermo import STORM_PHYSICS_FEATURES

    f = {name: float(physics[i]) for i, name in enumerate(STORM_PHYSICS_FEATURES)}
    if f["observed"] < 0.5:
        return {"available": False, "reason": "the state carries no pressure-level winds, temperature and humidity"}
    potential_kt = f["potential_intensity"] * 50.0 / 0.514444
    b, vtl, vtu = f["phase_b"] * 20.0, f["phase_vtl"] * 100.0, f["phase_vtu"] * 100.0
    ventilation = f["ventilation"] / 10.0
    return {
        "potential_intensity_kt": round(potential_kt, 0),
        "below_potential_kt": round(f["intensity_gap"] * 50.0 / 0.514444, 0),
        "outflow_temperature_c": round(f["outflow_temperature"] * 15.0 + 200.0 - 273.15, 1),
        "ventilation_index": round(ventilation, 3),
        "ventilation": ("strongly inhibits intensification" if ventilation > 0.1 else
                        "inhibits intensification" if ventilation > 0.05 else "favourable"),
        "mid_level_entropy_deficit": round(f["entropy_deficit"], 2),
        "cape_j_per_kg": round(f["cape"] * 2000.0, 0),
        "divergence_200hpa_per_s": round(f["divergence_200"] * 1e-5, 7),
        "vorticity_850hpa_per_s": round(f["vorticity_850"] * 1e-5, 7),
        "mid_level_humidity": round(f["rh_mid"], 2),
        "phase": {"b_m": round(b, 1), "lower_thermal_wind": round(vtl, 0), "upper_thermal_wind": round(vtu, 0),
                  "label": phase_label(b, vtl, vtu)},
        "genesis_potential": round(math.expm1(f["genesis_potential"]), 2),
        "land_share_within_300km": round(f["land_share"], 2),
    }


def decode_consensus_cone(consensus: dict, model, storm_center) -> list[dict]:
    """The predicted-consensus-error radius at each lead, from how far apart the members are (km)."""
    from .statistical import consensus_radius_km
    from .steering import great_circle_km

    rows = []
    for index, point in enumerate(consensus["track"]):
        if point.get("latitude") is None:
            continue
        distances = [float(great_circle_km(torch.tensor(point["latitude"]), torch.tensor(point["longitude"]),
                                           torch.tensor(m["latitude"]), torch.tensor(m["longitude"])))
                     for m in point["members"].values() if m is not None]
        spread = torch.tensor([[sum(distances) / max(len(distances), 1)]]).expand(1, len(consensus["track"]))
        radius = float(consensus_radius_km(model, spread.float())[0, index])
        rows.append({"lead_hours": point["lead_hours"], "latitude": point["latitude"], "longitude": point["longitude"],
                     "member_spread_km": round(spread[0, 0].item(), 0), "radius_km": round(radius, 0)})
    return rows


def decode_wind_hazard(outputs: dict, storm_center, point, lead_times_hours, issued: dt.datetime, batch_index: int,
                       land_fn, trained, samples: int) -> dict:
    """Wind-speed probabilities, gale arrival and rain at one point, by Monte Carlo over the forecast."""
    from .windfield import THRESHOLDS_KT, wind_hazard

    result = wind_hazard(outputs, batch_index, storm_center, lead_times_hours, [point[0]], [point[1]], samples=samples,
                         land_fn=land_fn, trained=trained)
    rows = []
    for lead, hours in enumerate(lead_times_hours):
        rows.append({"lead_hours": hours, "valid_time": (issued + dt.timedelta(hours=hours)).isoformat(),
                     **{f"p_{int(t)}kt": round(float(result["probability"][0, lead, i]), 3)
                        for i, t in enumerate(THRESHOLDS_KT)}})
    arrival = result["arrival_hours"][0]

    def when(hours):
        return None if not math.isfinite(float(hours)) else (issued + dt.timedelta(hours=float(hours))).isoformat()

    return {"point": {"latitude": point[0], "longitude": point[1]}, "by_lead": rows,
            "gale_arrival": {"earliest_reasonable": when(arrival[0]), "most_likely": when(arrival[1])},
            "median_peak_wind_kt": round(float(result["peak_kt"][0]), 0),
            "rain_mm": {"median": round(float(result["rain_mm"][0, 0]), 0), "p90": round(float(result["rain_mm"][0, 1]), 0)},
            "samples": result["samples"],
            "note": "Holland-profile wind fields with inflow, motion asymmetry, land roughness and inland decay, over "
                    "coherent realisations of the forecast; rain from the R-CLIPER climatology"}


#: What an untrained section is replaced with.
_UNAVAILABLE = {"available": False,
                "reason": "this output has never been trained; showing it would be showing its initialisation"}


def _untrained(name: str, trained: dict | None) -> bool:
    """True only when a training record exists and says this output never had a step."""
    return trained is not None and int(trained.get(name, 0)) == 0


def _enso_phase(index: float) -> str:
    """NOAA's convention: the Nino 3.4 anomaly thresholds at +/- 0.5 K."""
    if index >= 1.5:
        return "Strong El Nino"
    if index >= 0.5:
        return "El Nino"
    if index <= -1.5:
        return "Strong La Nina"
    if index <= -0.5:
        return "La Nina"
    return "Neutral"
