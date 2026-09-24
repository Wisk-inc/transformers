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
                    "cone_radius_km_68": round(major * 111.0, 1),
                    "cone_radius_km_95": round(major * 111.0 * 2.448, 1),
                }
            )
        scenarios.append(
            {"rank": rank + 1, "probability": round(float(probabilities[mode]), 4), "track": points}
        )
    return scenarios


def decode_landfall(
    outputs: dict, lead_times_hours: tuple[int, ...], issued: dt.datetime, batch_index: int = 0,
    trained: dict | None = None,
) -> dict:
    """Landfall probability per lead time, plus the window it is most likely to happen in."""
    probability = outputs["landfall_logit"][batch_index].sigmoid()
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
    radii = outputs["wind_radii_nmi"][batch_index]

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
) -> dict:
    """
    Assemble the whole forecast bundle: track scenarios, landfall, intensity, local weather, ENSO.

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

    if _untrained("enso", trained):
        bundle["enso"] = _UNAVAILABLE
    else:
        bundle["enso"] = {
            "nino34_index": round(float(outputs["enso_mean"][batch_index]), 3),
            "sigma": round(float(outputs["enso_log_var"][batch_index].mul(0.5).exp()), 3),
            "phase": _enso_phase(float(outputs["enso_mean"][batch_index])),
        }
    if storm_center is not None:
        bundle["storm_center"] = {"latitude": storm_center[0], "longitude": storm_center[1]}
        bundle["track_scenarios"] = (_UNAVAILABLE if _untrained("track", trained) else decode_track(
            outputs, storm_center[0], storm_center[1], lead_times_hours, issued, batch_index))
        bundle["landfall"] = (_UNAVAILABLE if _untrained("landfall", trained) else decode_landfall(
            outputs, lead_times_hours, issued, batch_index, trained))
        bundle["intensity"] = (_UNAVAILABLE if _untrained("intensity", trained) else decode_intensity(
            outputs, lead_times_hours, issued, batch_index))
        bundle["eyewall"] = (_UNAVAILABLE if _untrained("eyewall_peak", trained) else decode_eyewall(
            outputs, lead_times_hours, issued, batch_index, trained))
        bundle["rapid_intensification"] = (
            _UNAVAILABLE if _untrained("ri", trained)
            else decode_rapid_intensification(outputs, lead_times_hours, issued, batch_index, trained))
    if point_of_interest is not None and output_grid is not None:
        index = nearest_point(output_grid, *point_of_interest)
        bundle["point_forecast"] = {
            "requested": {"latitude": point_of_interest[0], "longitude": point_of_interest[1]},
            "forecast": decode_point_forecast(outputs, index, lead_times_hours, issued, batch_index,
                                              normalizer, trained),
        }
    return bundle


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
