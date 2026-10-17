# Copyright 2026 Nathan. Apache-2.0.
"""
Stage two: best-track labels, paired with the reanalysis the backbone was pretrained on.

Stage one taught the backbone what the atmosphere does, from 1.6e10 free supervised values. This module
supplies the other half -- what a storm did -- and it is small on purpose, because it is small in
reality: 55,230 Atlantic track points, 1,167 of them landfalls, 1,839 rapid intensifications at the
30-knot threshold, 2,587 with a radius of maximum wind. That is the entire observational record, and no
amount of engineering makes it bigger.

What engineering can do is make sure those few thousand labels are spent on a few thousand parameters
instead of eighty-nine million. :meth:`naturev1.NatureV1.freeze_backbone` leaves 0.96M trainable, and
this module hands those heads real inputs -- the actual global atmosphere at the hour the storm was
observed -- rather than the random tensors a placeholder pipeline would feed them.

Three decisions here are the ones that decide whether the result means anything:

**Windows are paired by timestamp, not by index.** Every sample is a genuine (atmosphere, outcome) pair:
the ERA5 state at the synoptic hour the Hurricane Center recorded the storm at, and what that storm went
on to do at +6 through +120 hours. Points the reanalysis does not cover -- anything before 1959, and
HURDAT2's off-synoptic landfall specials -- are dropped rather than approximated.

**Scarce labels are masked, never imputed.** Radius of maximum wind exists on 4.7% of points. Filling
the other 95.3% with a plausible number teaches the head that plausible number. Every target here
arrives as NaN where it was not observed, and the losses skip it.

**Splits are by season.** Six-hourly points from one storm are near-duplicates of each other; split them
at random and the same hurricane lands on both sides of the split. Holding out whole seasons also keeps
that year's ENSO state and sea-surface temperatures out of training, which is the honest test.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from .besttrack import Track, rapid_intensification
from .heartbeat import beat
from .model import (
    PEAK_WIND_ANCHOR_KT,
    PRESSURE_ANCHOR_HPA,
    RI_THRESHOLDS_KT,
    STORM_ENVIRONMENT_FEATURES,
    SURFACE_FIELDS,
    TRACK_MEMBERS,
    WIND_RADII_THRESHOLDS_KT,
)
from .steering import displacement_km, extrapolation_track, steering_track, storm_environment


#: Feature banks written by an earlier version lack the far pool and the physics; they are rebuilt.
BANK_VERSION = 2


#: Knots to metres per second. :func:`naturev1.forecast.decode_intensity` reads the intensity head in
#: m/s and absolute hPa, so the targets are built in those units rather than in the archive's knots.
KT_TO_MS = 0.514444


def _displacement(track: Track, start: int, end: int) -> tuple[float, float]:
    """Latitude and longitude change between two track points, taking the short way around."""
    dlat = float(track.latitude[end] - track.latitude[start])
    dlon = float((track.longitude[end] - track.longitude[start] + 180.0) % 360.0 - 180.0)
    return dlat, dlon


def _nearest_points(times: np.ndarray, targets: np.ndarray, tolerance_hours: float) -> np.ndarray:
    """
    For each target time, the index of the nearest entry in ``times``, or -1 if none is close enough.

    Best tracks are six-hourly on the synoptic hours, which is exactly ERA5's cadence, so almost every
    point matches to the minute. The exceptions are HURDAT2's landfall specials, recorded at the hour
    the eye actually crossed the coast; those fall outside the tolerance and are dropped rather than
    snapped to a state up to three hours away from the event they describe.
    """
    order = np.searchsorted(times, targets)
    order = np.clip(order, 1, len(times) - 1)
    before, after = times[order - 1], times[order]
    nearest = np.where(targets - before <= after - targets, order - 1, order)
    gap = np.abs(times[nearest] - targets)
    return np.where(gap <= tolerance_hours * 3600, nearest, -1)


class StormTargets:
    """
    Best-track outcomes for one point in one storm, at every forecast lead.

    Built once per sample and cached, because the RI walk and the wind-radii lookups cost far more than
    the tensor construction and never change.
    """

    __slots__ = ("track", "point", "offsets", "delta", "leads", "lead_points")

    def __init__(self, track: Track, point: int, offsets: np.ndarray, delta: np.ndarray,
                 lead_points: np.ndarray | None = None, cadence_hours: float = 6.0) -> None:
        self.track, self.point, self.offsets, self.delta = track, point, offsets, delta
        self.leads = len(offsets)
        if lead_points is None:
            lead_points = lead_track_points(track, point, offsets, cadence_hours)
        self.lead_points = lead_points

    def build(self) -> dict[str, torch.Tensor]:
        track, start, leads = self.track, self.point, self.leads
        nan = float("nan")

        displacement = torch.zeros(leads, 2)
        valid = torch.zeros(leads)
        intensity = torch.zeros(leads, 2)
        intensity_mask = torch.zeros(leads, 2)
        landfall = torch.zeros(leads)
        landfall_mask = torch.zeros(leads)
        peak_wind = torch.full((leads,), nan)
        rmw = torch.full((leads,), nan)
        radii = torch.full((leads, len(WIND_RADII_THRESHOLDS_KT), 4), nan)

        for lead in range(leads):
            end = int(self.lead_points[lead])
            if end < 0:
                continue                      # no best-track fix at this valid time; nothing to verify
            valid[lead] = 1.0
            displacement[lead, 0], displacement[lead, 1] = _displacement(track, start, end)

            wind_kt = float(track.max_wind_kt[end])
            if np.isfinite(wind_kt):
                intensity[lead, 0] = wind_kt * KT_TO_MS
                intensity_mask[lead, 0] = 1.0
                peak_wind[lead] = wind_kt
            pressure = float(track.min_pressure_hpa[end])
            if np.isfinite(pressure):
                intensity[lead, 1] = pressure
                intensity_mask[lead, 1] = 1.0

            # Landfall is cumulative: "will it make landfall by +48h", not "exactly at +48h". The window
            # opens *after* the issue point: a storm crossing the coast right now is not a forecast.
            landfall[lead] = float(track.landfall[start + 1 : end + 1].any())
            landfall_mask[lead] = 1.0

            if np.isfinite(track.rmw_nmi[end]):
                rmw[lead] = float(track.rmw_nmi[end])
            observed = np.isfinite(track.wind_radii_nmi[end])
            if observed.any():
                radii[lead] = torch.tensor(
                    np.where(observed, track.wind_radii_nmi[end], np.nan), dtype=torch.float32
                )

        change = float(self.delta[start])
        eligible = np.isfinite(change)
        return {
            # Where this storm is now: the storm heads pool the atmosphere around it.
            "storm_center": torch.tensor([float(track.latitude[start]), float(track.longitude[start])],
                                         dtype=torch.float32),
            # What it is doing now: intensity and recent motion, the operational predictors.
            "storm_state": torch.tensor(storm_state_vector(track, start), dtype=torch.float32),
            "current_wind_kt": torch.tensor(float(track.max_wind_kt[start]), dtype=torch.float32),
            "track_target": displacement,
            "track_valid": valid,
            "intensity_target": intensity,
            "intensity_mask": intensity_mask,
            "landfall_target": landfall,
            "landfall_mask": landfall_mask,
            "eyewall_target": peak_wind,
            "rmw_target": rmw,
            "wind_radii_target": radii,
            "ri_target": torch.tensor(
                [float(eligible and change >= t) for t in RI_THRESHOLDS_KT], dtype=torch.float32
            ),
            "ri_mask": torch.full((len(RI_THRESHOLDS_KT),), float(eligible)),
            "ri_delta_target": torch.tensor(change if eligible else nan, dtype=torch.float32),
            # The season, so a consensus can be fitted on some seasons and judged on others.
            "season": torch.tensor(track.year, dtype=torch.int64),
        }


def storm_state_vector(track: Track, point: int, cadence_hours: float = 6.0) -> np.ndarray:
    """
    What the storm is doing at this fix, scaled to order one, in :data:`STORM_STATE_FEATURES` order.

    Current maximum wind and central pressure, and the change in position and wind over the previous
    twelve hours. Those are what the Hurricane Center's own statistical models start from, and they
    are exactly what a 1.5-degree reanalysis cannot show: it smears a 30 km eyewall into a broad low,
    so a category 5 and a tropical storm look alike. A missing value is zero with ``observed`` telling
    the heads which it is; a storm's first fix has no twelve-hour history and reports no motion.
    """
    wind = float(track.max_wind_kt[point])
    pressure = float(track.min_pressure_hpa[point])
    earlier = int(lead_track_points(track, point, (-2,), cadence_hours)[0])
    motion_lat = motion_lon = change = 0.0
    if earlier >= 0:
        motion_lat, motion_lon = _displacement(track, earlier, point)
        before = float(track.max_wind_kt[earlier])
        if np.isfinite(before) and np.isfinite(wind):
            change = wind - before
    observed = float(np.isfinite(wind))
    return np.array([
        (wind - PEAK_WIND_ANCHOR_KT[0]) / PEAK_WIND_ANCHOR_KT[1] if np.isfinite(wind) else 0.0,
        (pressure - PRESSURE_ANCHOR_HPA[0]) / PRESSURE_ANCHOR_HPA[1] if np.isfinite(pressure) else 0.0,
        motion_lat, motion_lon, change / 15.0, observed,
    ], dtype=np.float32)


def lead_track_points(track: Track, point: int, offsets, cadence_hours: float = 6.0) -> np.ndarray:
    """
    For each lead, the index of the best-track fix at exactly that valid time, or -1 if there is none.

    Looked up by *time*, not by position. HURDAT2 inserts extra rows at landfall -- the hour the eye
    crossed the coast -- so "twenty rows later" is not "120 hours later" on any stretch containing
    one. Indexed by position, 10.9% of the 1959-onward lead targets had the wrong valid time, and 92%
    of those were on landfalling stretches: the labels were wrong precisely for the storms that matter.
    """
    seconds = track.time.astype("datetime64[s]").astype(np.int64)
    wanted = seconds[point] + np.asarray(offsets, dtype=np.int64) * int(cadence_hours * 3600)
    found = np.searchsorted(seconds, wanted)
    inside = found < len(seconds)
    exact = np.zeros(len(wanted), dtype=bool)
    exact[inside] = seconds[found[inside]] == wanted[inside]
    return np.where(exact, found, -1).astype(np.int64)


class StormWindow(torch.utils.data.Dataset):
    """
    Reanalysis input at a storm's observed hour, with what that storm actually went on to do.

    Wraps an :class:`naturev1.ERA5Window` or :class:`naturev1.CachedERA5` whose ``indices`` were chosen
    to line up one-to-one with the track points, so the input half reuses machinery that is already
    tested rather than reimplementing the read.

    Build it with :func:`pair_tracks_with_reanalysis`, which does the alignment.
    """

    def __init__(self, base: torch.utils.data.Dataset, targets: list[StormTargets]) -> None:
        if len(base) != len(targets):
            raise ValueError(f"base has {len(base)} windows but {len(targets)} storm points were paired")
        self.base, self.targets = base, targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, item: int) -> dict:
        sample = dict(self.base[item])
        sample.update(self.targets[item].build())
        return sample

    def describe(self) -> str:
        """What the labels in this split actually contain -- worth printing before trusting a metric."""
        built = [target.build() for target in self.targets]
        total = len(built)
        if not total:
            return "no paired samples"

        def observed(key):
            stacked = torch.stack([item[key] for item in built])
            return 100.0 * torch.isfinite(stacked).float().mean()

        ri = torch.stack([item["ri_target"] for item in built])
        mask = torch.stack([item["ri_mask"] for item in built])
        lines = [f"{total:,} paired (reanalysis, outcome) samples"]
        for index, threshold in enumerate(RI_THRESHOLDS_KT):
            eligible = mask[:, index].sum()
            rate = 100.0 * ri[:, index].sum() / eligible.clamp_min(1.0)
            lines.append(f"  RI >= {threshold:.0f} kt / 24 h   {int(ri[:, index].sum()):>6,} of "
                         f"{int(eligible):>6,} eligible   base rate {rate:.2f}%")
        landfall = torch.stack([item["landfall_target"] for item in built])
        lines.append(f"  landfall by +120 h    {int(landfall[:, -1].sum()):>6,} of {total:,} "
                     f"({100.0 * landfall[:, -1].mean():.2f}%)")
        lines.append(f"  peak wind observed    {observed('eyewall_target'):.1f}% of lead slots")
        lines.append(f"  wind radii observed   {observed('wind_radii_target'):.1f}%")
        lines.append(f"  RMW observed          {observed('rmw_target'):.1f}%  <- the scarce one")
        return "\n".join(lines)


def pair_tracks_with_reanalysis(
    tracks: list[Track],
    store_times: np.ndarray,
    lead_offsets: tuple[int, ...],
    history: int = 6,
    cadence_hours: float = 6.0,
    tolerance_hours: float = 1.0,
    tropical_only: bool = True,
    ri_window_hours: int = 24,
) -> tuple[np.ndarray, list[StormTargets], dict]:
    """
    Line best-track points up with the reanalysis timesteps that cover them.

    Args:
        tracks: parsed best tracks.
        store_times: the reanalysis store's timestamps, as int64 seconds, ascending.
        lead_offsets: store-step offsets of the forecast leads, from :func:`naturev1.lead_offsets`.
        history: input frames each window needs before the storm's hour.
        tolerance_hours: how far a track point may sit from a reanalysis timestep and still pair.

    Returns:
        ``(window_starts, targets, report)``. Pass ``window_starts`` as the ``indices`` of an
        :class:`naturev1.ERA5Window`, and the two line up one-to-one.
    """
    store_times = np.asarray(store_times, dtype=np.int64)
    if not np.all(np.diff(store_times) > 0):
        raise ValueError("store_times must be strictly ascending")

    # The RI walk is over whole tracks and independent of the threshold -- it measures the 24-hour wind
    # change, and every threshold is a comparison against it -- so it runs once.
    walk = rapid_intensification(tracks, threshold_kt=min(RI_THRESHOLDS_KT),
                                 window_hours=ri_window_hours, tropical_only=tropical_only)
    deltas = walk["delta"]

    # Track points are spaced by the archive's own cadence, which must match the store's for a lead
    # offset in store steps to mean the same number of track points.
    steps = np.asarray(lead_offsets, dtype=np.int64)
    starts, targets = [], []
    considered = dropped_time = dropped_room = 0

    for track in tracks:
        seconds = track.time.astype("datetime64[s]").astype(np.int64)
        # Only points on the archive's regular grid: the off-synoptic landfall specials would otherwise
        # make a "lead offset" mean a different number of hours for different samples.
        regular = (seconds % int(cadence_hours * 3600)) == 0
        matched = _nearest_points(store_times, seconds, tolerance_hours)
        delta = deltas.get(track.storm_id)
        if delta is None:
            continue

        for point in range(len(track)):
            considered += 1
            if not regular[point] or matched[point] < 0:
                dropped_time += 1
                continue
            start = int(matched[point]) - history + 1
            if start < 0 or start + history - 1 + int(steps.max()) >= len(store_times):
                dropped_room += 1
                continue
            starts.append(start)
            targets.append(StormTargets(track, point, steps, delta,
                                        lead_track_points(track, point, steps, cadence_hours)))

    report = {
        "considered": considered,
        "paired": len(starts),
        "dropped_outside_reanalysis": dropped_time,
        "dropped_no_room": dropped_room,
        "storms": len({target.track.storm_id for target in targets}),
    }
    return np.asarray(starts, dtype=np.int64), targets, report


def format_pairing(report: dict) -> str:
    """A printable summary of what paired and what did not."""
    considered = max(report["considered"], 1)
    return "\n".join([
        f"  track points considered       {report['considered']:>8,}",
        f"  paired with reanalysis        {report['paired']:>8,}  "
        f"({100 * report['paired'] / considered:.1f}%)",
        f"  dropped, outside the record   {report['dropped_outside_reanalysis']:>8,}  "
        "(pre-1959, or off-synoptic landfall specials)",
        f"  dropped, no room for a window {report['dropped_no_room']:>8,}",
        f"  distinct storms               {report['storms']:>8,}",
    ])


# --------------------------------------------------------------------------------------------------
# Stage two, fast: the frozen backbone once per storm, then the heads on a feature bank
# --------------------------------------------------------------------------------------------------

def _backbone_fingerprint(model) -> str:
    """Changes whenever any frozen weight the storm features depend on changes."""
    heads = {id(p) for p in model.storm_head_parameters()}
    heads |= {id(p) for p in getattr(model, "satellite_parameters", lambda: [])()}
    total = torch.zeros(2, dtype=torch.float64)
    for index, parameter in enumerate(model.parameters()):
        if id(parameter) in heads:
            continue
        values = parameter.detach().double()
        total += torch.stack([values.sum() * (index + 1), values.abs().sum()]).cpu()
    return f"{total[0].item():.10e}:{total[1].item():.10e}"


def _stack_targets(targets: list[StormTargets]) -> dict[str, torch.Tensor]:
    built = [target.build() for target in targets]
    return {key: torch.stack([item[key] for item in built]) for key in built[0]}


def storm_feature_bank(
    model,
    store,
    base,
    starts: np.ndarray,
    targets: list[StormTargets],
    grid,
    cache: str | None = None,
    batch_size: int = 8,
    precision: str = "auto",
    workers: int = 4,
    progress: bool = True,
) -> dict:
    """
    Run the frozen backbone once for every storm sample and keep only what the heads read.

    Stage two used to stream a full input window per sample. With the 13 pressure levels that window
    is 89 channels over the globe for six frames -- about 60 MB -- and consecutive six-hourly points of
    one storm re-read five of their six frames. An epoch over 24,585 storm points was on the order of
    two days of downloading, for heads that hold 0.12M parameters.

    The backbone is frozen in stage two, so each sample's pooled features never change. This reads
    each storm's frames once, as one contiguous block, runs the backbone over its windows, and keeps a
    ``hidden``-wide vector per sample: 22k storm points become about 45 MB. The heads then train in
    seconds per epoch. Everything is checkpointed as it goes, and reused while the backbone is unchanged.

    Args:
        store: the ERA5 store, e.g. from :func:`naturev1.open_weatherbench_levels`.
        base: any :class:`naturev1.ERA5Window` built with the model's variables, levels and channels --
            its normalizer, variable list and padding are what the windows are built with.
        starts, targets: from :func:`pair_tracks_with_reanalysis`.
        cache: file to keep the bank in. Resumes from a partial one; rebuilt if the backbone changed.

    Returns:
        ``{"features": (N, hidden), "features_far": (N, hidden), "targets": {...}, "fingerprint": str}``.
        ``targets`` also carries, per sample, the storm's physical environment (``storm_environment``)
        and the steering-flow track (``track_prior``), both read from the same analysis frame.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .era5 import read_block, split_prepared
    from .model import calendar_features
    from .safeio import atomic_torch_save
    from .state import _autocast

    device = next(model.parameters()).device
    fingerprint = _backbone_fingerprint(model)
    count, hidden = len(starts), model.config.hidden_size
    leads = model.config.lead_times_hours
    features = torch.zeros(count, hidden)
    features_far = torch.zeros(count, hidden)
    environment = torch.zeros(count, len(STORM_ENVIRONMENT_FEATURES))
    prior = torch.zeros(count, len(leads), 2)
    done: set[str] = set()
    partial = Path(f"{cache}.partial") if cache else None

    def readable(path):
        # A cache cut off by a kill would crash every restart here; it is rebuilt instead. So is one
        # from before the bank carried the far pool and the physics -- those are what the heads train on.
        try:
            stored = torch.load(path, map_location="cpu", weights_only=False)
            usable = (stored.get("fingerprint") == fingerprint and stored["features"].shape[0] == count
                      and stored.get("version") == BANK_VERSION)
            return stored if usable else None
        except Exception as error:
            print(f"[storms] {Path(path).name} is damaged ({type(error).__name__}); rebuilding it", flush=True)
            return None

    if cache and Path(cache).exists():
        stored = readable(cache)
        if stored is not None:
            if progress:
                print(f"[storms] feature bank reused: {cache} ({count:,} samples)", flush=True)
            return stored
    if partial is not None and partial.exists():
        stored = readable(partial)
        if stored is not None:
            features, features_far, done = stored["features"], stored["features_far"], set(stored["done"])
            environment, prior = stored["environment"], stored["prior"]

    groups: dict[str, list[int]] = {}
    for index, target in enumerate(targets):
        groups.setdefault(target.track.storm_id, []).append(index)
    pending = [storm for storm in groups if storm not in done]
    history, normalizer = base.history, base.normalizer
    times = store.time.values.astype("datetime64[s]").astype(np.int64)

    def read(storm):
        members = groups[storm]
        first = int(min(starts[i] for i in members))
        last = int(max(starts[i] for i in members)) + history
        from .fallback import with_retry

        raw = with_retry(lambda: read_block(store, base.source_variables, slice(first, last), base.levels),
                         attempts=5, what=f"ERA5 read for storm {storm}")
        values, observed = split_prepared(normalizer.prepare(raw))
        if normalizer.masked:
            values = np.concatenate([values, observed[..., normalizer.masked].astype(np.float32)], axis=-1)
        block = torch.from_numpy(values.reshape(values.shape[0], -1, values.shape[-1]))
        if base.channels is not None and base.channels > block.shape[-1]:
            block = torch.nn.functional.pad(block, (0, base.channels - block.shape[-1]))
        return storm, first, block

    started, last_save = time.time(), time.time()
    model.eval()
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        for position, (storm, first, block) in enumerate(pool.map(read, pending), start=1):
            members = groups[storm]
            for chunk in range(0, len(members), batch_size):
                chosen = members[chunk : chunk + batch_size]
                offsets = [int(starts[i]) - first for i in chosen]
                analysis = torch.stack([block[o : o + history] for o in offsets]).to(device)
                stamps = torch.tensor(np.stack([times[int(starts[i]) : int(starts[i]) + history] for i in chosen]),
                                      dtype=torch.float64)
                centres = torch.stack([targets[i].build()["storm_center"] for i in chosen]).to(device)
                with _autocast(device, precision):
                    near, wide = model.storm_features(analysis, grid, calendar_features(stamps).to(device),
                                                      centres, far=True)
                features[chosen] = near.float().cpu()
                features_far[chosen] = wide.float().cpu()
                # The physics, from the same frame the backbone saw: the storm's environment now, and
                # where the steering flow as it stands would carry it.
                now = analysis[:, -1].float()
                environment[chosen] = storm_environment(now, base.variables, normalizer, grid, centres).cpu()
                prior[chosen] = steering_track(now, base.variables, normalizer, grid, centres, leads).cpu()
            done.add(storm)
            beat("storm features")
            if progress:
                rate = position / max(time.time() - started, 1e-9)
                left = (len(pending) - position) / max(rate, 1e-9)
                print(f"\r[storms] features {len(done):,}/{len(groups):,} storms "
                      f"({left / 60:.0f} min left)", end="", flush=True)
            if partial is not None and time.time() - last_save > 60:
                atomic_torch_save({"features": features, "features_far": features_far, "environment": environment,
                                   "prior": prior, "done": sorted(done), "fingerprint": fingerprint,
                                   "version": BANK_VERSION}, partial)
                last_save = time.time()
    if progress and pending:
        print(flush=True)

    stacked = _stack_targets(targets)
    stacked["storm_environment"], stacked["track_prior"] = environment, prior
    bank = {"features": features, "features_far": features_far, "targets": stacked,
            "fingerprint": fingerprint, "version": BANK_VERSION}
    if cache:
        atomic_torch_save(bank, cache)
        if partial is not None:
            partial.unlink(missing_ok=True)
    return bank


def _satellite_batch(bank, index, device, augment: bool = False):
    """
    The storm-centred infrared for these samples, ``(codes, available)``, or None if the bank has none.

    With ``augment``: a random quarter turn (rotation keeps a storm's spiral turning the right way; a
    mirror image would not), a shift of up to three pixels (the best-track centre is uncertain by about
    that much), and a brightness offset of up to 2 K (the spread between satellites and calibrations).
    """
    images = bank.get("satellite")
    if images is None:
        return None
    codes = images["ir"][index].to(device)
    available = images["available"][index].to(device)
    if augment:
        turns = int(torch.randint(0, 4, ()).item())
        codes = torch.rot90(codes, turns, dims=(-2, -1))
        shift = torch.randint(-3, 4, (2,)).tolist()
        codes = torch.roll(codes, shifts=shift, dims=(-2, -1))
        offset = torch.randint(-3, 4, (codes.shape[0], 1, 1, 1), device=device)
        codes = torch.where(codes > 0, (codes.long() + offset).clamp(1, 255), 0).to(torch.uint8)
    return codes, available


def ri2_targets(targets: dict, lead_times_hours) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RI 2.0 labels from a bank's targets: for every (window, threshold) of :data:`naturev1.satnet.RI_WINDOWS`,
    whether the best-track wind ``window`` hours later is at least ``threshold`` above the wind now --
    the endpoint definition SHIPS-RII uses. Masked where the start is not an eligible tropical fix or the
    track has no wind at the end of the window.
    """
    from .satnet import RI_WINDOWS

    leads = list(lead_times_hours)
    now = targets["current_wind_kt"].float()
    eligible = targets["ri_mask"][:, 0] > 0
    labels, masks = [], []
    for hours, thresholds in RI_WINDOWS:
        if hours in leads:
            lead = leads.index(hours)
            later = targets["intensity_target"][:, lead, 0].float() / KT_TO_MS
            seen = (targets["intensity_mask"][:, lead, 0] > 0) & torch.isfinite(now) & eligible
        else:
            later, seen = torch.zeros_like(now), torch.zeros_like(eligible)
        for threshold in thresholds:
            labels.append(((later - now) >= threshold).float() * seen.float())
            masks.append(seen.float())
    return torch.stack(labels, -1), torch.stack(masks, -1)


def _satnet_batch(bank, index, device):
    """The storm network's outputs for these samples, from the bank, or None."""
    stored = bank.get("satnet")
    if stored is None:
        return None
    return {key: value[index].to(device) for key, value in stored.items() if torch.is_tensor(value)}


def _bank_loss(model, bank, index, device, augment: bool = False):
    from .losses import total_loss

    batch = {key: value[index].to(device) for key, value in bank["targets"].items()}
    batch["pooled_far"] = bank["features_far"][index].to(device)
    if getattr(model, "ri2_head", None) is not None:
        batch["ri2_target"], batch["ri2_mask"] = ri2_targets(batch, model.config.lead_times_hours)
    satellite = _satellite_batch(bank, index, device, augment)
    satnet = _satnet_batch(bank, index, device)
    outputs = model.storm_heads(bank["features"][index].to(device), batch["storm_center"], batch["storm_state"],
                                batch["storm_environment"], batch["track_prior"], batch["pooled_far"], satellite,
                                satnet, batch.get("storm_physics"))
    loss, parts = total_loss(outputs, batch, SURFACE_FIELDS)
    if satellite is not None and bool(satellite[1][:, 0].any()):
        batch["storm_satellite"] = satellite
    if satnet is not None:
        batch["storm_satnet"] = satnet
        if getattr(model, "ri2_head", None) is not None:
            parts.setdefault("ri2", parts.get("ri2", 0.0))
    return loss, parts, batch


def train_storm_heads(
    model,
    train_bank: dict,
    val_bank: dict | None = None,
    epochs: int = 300,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.05,
    patience: int = 25,
    checkpoint_dir: str | None = None,
    hub_repo: str | None = None,
    log_every: int = 10,
    satellite_learning_rate: float | None = None,
    augment: bool = False,
    keep_if_no_better: bool = False,
) -> list[dict]:
    """
    Fit the storm heads on a feature bank, early-stopping on held-out seasons.

    The backbone is frozen (:meth:`NatureV1.freeze_backbone`), so only the heads -- about 0.12M
    parameters against 22k storm points -- ever see a best-track label. The weights from the epoch
    with the best held-out loss are the ones kept, not the last ones.

    If the banks carry ``"satellite"`` (see :func:`naturev1.stormsat.satellite_bank`), the infrared
    branch trains too, at ``satellite_learning_rate``. With ``keep_if_no_better`` the held-out loss of
    the model as it arrives is the score to beat, so fine-tuning can never leave it worse.

    Returns:
        One entry per epoch: training and held-out loss.
    """
    from .checkpoint import CheckpointManager, TrainingState

    device = next(model.parameters()).device
    model.freeze_backbone(True)
    heads = list(model.storm_head_parameters())
    satellite = {id(p) for p in model.storm_satellite.parameters()}
    groups = [{"params": [p for p in heads if id(p) not in satellite], "lr": learning_rate}]
    if satellite_learning_rate is not None:
        groups.append({"params": [p for p in heads if id(p) in satellite], "lr": satellite_learning_rate})
    else:
        groups[0]["params"] = heads
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    # Calibration is fitted on the finished heads; training on calibrated logits would fit the wrong thing.
    model.landfall_calibration[:] = torch.tensor([1.0, 0.0])
    model.ri_calibration[:] = torch.tensor([1.0, 0.0])
    count = train_bank["features"].shape[0]
    best, best_state, since, history = math.inf, None, 0, []

    def held_out() -> float:
        model.eval()
        with torch.no_grad():
            total, size = 0.0, val_bank["features"].shape[0]
            for start in range(0, size, 1024):
                index = torch.arange(start, min(start + 1024, size))
                total += float(_bank_loss(model, val_bank, index, device)[0]) * len(index)
        return total / max(size, 1)

    if keep_if_no_better and val_bank is not None:
        best = held_out()
        best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        print(f"[storms] held-out loss before fine-tuning {best:.4f} -- the score to beat", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(count)
        running = 0.0
        for start in range(0, count, batch_size):
            index = order[start : start + batch_size]
            loss, parts, batch = _bank_loss(model, train_bank, index, device, augment=augment)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads, 1.0)
            optimizer.step()
            model.mark_trained(parts, {key: batch.get(key) for key in ("storm_center", "storm_state",
                                                                      "storm_environment", "pooled_far",
                                                                      "track_prior", "storm_satellite",
                                                                      "storm_satnet", "storm_physics")})
            running += float(loss.detach()) * len(index)
        entry = {"epoch": epoch, "train": running / count}
        beat(f"storm heads epoch {epoch}")

        if val_bank is not None:
            entry["val"] = held_out()
            if entry["val"] < best - 1e-4:
                best, since = entry["val"], 0
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            else:
                since += 1
        history.append(entry)
        if epoch % log_every == 0 or epoch == 1:
            line = f"[storms] epoch {epoch:>4}  train {entry['train']:.4f}"
            if "val" in entry:
                line += f"  held-out {entry['val']:.4f}  gap {entry['val'] - entry['train']:+.4f}"
            print(line, flush=True)
        if val_bank is not None and since >= patience:
            print(f"[storms] held-out loss flat for {patience} epochs; best {best:.4f}", flush=True)
            break

    if best_state is not None:
        steps = model.head_steps.clone()
        model.load_state_dict(best_state)
        model.head_steps.copy_(steps)
        if keep_if_no_better and history and min((h.get("val", math.inf) for h in history)) >= best:
            print("[storms] fine-tuning never beat the model it started from on held-out storms; "
                  "that model is kept unchanged", flush=True)
    if val_bank is not None:
        # The consensus weights are fitted on storms the heads never trained on; the scorecard then
        # judges them season by season on storms the *weights* were not fitted on either.
        weights = fit_track_consensus(*_consensus_inputs(model, val_bank, model.config.lead_times_hours))
        model.track_consensus.copy_(weights.to(model.track_consensus.device))
        print("[storms] track consensus weights fitted on held-out storms: "
              + describe_consensus(model.track_consensus, model.config.lead_times_hours), flush=True)
    if checkpoint_dir:
        manager = CheckpointManager(checkpoint_dir, repo_id=hub_repo)
        manager.save(model, None, state=TrainingState(step=len(history), epoch=len(history),
                                                      best_loss=best), config=model.config.to_dict(),
                     force=True)
    return history


def _great_circle_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (torch.deg2rad(x) for x in (lat1, lon1, lat2, lon2))
    inner = (torch.sin((lat2 - lat1) / 2) ** 2
             + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * torch.asin(inner.clamp(0, 1).sqrt())


# --------------------------------------------------------------------------------------------------
# A multi-model consensus, within one model
# --------------------------------------------------------------------------------------------------

@torch.no_grad()
def _head_outputs(model, bank: dict, chunk: int = 4096) -> dict[str, torch.Tensor]:
    """
    Every storm output for every sample in a bank, on the CPU.

    When the model uses a rotation ensemble (``config.storm_rotations``) and the bank kept the rotated
    pools, the outputs are the ensemble's (:func:`naturev1.model.merge_rotations`) -- the same thing a
    forecast computes, so every fit made on a bank matches what it is applied to.
    """
    from .model import merge_rotations

    device = next(model.parameters()).device
    model.eval()
    outputs, size = [], bank["features"].shape[0]
    extra = 0
    if "features_tta" in bank:
        extra = min(int(getattr(model.config, "storm_rotations", 1)) - 1, int(bank["features_tta"].shape[1]))
    for start in range(0, size, chunk):
        index = torch.arange(start, min(start + chunk, size))
        batch = {key: value[index].to(device) for key, value in bank["targets"].items()}
        prior = batch["track_prior"] if model.uses_track_prior() else None
        rollout = {key: batch[key] for key in ("rollout_wind_kt", "rollout_pressure_hpa") if key in batch}
        satellite, satnet = _satellite_batch(bank, index, device), _satnet_batch(bank, index, device)
        pools = [(bank["features"][index], bank["features_far"][index])]
        pools += [(bank["features_tta"][index, r], bank["features_far_tta"][index, r]) for r in range(max(extra, 0))]
        members = [model.storm_heads(near.to(device), batch["storm_center"], batch["storm_state"],
                                     batch["storm_environment"], prior, far.to(device), satellite, satnet,
                                     batch.get("storm_physics"), rollout or None) for near, far in pools]
        out = merge_rotations(members)
        outputs.append({k: v.float().cpu() for k, v in out.items()})
    return {key: torch.cat([o[key] for o in outputs]) for key in outputs[0]}


def track_members(outputs: dict, targets: dict, lead_times_hours) -> torch.Tensor:
    """
    The consensus members for every sample: ``(N, L, len(TRACK_MEMBERS), 2)`` displacements in degrees.

    The learned head's most likely scenario and its probability-weighted mean, the steering-flow track
    (from the bank, so it is there whether or not the head was trained on it), extrapolation, and -- when
    the bank carries them (:func:`naturev1.tracker.rollout_member_tracks`) -- the storm steered through
    the model's own rollout and the vortex followed in it. A member that was not computed is NaN, and the
    consensus is blended over the members each sample has.
    """
    size = outputs["displacement"].shape[0]
    probabilities = outputs["mode_logits"].softmax(-1)
    top = outputs["displacement"][torch.arange(size), probabilities.argmax(-1)]
    mean = (outputs["displacement"] * probabilities.view(size, -1, 1, 1)).sum(1)
    steering = targets["track_prior"].float()
    extrapolation = extrapolation_track(targets["storm_state"], lead_times_hours)
    missing = torch.full_like(steering, float("nan"))
    rolled = targets.get("rollout_steering", missing).float()
    vortex = targets.get("rollout_vortex", missing).float()
    analog = targets.get("analog_track", missing).float()
    ensemble = targets.get("rollout_ensemble", missing).float()
    return torch.stack([top, mean, steering, extrapolation, rolled, vortex, analog, ensemble], 2)


def _consensus_inputs(model, bank: dict, leads):
    outputs = _head_outputs(model, bank)
    targets = bank["targets"]
    return (track_members(outputs, targets, leads), targets["track_target"].float(),
            targets["track_valid"] > 0, targets["storm_center"][:, 0].float())


def _project_to_simplex(weights: torch.Tensor) -> torch.Tensor:
    """Nearest point on the probability simplex (Duchi et al. 2008)."""
    ordered, _ = torch.sort(weights, descending=True)
    cumulative = ordered.cumsum(0) - 1.0
    ranks = torch.arange(1, weights.numel() + 1, dtype=weights.dtype, device=weights.device)
    support = (ordered - cumulative / ranks > 0).nonzero().max()
    threshold = cumulative[support] / (support + 1).to(weights.dtype)
    return (weights - threshold).clamp_min(0.0)


def fit_track_consensus(members: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor,
                        latitude: torch.Tensor, iterations: int = 500) -> torch.Tensor:
    """
    Per-lead weights over the members that minimise the mean squared track error, in kilometres.

    Weights are non-negative and sum to one, which is what makes the result a blend of forecasts rather
    than an arbitrary regression that could extrapolate wildly from two members that happen to be
    correlated on this sample. Solved by projected gradient descent: a few hundred steps on a problem
    with four unknowns per lead.

    Args:
        members: ``(N, L, M, 2)`` degrees. truth: ``(N, L, 2)`` degrees. valid: ``(N, L)`` bool.
        latitude: ``(N,)`` the origin latitude, which sets how long a degree of longitude is.

    Returns:
        ``(L, M)`` weights. A lead with fewer verifying samples than members gets equal weights.
    """
    count, leads, size, _ = members.shape
    members_km = displacement_km(members.double(), latitude.double().view(count, 1, 1))
    truth_km = displacement_km(truth.double(), latitude.double().view(count, 1))
    finite = torch.isfinite(members).all(-1)                                         # (N, L, M)
    weights = torch.zeros((leads, size), dtype=torch.float64)
    for lead in range(leads):
        ok = valid[:, lead]
        # Members computed for (nearly) every verifying sample take part; the rest get no weight here.
        present = [m for m in range(size) if ok.any() and float(finite[ok, lead, m].float().mean()) >= 0.9]
        if not present:
            continue
        rows = ok & finite[:, lead, present].all(-1)
        k = len(present)
        weights[lead, present] = 1.0 / k
        if int(rows.sum()) < k:
            continue
        design = members_km[rows, lead][:, present].permute(0, 2, 1).reshape(-1, k)   # (2n, k)
        target = truth_km[rows, lead].reshape(-1)                                     # (2n,)
        gram = design.T @ design / design.shape[0]
        moment = design.T @ target / design.shape[0]
        step = 1.0 / float(torch.linalg.eigvalsh(gram).max().clamp_min(1e-9))
        w = weights[lead, present].clone()
        for _ in range(iterations):
            w = _project_to_simplex(w - step * (gram @ w - moment))
        weights[lead, present] = w
    return weights.float()


def describe_consensus(weights: torch.Tensor, lead_times_hours) -> str:
    rows = []
    for lead, hours in enumerate(lead_times_hours):
        parts = ", ".join(f"{name} {100 * float(w):.0f}%" for name, w in zip(TRACK_MEMBERS, weights[lead]))
        rows.append(f"+{hours}h: {parts}")
    return "; ".join(rows)


def consensus_track(members: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Blend ``(N, L, M, 2)`` members with ``(L, M)`` weights into ``(N, L, 2)``.

    Over the members each sample actually has: a member that is NaN for a sample drops out and the rest
    are renormalised. If every weighted member is missing, the plain mean of what is there is used.
    """
    finite = torch.isfinite(members).all(-1)                                          # (N, L, M)
    w = weights.to(members.dtype).view(1, *weights.shape) * finite.to(members.dtype)
    total = w.sum(-1, keepdim=True)
    filled = torch.nan_to_num(members)
    blended = (filled * w.unsqueeze(-1)).sum(2) / total.clamp_min(1e-12)
    count = finite.sum(-1, keepdim=True).to(members.dtype)
    fallback = (filled * finite.unsqueeze(-1).to(members.dtype)).sum(2) / count.clamp_min(1.0)
    return torch.where(total > 0, blended, fallback)


def _cross_validated_consensus(members, truth, valid, latitude, seasons) -> torch.Tensor:
    """
    Each season's consensus from weights fitted on the *other* seasons -- so the consensus is scored on
    storms its weights never saw. With a single season there is nothing to hold out, and the weights
    are fitted on all of it; the scorecard says so.
    """
    blended = torch.zeros_like(truth)
    distinct = sorted(set(seasons.tolist()))
    if len(distinct) < 2:
        return consensus_track(members, fit_track_consensus(members, truth, valid, latitude))
    for season in distinct:
        held = seasons == season
        weights = fit_track_consensus(members[~held], truth[~held], valid[~held], latitude[~held])
        blended[held] = consensus_track(members[held], weights)
    return blended


@torch.no_grad()
def storm_scorecard(model, bank: dict, lead_times_hours) -> str:
    """
    Track and intensity errors on held-out storms, member by member, against the forecaster's baselines.

    Track: great-circle error in km of the learned head's most likely scenario, of the steering-flow
    track (no training at all -- the atmosphere's own deep-layer wind), of the consensus (weights
    fitted on other seasons), of a storm that stays put, and of straight extrapolation of the last twelve
    hours -- the genuinely hard baseline at 12-24 h. Intensity: mean absolute error in knots, against
    persistence.
    """
    out = _head_outputs(model, bank)
    targets = bank["targets"]
    size = out["displacement"].shape[0]
    centre = targets["storm_center"].float()
    truth, valid = targets["track_target"].float(), targets["track_valid"] > 0
    members = track_members(out, targets, lead_times_hours)
    seasons = targets.get("season", torch.zeros(size, dtype=torch.int64))
    blended = _cross_validated_consensus(members, truth, valid, centre[:, 0], seasons)
    wind_now = targets["current_wind_kt"]
    distinct = len(set(seasons.tolist()))
    # The rollout members, where the bank carries them: steered through the model's own forecast, and the
    # vortex followed in it. Scored over the samples that have them.
    extra = [(index, label) for index, label in ((4, "roll-steer"), (5, "vortex"))
             if bool(torch.isfinite(members[:, :, index]).any())]

    lines = [f"{'lead':>6} {'n':>6} {'model km':>9} {'steering':>9} {'consensus':>9} {'stay-put':>9} "
             f"{'extrap.':>9}" + "".join(f" {label:>10}" for _, label in extra) + f" {'wind MAE kt':>12} {'persist':>8}"]
    for lead, hours in enumerate(lead_times_hours):
        ok = valid[:, lead]
        if not ok.any():
            continue
        lat0, lon0 = centre[ok, 0], centre[ok, 1]
        actual = (lat0 + truth[ok, lead, 0], lon0 + truth[ok, lead, 1])

        def error(d):
            return float(_great_circle_km(actual[0], actual[1], lat0 + d[:, 0], lon0 + d[:, 1]).mean())

        scenario = error(members[ok, lead, 0])
        steer = error(members[ok, lead, 2])
        consensus = error(blended[ok, lead])
        still = error(torch.zeros_like(truth[ok, lead]))
        extrapolated = error(members[ok, lead, 3])
        observed = targets["intensity_mask"][:, lead, 0] > 0
        both = ok & observed & torch.isfinite(wind_now)
        wind_true = targets["intensity_target"][both, lead, 0] / KT_TO_MS
        wind_pred = out["intensity_mean"][both, lead, 0] / KT_TO_MS
        mae = float((wind_pred - wind_true).abs().mean()) if both.any() else float("nan")
        persist = float((wind_now[both] - wind_true).abs().mean()) if both.any() else float("nan")
        columns = ""
        for index, _ in extra:
            have = torch.isfinite(members[ok, lead, index]).all(-1)
            if have.any():
                d = members[ok, lead, index][have]
                km = float(_great_circle_km(actual[0][have], actual[1][have], lat0[have] + d[:, 0],
                                            lon0[have] + d[:, 1]).mean())
                columns += f" {km:>10.0f}"
            else:
                columns += f" {'-':>10}"
        lines.append(f"{hours:>5}h {int(ok.sum()):>6} {scenario:>9.0f} {steer:>9.0f} {consensus:>9.0f} "
                     f"{still:>9.0f} {extrapolated:>9.0f}{columns} {mae:>12.1f} {persist:>8.1f}")
    lines.append("")
    lines.append("'steering' is physics alone: the deep-layer wind around the storm, no training. 'consensus' blends")
    lines.append("model, steering and extrapolation with weights fitted on "
                 + ("the other held-out seasons" if distinct > 1
                    else "these same storms (one season: nothing to hold out)") + ".")
    lines.append("Beating 'extrap.' on track and 'persist' on wind is the bar; beating 'steering' is what the")
    lines.append("learned part is for. Losing to them means the heads have not learned anything the storm's own")
    lines.append("recent history, or the flow around it, did not already say.")
    if float(model.track_consensus.abs().sum()) > 0:
        lines.append("consensus weights in the checkpoint: "
                     + describe_consensus(model.track_consensus, lead_times_hours))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# Calibration and the input monitor, fitted after the heads are trained
# --------------------------------------------------------------------------------------------------

def _platt(logits: torch.Tensor, outcomes: torch.Tensor, steps: int = 200) -> tuple[float, float]:
    """``(a, b)`` minimising the log loss of ``sigmoid(a * logit + b)``; ``a`` kept positive (ranking kept)."""
    with torch.enable_grad():                       # callers may be inside no_grad (scorecards)
        log_a = torch.zeros((), dtype=torch.float64, requires_grad=True)
        b = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_a, b], lr=0.5, max_iter=steps)
        logits, outcomes = logits.double().detach().cpu(), outcomes.double().detach().cpu()

        def closure():
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(log_a.exp() * logits + b, outcomes)
            loss.backward()
            return loss

        optimizer.step(closure)
    return float(log_a.detach().exp()), float(b.detach())


def _raw_logits(model, bank, with_ri2: bool = False):
    """Landfall and RI logits (and RI 2.0's, if asked) with the calibration switched off."""
    stores = [model.landfall_calibration, model.ri_calibration]
    if hasattr(model, "ri2_calibration"):
        stores.append(model.ri2_calibration)
    saved = [store.clone() for store in stores]
    for store in stores:
        store[:] = torch.tensor([1.0, 0.0])
    try:
        out = _head_outputs(model, bank)
    finally:
        for store, value in zip(stores, saved):
            store.copy_(value)
    if with_ri2:
        return out["landfall_logit"], out["ri_logits"], out.get("ri2_logits")
    return out["landfall_logit"], out["ri_logits"]


def fit_probability_calibration(model, bank: dict, min_events: int = 5) -> str:
    """
    Fit the landfall and rapid-intensification calibration on held-out storms; returns a summary.

    The red team found the RI head said 21% when 1% happened, and landfall 39% when 31% did. Both rank
    storms well (AUC 0.82-0.86), so a monotone rescaling of each logit -- Platt scaling -- fixes the
    probabilities without touching the ranking. A column with fewer than ``min_events`` outcomes keeps
    no calibration rather than one fitted to noise.
    """
    landfall, ri, ri2 = _raw_logits(model, bank, with_ri2=True)
    t = bank["targets"]
    lines = []
    columns = ([(landfall[:, l], t["landfall_target"][:, l], t["landfall_mask"][:, l], f"landfall +{h}h",
                 model.landfall_calibration[l]) for l, h in enumerate(model.config.lead_times_hours)]
               + [(ri[:, j], t["ri_target"][:, j], t["ri_mask"][:, j], f"RI threshold {j}", model.ri_calibration[j])
                  for j in range(ri.shape[1])])
    if ri2 is not None:
        from .satnet import ri_threshold_list

        target2, mask2 = ri2_targets(t, model.config.lead_times_hours)
        columns += [(ri2[:, j], target2[:, j], mask2[:, j], f"RI2 +{kt:.0f} kt/{hours} h", model.ri2_calibration[j])
                    for j, (hours, kt) in enumerate(ri_threshold_list())]
    for column, (logits, target, mask, label, store) in enumerate(columns):
        keep = mask > 0
        y = target[keep].float()
        if int(y.sum()) < min_events or int((1 - y).sum()) < min_events:
            store.copy_(torch.tensor([1.0, 0.0]))
            continue
        a, b = _platt(logits[keep], y)
        store.copy_(torch.tensor([a, b]))
        before = float(((torch.sigmoid(logits[keep]) - y) ** 2).mean())
        after = float(((torch.sigmoid(a * logits[keep] + b) - y) ** 2).mean())
        lines.append(f"{label}: Brier {before:.4f} -> {after:.4f}")
    return "; ".join(lines)


def calibration_report(model, bank: dict) -> str:
    """
    Reliability of landfall-by-72h and RI +30 kt, with the calibration fitted leave-one-season-out --
    each season's probabilities come from a calibration that never saw that season.
    """
    landfall, ri = _raw_logits(model, bank)
    t = bank["targets"]
    seasons = t.get("season", torch.zeros(landfall.shape[0], dtype=torch.int64))
    lines = []
    lead = list(model.config.lead_times_hours).index(72) if 72 in model.config.lead_times_hours else -1
    for label, logits, target, mask in (("landfall by 72 h", landfall[:, lead], t["landfall_target"][:, lead],
                                         t["landfall_mask"][:, lead]),
                                        ("RI +30 kt / 24 h", ri[:, 1], t["ri_target"][:, 1], t["ri_mask"][:, 1])):
        keep = mask > 0
        p_raw = torch.sigmoid(logits)
        p_cal = p_raw.clone()
        for season in seasons[keep].unique().tolist():
            fit = keep & (seasons != season)
            apply = keep & (seasons == season)
            y = target[fit].float()
            if int(y.sum()) < 5 or int((1 - y).sum()) < 5:
                continue
            a, b = _platt(logits[fit], y)
            p_cal[apply] = torch.sigmoid(a * logits[apply] + b)
        y = target[keep].float()
        base = float(((y.mean() - y) ** 2).mean()) if y.numel() else 0.0
        if base == 0.0:
            lines.append(f"{label}: nothing to score -- every held-out sample has the same outcome")
            continue
        raw, cal = float(((p_raw[keep] - y) ** 2).mean()), float(((p_cal[keep] - y) ** 2).mean())
        bins = []
        for low, high in ((0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)):
            sel = (p_cal[keep] >= low) & (p_cal[keep] < high)
            if int(sel.sum()) >= 5:
                bins.append(f"said {float(p_cal[keep][sel].mean()):.0%} got {float(y[sel].mean()):.0%}")
        lines.append(f"{label}: Brier skill vs base rate {1 - raw / base:+.0%} raw, {1 - cal / base:+.0%} calibrated"
                     f" (leave-one-season-out); " + ", ".join(bins))
    return "\n".join(lines)


@torch.no_grad()
def fit_feature_monitor(model, bank: dict, quantile: float = 0.999, margin: float = 1.5) -> float:
    """
    Learn what the pooled storm features look like in training, so unlike inputs can be flagged.

    The red team fed random noise in place of the backbone features and got a normal-looking forecast
    that passed every check. This stores the per-dimension mean and spread of the training features; a
    forecast whose mean squared z-score exceeds ``margin`` times the training ``quantile`` is marked
    untrustworthy by :func:`naturev1.build_forecast`. Returns the threshold.
    """
    features = bank["features"].double()
    mean, std = features.mean(0), features.std(0).clamp_min(1e-6)
    scores = ((features - mean) / std).pow(2).mean(-1)
    threshold = float(scores.quantile(quantile)) * margin
    norms = features.norm(dim=-1)
    model.feature_mean.copy_(mean.to(model.feature_mean.dtype))
    model.feature_std.copy_(std.to(model.feature_std.dtype))
    model.feature_threshold.fill_(threshold)
    model.feature_norm_bounds.copy_(torch.tensor([float(norms.quantile(1 - quantile)) / margin,
                                                  float(norms.quantile(quantile)) * margin]))
    return threshold


@torch.no_grad()
def fit_intensity_correction(model, bank: dict, ridge: float = 50.0) -> str:
    """
    Fit the per-lead intensity bias correction on held-out storms; returns before/after by intensity class.

    The red team measured the heads 10-18 kt too weak for major hurricanes -- regression to the mean,
    the classic failure of a likelihood-trained intensity model. A correction linear in the predicted wind
    (an offset, and a slope around 65 kt) is fitted per lead by ridge-regularised least squares on
    storms the heads never trained on, for the intensity head and the eyewall peak-wind head separately.
    Zero is no correction, and a lead with too few verifying storms keeps zero.
    """
    model.intensity_correction.zero_()
    model.eyewall_correction.zero_()
    out = _head_outputs(model, bank)
    t = bank["targets"]
    lines = []
    for name, predicted, truth, mask, store in (
            ("intensity", out["intensity_mean"][..., 0] / KT_TO_MS, t["intensity_target"][..., 0] / KT_TO_MS,
             t["intensity_mask"][..., 0] > 0, model.intensity_correction),
            ("eyewall", out["eyewall_peak_wind_kt"], t["eyewall_target"],
             torch.isfinite(t["eyewall_target"]), model.eyewall_correction)):
        for lead, hours in enumerate(model.config.lead_times_hours):
            ok = mask[:, lead] & torch.isfinite(truth[:, lead])
            if int(ok.sum()) < 50:
                continue
            x = (predicted[ok, lead].double() - 65.0) / 30.0
            y = truth[ok, lead].double() - predicted[ok, lead].double()
            design = torch.stack([torch.ones_like(x), x], -1)
            gram = design.T @ design + torch.diag(torch.tensor([0.0, ridge], dtype=torch.float64))
            coef = torch.linalg.solve(gram, design.T @ y)
            store[lead] = coef.to(store.dtype)
        lines.append(f"{name}: " + ", ".join(f"+{h}h {float(store[l, 0]):+.1f} kt / {float(store[l, 1]):+.1f}"
                                             for l, h in enumerate(model.config.lead_times_hours)))
    return "\n".join(lines) + "\n" + intensity_by_category(model, bank)


@torch.no_grad()
def intensity_by_category(model, bank: dict, leads_hours=(24, 48, 72)) -> str:
    """Wind bias and MAE by the storm's *verifying* intensity class -- where regression to the mean shows."""
    out = _head_outputs(model, bank)
    t = bank["targets"]
    lines = [f"{'lead':>6} {'class':>8} {'n':>6} {'bias kt':>8} {'MAE kt':>7}"]
    leads = list(model.config.lead_times_hours)
    for hours in leads_hours:
        if hours not in leads:
            continue
        lead = leads.index(hours)
        truth = t["intensity_target"][:, lead, 0] / KT_TO_MS
        predicted = out["intensity_mean"][:, lead, 0] / KT_TO_MS
        ok = t["intensity_mask"][:, lead, 0] > 0
        for label, low, high in (("TS", 34, 64), ("Cat 1-2", 64, 96), ("Cat 3-5", 96, 400)):
            band = ok & (truth >= low) & (truth < high)
            if band.any():
                error = predicted[band] - truth[band]
                lines.append(f"{hours:>5}h {label:>8} {int(band.sum()):>6} {float(error.mean()):>+8.1f} "
                             f"{float(error.abs().mean()):>7.1f}")
    return "\n".join(lines)


@torch.no_grad()
def ri2_report(model, bank: dict) -> str:
    """
    RI 2.0 on held-out storms, per threshold: base rate, Brier score and Brier skill against climatology,
    and how many events the top-decile probabilities catch (the share of real RI a forecaster's watch
    list would have contained).
    """
    from .satnet import ri_threshold_list

    out = _head_outputs(model, bank)
    if "ri2_logits" not in out:
        return "RI 2.0: not in this model"
    target, mask = ri2_targets(bank["targets"], model.config.lead_times_hours)
    probability = torch.sigmoid(out["ri2_logits"])
    lines = [f"{'threshold':>14} {'n':>6} {'base %':>7} {'Brier':>7} {'skill':>7} {'caught by top 10%':>18}"]
    for j, (hours, kt) in enumerate(ri_threshold_list()):
        keep = mask[:, j] > 0
        if int(keep.sum()) < 20:
            continue
        y, p = target[keep, j], probability[keep, j]
        base = float(y.mean())
        brier = float(((p - y) ** 2).mean())
        skill = 1.0 - brier / max(base * (1 - base), 1e-9)
        cut = torch.quantile(p, 0.9)
        caught = float((y * (p >= cut)).sum() / y.sum().clamp_min(1.0))
        lines.append(f"{f'+{kt:.0f} kt / {hours} h':>14} {int(keep.sum()):>6} {100 * base:>6.2f}% {brier:>7.4f} "
                     f"{skill:>+7.3f} {100 * caught:>17.0f}%")
    return "\n".join(lines)


@torch.no_grad()
def consensus_scorecard(model, bank: dict) -> str:
    """
    Every track and intensity member against the truth, the plain and the corrected consensus, and how well the
    predicted-error cone covers -- on whatever storms ``bank`` holds (the test seasons, in stage B).
    """
    from .statistical import (
        INTENSITY_MEMBERS, bank_intensity_members, consensus_radius_km, corrected_track, member_spread_km,
    )
    from .steering import displacement_km

    leads = model.config.lead_times_hours
    out = _head_outputs(model, bank)
    t = bank["targets"]
    centre = t["storm_center"].float()
    truth, valid = t["track_target"].float(), t["track_valid"] > 0
    members = track_members(out, t, leads)
    plain = consensus_track(members, model.track_consensus.cpu().float())
    corrected = corrected_track(model, members, centre[:, 0])
    spread = member_spread_km(members, corrected, centre[:, 0])
    radius = consensus_radius_km(model, spread)

    def km(d):
        return displacement_km((truth - d).double(), centre[:, 0].double().view(-1, 1)).norm(dim=-1)

    columns = [name for i, name in enumerate(TRACK_MEMBERS) if bool(torch.isfinite(members[:, :, i]).any())]
    lines = ["TRACK (km): " + " ".join(f"{c[:10]:>10}" for c in columns) + f" {'consensus':>10} {'corrected':>10} {'cone hit':>9}"]
    for lead, hours in enumerate(leads):
        ok = valid[:, lead]
        if not ok.any():
            continue
        row = []
        for name in columns:
            index = TRACK_MEMBERS.index(name)
            have = ok & torch.isfinite(members[:, lead, index]).all(-1)
            row.append(f"{float(km(members[:, :, index])[have, lead].mean()):>10.0f}" if have.any() else f"{'-':>10}")
        error = km(corrected)[ok, lead]
        hit = radius[ok, lead]
        cover = f"{100 * float((error <= hit).float().mean()):>8.0f}%" if bool(torch.isfinite(hit).all()) else f"{'-':>9}"
        lines.append(f"+{hours:>3}h      " + " ".join(row) + f" {float(km(plain)[ok, lead].mean()):>10.0f} "
                     f"{float(error.mean()):>10.0f} {cover}")
    intensity = bank_intensity_members(model, bank)
    blended = model.statistics.consensus(intensity)
    wind_truth = t["intensity_target"][..., 0] / KT_TO_MS
    wind_valid = t["intensity_mask"][..., 0] > 0
    present = [i for i, name in enumerate(INTENSITY_MEMBERS) if bool(torch.isfinite(intensity[..., i]).any())]
    lines.append("")
    lines.append("INTENSITY (MAE kt): " + " ".join(f"{INTENSITY_MEMBERS[i][:10]:>10}" for i in present)
                 + f" {'consensus':>10} {'persist':>8}")
    for lead, hours in enumerate(leads):
        ok = wind_valid[:, lead] & torch.isfinite(t["current_wind_kt"])
        if not ok.any():
            continue
        row = []
        for i in present:
            have = ok & torch.isfinite(intensity[:, lead, i])
            row.append(f"{float((intensity[have, lead, i] - wind_truth[have, lead]).abs().mean()):>10.1f}" if have.any()
                       else f"{'-':>10}")
        consensus = float((blended[ok, lead] - wind_truth[ok, lead]).abs().mean())
        persist = float((t["current_wind_kt"][ok] - wind_truth[ok, lead]).abs().mean())
        lines.append(f"+{hours:>3}h      " + " ".join(row) + f" {consensus:>10.1f} {persist:>8.1f}")
    return "\n".join(lines)
