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
from .model import (
    PEAK_WIND_ANCHOR_KT,
    PRESSURE_ANCHOR_HPA,
    RI_THRESHOLDS_KT,
    SURFACE_FIELDS,
    WIND_RADII_THRESHOLDS_KT,
)


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

            # Landfall is cumulative: "will it make landfall by +48h", not "exactly at +48h".
            landfall[lead] = float(track.landfall[start : end + 1].any())
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
    precision: str = "bf16",
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
        ``{"features": (N, hidden), "targets": {...}, "fingerprint": str}``.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .era5 import read_block, split_prepared
    from .model import calendar_features
    from .state import _autocast

    device = next(model.parameters()).device
    fingerprint = _backbone_fingerprint(model)
    count, hidden = len(starts), model.config.hidden_size
    features = torch.zeros(count, hidden)
    done: set[str] = set()
    partial = Path(f"{cache}.partial") if cache else None
    if cache and Path(cache).exists():
        stored = torch.load(cache, map_location="cpu", weights_only=False)
        if stored.get("fingerprint") == fingerprint and stored["features"].shape[0] == count:
            if progress:
                print(f"[storms] feature bank reused: {cache} ({count:,} samples)", flush=True)
            return stored
    if partial is not None and partial.exists():
        stored = torch.load(partial, map_location="cpu", weights_only=False)
        if stored.get("fingerprint") == fingerprint and stored["features"].shape[0] == count:
            features, done = stored["features"], set(stored["done"])

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
        raw = read_block(store, base.source_variables, slice(first, last), base.levels)
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
                    pooled = model.storm_features(analysis, grid, calendar_features(stamps).to(device), centres)
                features[chosen] = pooled.float().cpu()
            done.add(storm)
            if progress:
                rate = position / max(time.time() - started, 1e-9)
                left = (len(pending) - position) / max(rate, 1e-9)
                print(f"\r[storms] features {len(done):,}/{len(groups):,} storms "
                      f"({left / 60:.0f} min left)", end="", flush=True)
            if partial is not None and time.time() - last_save > 60:
                torch.save({"features": features, "done": sorted(done), "fingerprint": fingerprint}, partial)
                last_save = time.time()
    if progress and pending:
        print(flush=True)

    bank = {"features": features, "targets": _stack_targets(targets), "fingerprint": fingerprint}
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(bank, cache)
        if partial is not None:
            partial.unlink(missing_ok=True)
    return bank


def _bank_loss(model, bank, index, device):
    from .losses import total_loss

    batch = {key: value[index].to(device) for key, value in bank["targets"].items()}
    outputs = model.storm_heads(bank["features"][index].to(device), batch["storm_center"], batch["storm_state"])
    loss, parts = total_loss(outputs, batch, SURFACE_FIELDS)
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
) -> list[dict]:
    """
    Fit the storm heads on a feature bank, early-stopping on held-out seasons.

    The backbone is frozen (:meth:`NatureV1.freeze_backbone`), so only the heads -- about 0.12M
    parameters against 22k storm points -- ever see a best-track label. The weights from the epoch
    with the best held-out loss are the ones kept, not the last ones.

    Returns:
        One entry per epoch: training and held-out loss.
    """
    from .checkpoint import CheckpointManager, TrainingState

    device = next(model.parameters()).device
    model.freeze_backbone(True)
    heads = list(model.storm_head_parameters())
    optimizer = torch.optim.AdamW(heads, lr=learning_rate, weight_decay=weight_decay)
    count = train_bank["features"].shape[0]
    best, best_state, since, history = math.inf, None, 0, []

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(count)
        running = 0.0
        for start in range(0, count, batch_size):
            index = order[start : start + batch_size]
            loss, parts, batch = _bank_loss(model, train_bank, index, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads, 1.0)
            optimizer.step()
            model.mark_trained(parts, {"storm_center": batch["storm_center"], "storm_state": batch["storm_state"]})
            running += float(loss) * len(index)
        entry = {"epoch": epoch, "train": running / count}

        if val_bank is not None:
            model.eval()
            with torch.no_grad():
                total = 0.0
                size = val_bank["features"].shape[0]
                for start in range(0, size, 4096):
                    index = torch.arange(start, min(start + 4096, size))
                    total += float(_bank_loss(model, val_bank, index, device)[0]) * len(index)
            entry["val"] = total / max(size, 1)
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


@torch.no_grad()
def storm_scorecard(model, bank: dict, lead_times_hours) -> str:
    """
    Track and intensity errors on held-out storms, against the baselines a forecaster would use.

    Track: great-circle error of the most likely scenario, against a storm that stays put and against
    straight-line extrapolation of its last twelve hours of motion -- the second is a genuinely hard
    baseline at 12-24 h. Intensity: mean absolute error in knots, against persistence.
    """
    device = next(model.parameters()).device
    model.eval()
    outputs, size = [], bank["features"].shape[0]
    for start in range(0, size, 4096):
        index = torch.arange(start, min(start + 4096, size))
        batch = {key: value[index].to(device) for key, value in bank["targets"].items()}
        outputs.append({k: v.float().cpu() for k, v in model.storm_heads(
            bank["features"][index].to(device), batch["storm_center"], batch["storm_state"]).items()})
    out = {key: torch.cat([o[key] for o in outputs]) for key in outputs[0]}
    targets = bank["targets"]

    centre = targets["storm_center"]
    top = out["mode_logits"].argmax(-1)
    predicted = out["displacement"][torch.arange(size), top]                          # (N, L, 2)
    truth, valid = targets["track_target"], targets["track_valid"] > 0
    motion = targets["storm_state"][:, 2:4]                                             # degrees / 12 h
    wind_now = targets["current_wind_kt"]

    lines = [f"{'lead':>6} {'n':>6} {'track km':>9} {'stay-put':>9} {'extrap.':>9} "
             f"{'wind MAE kt':>12} {'persist':>8}"]
    for lead, hours in enumerate(lead_times_hours):
        ok = valid[:, lead]
        if not ok.any():
            continue
        lat0, lon0 = centre[ok, 0], centre[ok, 1]
        actual = (lat0 + truth[ok, lead, 0], lon0 + truth[ok, lead, 1])

        def error(d):
            return float(_great_circle_km(actual[0], actual[1], lat0 + d[:, 0], lon0 + d[:, 1]).mean())

        model_err = error(predicted[ok, lead])
        still = error(torch.zeros_like(predicted[ok, lead]))
        extrapolated = error(motion[ok] * hours / 12.0)
        observed = targets["intensity_mask"][:, lead, 0] > 0
        both = ok & observed & torch.isfinite(wind_now)
        wind_true = targets["intensity_target"][both, lead, 0] / KT_TO_MS
        wind_pred = out["intensity_mean"][both, lead, 0] / KT_TO_MS
        mae = float((wind_pred - wind_true).abs().mean()) if both.any() else float("nan")
        persist = float((wind_now[both] - wind_true).abs().mean()) if both.any() else float("nan")
        lines.append(f"{hours:>5}h {int(ok.sum()):>6} {model_err:>9.0f} {still:>9.0f} {extrapolated:>9.0f} "
                     f"{mae:>12.1f} {persist:>8.1f}")
    lines.append("")
    lines.append("Beating 'extrap.' on track and 'persist' on wind is the bar. Losing to them means the heads")
    lines.append("have not learned anything the storm's own recent history did not already say.")
    return "\n".join(lines)
