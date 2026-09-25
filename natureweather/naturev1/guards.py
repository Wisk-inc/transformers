# Copyright 2026 Nathan. Apache-2.0.
"""
Guards: catch the wrong answer before it costs a day, and refuse to dress it up as a forecast.

Everything in this module exists because of a specific failure that already happened, in this codebase,
on real data, and that looked healthy while it was wrong:

* an ERA5 read bound every sample to the wrong place on Earth -- right point count, sane values, falling
  loss, and a temperature/latitude correlation of zero;
* ``cos(latitude)`` weights silently deleted the poles, at 6.1e-17 in float32;
* precipitation z-scored to a 30-sigma spike, so the optimal model predicted zero rain everywhere;
* ``model.to("cuda")`` left every grid on the CPU, surfacing four frames deep as a Linear-layer dtype
  error with no mention of a grid;
* early stopping counted training loss while claiming validation, so a run would halt in minutes and
  report it as convergence;
* an untrained model produced a confident 65 kt hurricane forecast with a tidy uncertainty interval.

The common shape: **nothing raised**. A silent wrong answer costs more than a crash, because a crash
stops you and a wrong answer gets published. So the checks here are loud, they run before the expensive
part, and the last one refuses to emit a storm forecast from a model that has not learned anything --
which is the only failure in the list that could hurt somebody.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch


class PreflightError(RuntimeError):
    """Raised when something would produce a wrong answer rather than an error."""


@dataclass(frozen=True)
class Bounds:
    """A physically possible range, with the observed record that justifies it."""

    low: float
    high: float
    unit: str
    why: str

    def check(self, values: torch.Tensor, name: str) -> list[str]:
        """Report what falls outside, rather than raising -- the caller decides how loud to be."""
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return [f"{name}: every value is NaN or Inf"]
        low, high = float(finite.min()), float(finite.max())
        problems = []
        if low < self.low:
            problems.append(f"{name}: minimum {low:.2f} {self.unit} is below {self.low} ({self.why})")
        if high > self.high:
            problems.append(f"{name}: maximum {high:.2f} {self.unit} is above {self.high} ({self.why})")
        if values.numel() != finite.numel():
            share = 100 * (1 - finite.numel() / values.numel())
            problems.append(f"{name}: {share:.1f}% of values are NaN or Inf")
        return problems


#: Physical limits for every quantity the model reports, set from observed records with headroom.
#: A forecast outside these is not a bold forecast, it is a broken one.
PHYSICAL = {
    "eyewall_peak_wind_kt": Bounds(0.0, 200.0, "kt", "world record sustained wind is ~190 kt"),
    "eyewall_rmw_nmi": Bounds(1.0, 250.0, "nmi", "eyewalls run ~2-100 nmi; 250 is already absurd"),
    "wind_radii_nmi": Bounds(0.0, 600.0, "nmi", "the largest storms reach ~600 nmi gale radius"),
    "ri_delta_wind_kt": Bounds(-120.0, 150.0, "kt/24h", "record intensification is ~100 kt/24h"),
    "intensity_wind_ms": Bounds(0.0, 105.0, "m/s", "190 kt is 98 m/s"),
    "intensity_pressure_hpa": Bounds(850.0, 1100.0, "hPa", "record low 870, record high 1084"),
    "enso_index": Bounds(-4.0, 4.0, "K", "Nino 3.4 has never exceeded about +-3 K"),
    "t2m": Bounds(170.0, 340.0, "K", "record -89.2 C to +56.7 C, with headroom"),
    "mslp": Bounds(85000.0, 110000.0, "Pa", "same records in pascals"),
    "wind10m": Bounds(-150.0, 150.0, "m/s", "component wind; 113 m/s is the record gust"),
    "track_displacement_deg": Bounds(-60.0, 60.0, "deg", "40 kt translation for 120 h is ~26 deg"),
}


def check_environment(require_cuda: bool = False, min_vram_gb: float = 0.0,
                      min_disk_gb: float = 0.0, path: str = ".") -> dict:
    """
    What this machine actually is, and whether the run about to start can fit on it.

    Checked first because the alternative is discovering it four hours in, with the staging done and
    the checkpoint directory on a disk that filled up.
    """
    report = {"cuda": torch.cuda.is_available(), "problems": []}

    if report["cuda"]:
        properties = torch.cuda.get_device_properties(0)
        report["gpu"] = properties.name
        report["vram_gb"] = properties.total_memory / 1024**3
        report["bf16"] = torch.cuda.is_bf16_supported()
        if min_vram_gb and report["vram_gb"] < min_vram_gb:
            report["problems"].append(
                f"{report['vram_gb']:.0f} GB of VRAM, {min_vram_gb:.0f} GB wanted")
        if not report["bf16"]:
            report["problems"].append(
                "bf16 unsupported on this card: set precision='fp16', which needs loss scaling")
    elif require_cuda:
        report["problems"].append("no CUDA device; this run would take weeks on CPU")

    if min_disk_gb:
        usage = os.statvfs(path)
        free = usage.f_bavail * usage.f_frsize / 1e9
        report["free_disk_gb"] = free
        if free < min_disk_gb:
            report["problems"].append(f"{free:.0f} GB free at {path}, {min_disk_gb:.0f} GB needed")

    return report


def check_grid_alignment(grid, values: torch.Tensor, latitudes=None, tolerance: float = -0.5) -> list[str]:
    """
    Are the samples where the grid says they are.

    This is the transposition check, and it is the single highest-value test in the package. A
    latitude-major coordinate mesh read against a longitude-major store gives the right number of
    points, plausible values and a falling loss, with every sample on the wrong continent. Temperature
    against absolute latitude is the cheapest physical probe: strongly negative when correct, about
    zero when transposed.

    Args:
        values: ``(points,)`` a temperature-like field on this grid, in any units.
        tolerance: correlation that must be beaten. -0.5 is loose enough for a single snapshot and
            nowhere near what a transposed read produces.
    """
    if grid.num_points != values.shape[-1]:
        return [f"grid has {grid.num_points} points, field has {values.shape[-1]}"]

    point_lat = np.degrees(grid.coords[:, 0].detach().cpu().numpy())
    field = values.detach().cpu().numpy().ravel()
    finite = np.isfinite(field)
    if finite.sum() < 100:
        return ["too few finite values to check alignment"]

    correlation = float(np.corrcoef(np.abs(point_lat[finite]), field[finite])[0, 1])
    if not np.isfinite(correlation):
        return ["alignment check produced no correlation (constant field?)"]
    if correlation > tolerance:
        return [
            f"temperature vs |latitude| correlates {correlation:+.3f}, expected below {tolerance}. "
            "Samples are probably bound to the wrong coordinates -- check the dimension order of the "
            "read (the store is usually longitude-major; the mesh is latitude-major)."
        ]
    return []


def check_weights(grid, max_ratio: float = 1e6) -> list[str]:
    """
    Do the quadrature weights delete part of the domain.

    ``cos(latitude)`` on an equiangular grid gives the pole row 6.1e-17 -- exactly zero in float32 --
    so every area-weighted quantity silently omits the poles. True cell area gives a ratio around 300.
    Anything past a million means a region is contributing nothing at all.
    """
    weights = grid.weights.detach().cpu()
    problems = []
    if not torch.isfinite(weights).all():
        problems.append("grid weights contain NaN or Inf")
        return problems
    if float(weights.min()) <= 0.0:
        problems.append(f"{int((weights <= 0).sum())} grid weights are zero or negative: those samples "
                        "contribute nothing and are effectively deleted")
    ratio = float(weights.max() / weights.min().clamp_min(1e-30))
    if ratio > max_ratio:
        problems.append(f"weight ratio {ratio:.2e} across the grid (max/min). Above {max_ratio:.0e} the "
                        "smallest cells are numerically absent -- cos(latitude) instead of true cell area?")
    total = float(weights.sum())
    if not 0.5 < total < 2.0:
        problems.append(f"weights sum to {total:.4f}; they should be normalised to about 1")
    return problems


def check_normalization(values: torch.Tensor, name: str = "input", limit: float = 12.0) -> list[str]:
    """
    Is any channel a spike rather than a distribution.

    Precipitation z-scored raw peaks at 30 sigma with 15% exact zeros, and the loss-minimising answer
    to that is "predict zero everywhere, forever". A channel beyond about 12 sigma wants a transform,
    not a rescale.
    """
    problems = []
    if not torch.isfinite(values).all():
        share = 100 * (1 - torch.isfinite(values).float().mean().item())
        problems.append(f"{name}: {share:.1f}% NaN or Inf")
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return problems + [f"{name}: no finite values"]

    extreme = float(finite.abs().max())
    if extreme > limit:
        problems.append(f"{name}: reaches {extreme:.1f} standard deviations. A heavy-tailed channel "
                        "(precipitation) needs log1p before the z-score, not a wider clip.")
    spread = float(finite.std())
    if spread < 1e-3:
        problems.append(f"{name}: standard deviation {spread:.2e} -- this channel is nearly constant "
                        "and is spending bandwidth on nothing")
    return problems


def check_device_agreement(model, *grids) -> list[str]:
    """
    Do the weights and every grid live in the same place.

    Fixed in ihelix 0.4.0, checked here anyway: a caller can still build a grid after moving the model,
    and the resulting error names a Linear layer four frames away from the actual cause.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return ["model has no parameters"]

    problems = []
    for index, grid in enumerate(grids):
        if grid is None:
            continue
        points = getattr(grid, "points", None)
        if points is not None and points.device != device:
            problems.append(f"grid {index} is on {points.device}, model is on {device}. "
                            f"Call grid.to('{device}') or pass it through model.link().")
    return problems


@torch.no_grad()
def responds_to_input(model, make_batch, grid, trials: int = 3, threshold: float = 1e-3) -> tuple[bool, float]:
    """
    Does the model's answer depend on the weather, or is it reciting climatology.

    This is the check that catches an untrained model dressed as a forecaster. Feed genuinely different
    atmospheres and compare the storm outputs: a trained model moves, an untrained one returns its
    initialisation -- 65 kt peak wind and a 4.11% RI probability -- no matter what it is shown, because
    the heads are anchored at climatology and the backbone is noise.

    It also catches a *collapsed* model, which is the more dangerous case: training that diverged and
    settled on predicting the mean has a respectable loss curve and no skill whatsoever.

    Returns:
        ``(responds, relative_spread)`` where the spread is across trials, so 0.0 is a constant.
    """
    was_training = model.training
    model.eval()
    signals: dict[str, list[float]] = {"eyewall": [], "state": []}
    inputs = []
    for trial in range(trials):
        torch.manual_seed(1000 + trial)
        try:
            batch = make_batch()
        except StopIteration:
            break                                 # a small held-out set: compare what there is
        inputs.append(float(batch["analysis"].double().sum()))
        outputs = model(analysis=batch["analysis"], analysis_grid=grid,
                        calendar=batch["calendar"], output_grid=grid)
        signals["eyewall"].append(float(outputs["eyewall_peak_wind_kt"].mean()))
        if "state_delta" in outputs:
            signals["state"].append(float(outputs["state_delta"].abs().mean()))
    if was_training:
        model.train()

    if len(inputs) < 2:
        print("[guard] fewer than two held-out batches: cannot tell whether the model responds to its input")
        return False, 0.0
    if len(set(inputs)) < 2:
        # The check compares outputs across different atmospheres; given the same one three times it
        # reports "does not respond" for any model, trained or not. That is what happened when the
        # probe was `next(iter(val_loader))` on an unshuffled loader: every model was refused.
        raise PreflightError("responds_to_input was given the same batch every trial. Pass a make_batch "
                             "that returns different atmospheres, e.g. next() on one shared iterator.")

    spread = 0.0
    for values in signals.values():
        if len(values) >= 2:
            centre = abs(float(np.mean(values)))
            if centre > 1e-9:
                spread = max(spread, float(np.std(values)) / centre)
    return spread > threshold, spread


def check_forecast(bundle: dict) -> list[str]:
    """
    Is this forecast physically possible.

    Applied to the output of :func:`naturev1.build_forecast`, so nothing implausible reaches a file, a
    dashboard or a person. The point is not that the model is likely to predict a 400 kt hurricane --
    it is that if it ever does, that must be an error and not a headline.
    """
    problems = []
    if bundle.get("trustworthy") is False:
        problems.append(bundle.get("warning", "this forecast was built from an untrained input path"))

    def rows(key):
        # A section that was never trained is a dict saying so, not a list of rows.
        value = bundle.get(key, [])
        return value if isinstance(value, list) else []

    for entry in rows("eyewall"):
        lead = entry["lead_hours"]
        problems += PHYSICAL["eyewall_peak_wind_kt"].check(
            torch.tensor([entry["peak_wind_kt"]]), f"+{lead}h peak wind")
        if "rmw_nmi" in entry:
            problems += PHYSICAL["eyewall_rmw_nmi"].check(
                torch.tensor([entry["rmw_nmi"]]), f"+{lead}h RMW")
        radii = entry.get("wind_radii_nmi", {})
        for threshold, quadrants in radii.items():
            if isinstance(quadrants, dict):
                problems += PHYSICAL["wind_radii_nmi"].check(
                    torch.tensor(list(quadrants.values())), f"+{lead}h {threshold} radii")

    for entry in rows("intensity"):
        lead = entry["lead_hours"]
        problems += PHYSICAL["intensity_wind_ms"].check(
            torch.tensor([entry["max_wind_ms"]]), f"+{lead}h max wind")
        problems += PHYSICAL["intensity_pressure_hpa"].check(
            torch.tensor([entry["min_pressure_hpa"]]), f"+{lead}h pressure")

    ri = bundle.get("rapid_intensification", {})
    for key, value in ri.items():
        if key.startswith("probability_") and isinstance(value, float) and not 0.0 <= value <= 1.0:
            problems.append(f"{key} is {value}, which is not a probability")
    if "expected_change_kt" in ri:
        problems += PHYSICAL["ri_delta_wind_kt"].check(
            torch.tensor([ri["expected_change_kt"]]), "RI 24h change")

    total = sum(s["probability"] for s in rows("track_scenarios"))
    if rows("track_scenarios") and not 0.95 < total < 1.05:
        problems.append(f"track scenario probabilities sum to {total:.3f}, not 1")

    for scenario in rows("track_scenarios"):
        for point in scenario.get("track", []):
            if not -90.0 <= point["latitude"] <= 90.0:
                problems.append(f"track latitude {point['latitude']:.1f} is off the planet")
            if point.get("cone_radius_km_95", 0) > 8000:
                problems.append(f"+{point['lead_hours']}h cone is {point['cone_radius_km_95']:.0f} km, "
                                "wider than an ocean -- the forecast carries no information")

    if "nino34_index" in bundle.get("enso", {}):
        problems += PHYSICAL["enso_index"].check(
            torch.tensor([bundle["enso"]["nino34_index"]]), "Nino 3.4")

    return problems


class TrainingWatchdog:
    """
    Watches a run for the failures that a falling loss curve hides.

    Three of them, all seen in practice:

    * **non-finite loss** -- one NaN poisons every parameter on the next step, and the run continues
      producing NaN forever while still printing step numbers;
    * **divergence** -- loss climbing steadily, usually a learning rate that a warmup masked;
    * **collapse** -- gradient norm falling to nothing, which reads as convergence and is a model that
      has stopped learning and is predicting the mean.

    Call :meth:`observe` once per step. It raises on the first, warns on the others.
    """

    def __init__(self, patience: int = 50, divergence_factor: float = 3.0,
                 collapse_norm: float = 1e-7) -> None:
        self.patience = patience
        self.divergence_factor = divergence_factor
        self.collapse_norm = collapse_norm
        self.best = math.inf
        self.history: list[float] = []
        self.warnings: list[str] = []

    def observe(self, step: int, loss: float, grad_norm: float | None = None) -> None:
        if not math.isfinite(loss):
            raise PreflightError(
                f"step {step}: loss is {loss}. One non-finite step poisons every parameter, and the run "
                "will print step numbers forever while producing nothing. Stop, lower the learning "
                "rate or check for a NaN in the batch, and resume from the last checkpoint."
            )

        self.history.append(loss)
        self.best = min(self.best, loss)

        if len(self.history) >= self.patience:
            window = self.history[-self.patience:]
            if min(window) > self.best * self.divergence_factor:
                self._warn(f"step {step}: loss has been above {self.divergence_factor}x its best "
                           f"({self.best:.4f}) for {self.patience} steps. This is divergence, not noise.")

        if grad_norm is not None and math.isfinite(grad_norm) and grad_norm < self.collapse_norm:
            self._warn(f"step {step}: gradient norm {grad_norm:.2e}. The model has stopped learning; "
                       "a flat loss here is collapse, not convergence.")

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
            print(f"[watchdog] {message}", flush=True)


def preflight(
    model=None,
    dataset=None,
    grid=None,
    latitudes=None,
    require_cuda: bool = False,
    min_vram_gb: float = 0.0,
    min_disk_gb: float = 0.0,
    disk_path: str = ".",
    strict: bool = True,
) -> dict:
    """
    Run every check that can be run before the expensive part, and say plainly what is wrong.

    Args:
        strict: raise :class:`PreflightError` on any problem. Set ``False`` to collect and continue,
            which is right for a report and wrong before an overnight run.

    Returns:
        ``{"problems": [...], "notes": [...]}``.
    """
    problems: list[str] = []
    notes: list[str] = []

    environment = check_environment(require_cuda, min_vram_gb, min_disk_gb, disk_path)
    problems += environment["problems"]
    notes.append(f"device: {environment.get('gpu', 'cpu')}"
                 + (f", {environment['vram_gb']:.0f} GB VRAM" if environment.get("vram_gb") else "")
                 + (f", {environment['free_disk_gb']:.0f} GB free" if environment.get("free_disk_gb") else ""))

    if grid is not None:
        problems += check_weights(grid)
        notes.append(f"grid: {grid.num_points:,} points, weights sum {float(grid.weights.sum()):.4f}")

    if dataset is not None and len(dataset) > 0:
        item = dataset[0]
        analysis = item["analysis"]
        problems += check_normalization(analysis, "analysis")
        if grid is not None:
            # The probe needs a temperature-like channel; channel 0 is one only by convention.
            variables = list(getattr(dataset, "variables", []))
            probe = variables.index("2m_temperature") if "2m_temperature" in variables else 0
            problems += check_grid_alignment(grid, analysis[-1, :, probe])
        if "field_target" in item:
            target = item["field_target"]
            if torch.isnan(target).any():
                problems.append("field_target contains NaN. Targets must be finite with a separate "
                                "mask -- the loss multiplies by the mask, and NaN times zero is NaN.")
        notes.append(f"dataset: {len(dataset):,} windows, analysis {tuple(analysis.shape)}")

    if model is not None:
        problems += check_device_agreement(model, grid)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        notes.append(f"model: {total/1e6:.2f}M parameters, {trainable/1e6:.2f}M trainable "
                     f"({100*trainable/max(total,1):.1f}%)")
        if trainable == 0:
            problems.append("no trainable parameters -- the backbone is frozen and no head was unfrozen")

    report = {"problems": problems, "notes": notes}
    if problems and strict:
        raise PreflightError("preflight failed:\n  - " + "\n  - ".join(problems))
    return report


def format_preflight(report: dict) -> str:
    """The printable version, for the top of a run."""
    lines = ["preflight"]
    for note in report["notes"]:
        lines.append(f"  ok    {note}")
    for problem in report["problems"]:
        lines.append(f"  FAIL  {problem}")
    if not report["problems"]:
        lines.append("  -> nothing to stop the run")
    return "\n".join(lines)
