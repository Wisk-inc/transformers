# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The fail-safe loop: generate, listen, and act on what is heard, instead of hoping.

A one-shot generator ships whatever it produced. This loop ships only what passed its gates, and when a
take fails it does the next cheapest thing that fixes that specific failure:

1. **generate** with the current settings (steps, guidance, seed);
2. **listen**: the artifact report (:mod:`artifacts`), the pitch error against the control curve, the voice
   similarity against the reference, the stream-sync lag;
3. on failure, **escalate** where it matters: more flow steps, more guidance, a new seed; when the failure
   is local (clicks, dropouts, a smeared onset), **repair the span** only, through the model's own editing
   path, so the good parts are never regenerated;
4. when the output still smears consonants or air, **transplant** the source vocal's own unvoiced high band
   (:func:`refiner.transient_transplant`);
5. when harmonics still phase or buzz, render the failing span from the **source-filter sketch**
   (:func:`refiner.render_sketch`), which cannot phase because its harmonics are arithmetic;
6. always finish with glitch repair and a limiter, and return a report that says what was done and why.

Nothing here needs a trained model to run (the tests use the tiny one); everything here is what makes a
trained one safe to put in front of a listener.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..audio.refiner import render_sketch, transient_transplant
from ..audio.singing import estimate_f0
from .artifacts import artifact_report, summarize_report
from .effects import detect_glitches, limit, repair_glitches
from .humanize import fine_from_frames


@dataclass
class Gates:
    max_rounds: int = 3
    steps: int = 16
    max_steps: int = 64
    guidance: float = 1.5
    guidance_step: float = 0.5
    max_f0_rmse_cents: float = 60.0
    min_similarity: float | None = 0.6  # cosine to the reference voice; None skips the check
    min_similarity_gain: float = 0.0  # how much closer to the reference than to the source the output must be
    use_transplant: bool = True
    use_sketch_fallback: bool = True
    frame_rate: float = 25.0
    fine_rate: float = 100.0


@dataclass
class FailsafeReport:
    rounds: list[dict] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    final: dict = field(default_factory=dict)
    ok: bool = False

    def summary(self) -> str:
        lines = [f"{len(self.rounds)} round(s); applied: {', '.join(self.applied) or 'nothing extra'}"]
        for r in self.rounds:
            keys = ("f0_rmse_cents", "similarity", "similarity_source")
            extra = {k: round(v, 3) for k, v in r.items() if k in keys and isinstance(v, (int, float))}
            lines.append(f"  round {r['round']}: steps {r['steps']} guidance {r['guidance']} -> "
                         f"{'ok' if r['ok'] else 'failed ' + ', '.join(r['failed'])} {extra}")  # fmt: skip
        if self.final:
            lines.append(summarize_report(self.final))
        return "\n".join(lines)


def f0_error_cents(wave: torch.Tensor, f0_hz: torch.Tensor, sample_rate: int, frame_rate: float) -> float:
    est = estimate_f0(wave[None], sample_rate, frame_rate=frame_rate)[0]
    n = min(est.numel(), f0_hz.numel())
    est, ref = est[:n], f0_hz[:n]
    both = (est > 0) & (ref > 0)
    if both.sum() < 3:
        return 0.0
    cents = 1200 * torch.log2(est[both] / ref[both])
    return float(cents.pow(2).mean().sqrt())


def voice_similarity(codec, speaker_encoder, wave: torch.Tensor, reference: torch.Tensor) -> float:
    with torch.no_grad():
        a = speaker_encoder(codec.listen(wave[None]))
        b = speaker_encoder(codec.listen(reference[None]))
    return float(F.cosine_similarity(a, b, dim=-1).mean())


def frames_from_samples(positions: torch.Tensor, sample_rate: int, frame_rate: float, frames: int, pad: int = 2):
    """A frame mask around sample positions (clicks, dropouts): the span to repair."""
    mask = torch.zeros(frames, dtype=torch.bool)
    for pos in positions.tolist():
        f = int(pos / sample_rate * frame_rate)
        mask[max(0, f - pad) : min(frames, f + pad + 1)] = True
    return mask


def crossfade_in(base: torch.Tensor, patch: torch.Tensor, span: torch.Tensor, sample_rate: int, frame_rate: float):
    """Replace ``base`` by ``patch`` where the frame mask ``span`` is set, with 10 ms raised-cosine edges."""
    n = min(base.numel(), patch.numel())
    gate = F.interpolate(span.float()[None, None], size=n, mode="nearest")[0, 0]
    k = max(1, int(0.01 * sample_rate)) | 1
    gate = F.avg_pool1d(F.pad(gate[None, None], (k // 2, k // 2), mode="replicate"), k, stride=1)[0, 0]
    out = base.clone()
    out[:n] = base[:n] * (1 - gate) + patch[:n] * gate
    return out


@torch.no_grad()
def synthesize_with_gates(
    model,
    codec,
    batch: dict,
    sample_rate: int,
    f0_hz: torch.Tensor,
    *,
    refiner=None,
    f0_fine: torch.Tensor | None = None,
    speaker_encoder=None,
    reference_wave: torch.Tensor | None = None,
    source_wave: torch.Tensor | None = None,
    gates: Gates | None = None,
    mask: torch.Tensor | None = None,
    seed: int = 0,
    log=None,
) -> tuple[torch.Tensor, FailsafeReport]:
    """
    ``batch``: one TriStream record (batch size 1); ``f0_hz (T,)`` the control curve at the frame rate;
    ``f0_fine (T * subframes,)`` optional; ``reference_wave`` the voice to match; ``source_wave`` the original
    vocal (conversion) for the consonant and air checks and the transplant. Returns ``(wave (N,), report)``.
    """
    gates = gates or Gates()
    report = FailsafeReport()
    device = batch["source"].device
    t = batch["source"].shape[1]
    f0_hz = f0_hz.to(device).float()
    fine = (
        f0_fine.to(device).float()
        if f0_fine is not None
        else fine_from_frames(f0_hz, gates.fine_rate, gates.frame_rate)
    )
    steps, guidance = gates.steps, gates.guidance
    full = torch.ones(1, t, dtype=torch.bool, device=device) if mask is None else mask.to(device)
    repair = None
    frames = None
    wave = None

    def render(frames_now: torch.Tensor) -> torch.Tensor:
        w = codec.decode_frames(frames_now)[0, 0]
        if refiner is not None:
            n_fine = fine.numel()
            w = refiner(w[None, None], frames_now, fine[None, :n_fine])["wave"][0, 0]
        # a non-finite sample is the worst failure there is; it is zeroed here and named by the gates below
        return torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-4.0, 4.0)

    def listen(w: torch.Tensor) -> dict:
        rep = artifact_report(
            w,
            sample_rate,
            ref=source_wave,
            f0_hz=f0_hz,
            frame_rate=gates.frame_rate,
            f0_fine=fine,
            fine_rate=gates.fine_rate,
        )
        rep["f0_rmse_cents"] = f0_error_cents(w, f0_hz, sample_rate, gates.frame_rate)
        if rep["f0_rmse_cents"] > gates.max_f0_rmse_cents:
            rep["failed"].append("pitch")
        if float(w.abs().max()) >= 4.0 or float(w.pow(2).mean()) < 1e-10:
            rep["failed"].append("level")  # blown up or silent
        if gates.min_similarity is not None and speaker_encoder is not None and reference_wave is not None:
            rep["similarity"] = voice_similarity(codec, speaker_encoder, w, reference_wave.to(device))
            if source_wave is not None:
                rep["similarity_source"] = voice_similarity(codec, speaker_encoder, w, source_wave.to(device))
            if rep["similarity"] < gates.min_similarity or (
                source_wave is not None and rep["similarity"] - rep["similarity_source"] < gates.min_similarity_gain
            ):
                rep["failed"].append("voice")
        rep["ok"] = not rep["failed"]
        return rep

    for round_index in range(gates.max_rounds):
        g = torch.Generator(device="cpu").manual_seed(seed + round_index)
        this_batch = batch if frames is None else dict(batch, frames=frames)
        this_mask = full if repair is None else repair
        frames = model.sample(this_batch, steps=steps, mask=this_mask, guidance=guidance, generator=g)
        wave = render(frames)
        rep = listen(wave)
        entry = {
            "round": round_index,
            "steps": steps,
            "guidance": guidance,
            "ok": rep["ok"],
            "failed": list(rep["failed"]),
            "f0_rmse_cents": rep["f0_rmse_cents"],
            "similarity": rep.get("similarity"),
            "similarity_source": rep.get("similarity_source"),
            "span": "all" if repair is None else int(repair.sum()),
        }
        report.rounds.append(entry)
        if log:
            log(f"[failsafe] round {round_index}: {'ok' if rep['ok'] else 'failed ' + ', '.join(rep['failed'])}")
        if rep["ok"]:
            break
        # escalate, and narrow the next pass to the failing span when the failure is local
        steps = min(gates.max_steps, steps * 2)
        guidance = guidance + gates.guidance_step
        glitches = rep["glitches"]
        local = torch.cat([glitches["clicks"], glitches["dropouts"]])
        if local.numel() > 0 and set(rep["failed"]) <= {"glitches", "consonants"}:
            repair = frames_from_samples(local, sample_rate, gates.frame_rate, t)[None].to(device) & full
            report.applied.append(f"span repair ({int(repair.sum())} frames)")
        else:
            repair = None
    final = listen(wave)
    if (
        not final["ok"]
        and gates.use_transplant
        and source_wave is not None
        and ({"consonants", "hf"} & set(final["failed"]))
    ):
        wave = transient_transplant(wave, source_wave.to(device), f0_hz, sample_rate, gates.frame_rate)
        report.applied.append("transient transplant from the source vocal")
        final = listen(wave)
    if not final["ok"] and gates.use_sketch_fallback and ({"coherence", "buzz", "hf"} & set(final["failed"])):
        envelope, noise_bands = model.sketch_bands(batch)
        sketch_wave = render_sketch(fine[None], envelope, noise_bands, sample_rate, gates.fine_rate)[0]
        level = wave.pow(2).mean().sqrt() / (sketch_wave.pow(2).mean().sqrt() + 1e-9)
        wave = crossfade_in(wave, sketch_wave * level, full[0], sample_rate, gates.frame_rate)
        report.applied.append("source-filter sketch rendered for the failing span")
        final = listen(wave)
    glitches = detect_glitches(wave, sample_rate)
    if not glitches["ok"]:
        wave = repair_glitches(wave, sample_rate, glitches)
        report.applied.append("glitch repair")
    wave = limit(wave)
    report.final = final
    report.ok = final["ok"]
    return wave, report
