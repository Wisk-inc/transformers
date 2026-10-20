# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The critic: how likely is each moment of a song to give a listener chills, and why.

Frisson (musical chills) is the best-studied strong emotional response to music. Across the literature the
same triggers recur: a **sudden change in loudness or texture**, a crescendo reaching its peak, the **entry of a
new voice or instrument** (the solo voice emerging is the single most-cited trigger), an **expansion of the
frequency range** (a high note, a register leap), an **unprepared harmony** or melodic appoggiatura (surprise
against an established pattern), and **tension that resolves**. Listeners who get chills get them at the
moment of the event or a few seconds after; a build-up of tension before the event makes it stronger; and the
inverted-U of surprise holds: moderately unexpected beats wildly unexpected (Sloboda 1991; Panksepp 1995;
Grewe et al. 2007; Guhn, Hamm & Zentner 2007; Huron 2006; Salimpoor et al. 2011; Bannister 2020; de Fleurian
& Pearce 2021; Cheung et al. 2019). ``docs/TRISTREAM.md`` carries the citations and the numbers.

This module turns those findings into features computed per beat from :func:`analysis.analyze`, a weighted
score with the weights written out as **priors** (to be replaced by :class:`FrissonHead` once chill-timestamp
or arousal labels are available), peaks with explanations, and :func:`score_plan`, which estimates how much
a planned vocal intervention would add. The score is a prediction of a *population tendency*; it is not a
measurement of any listener, and it is not a claim about dopamine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .analysis import AnalysisResult

# Prior weights (log-odds units) from the literature's ranking of triggers; positive means "more chills".
PRIOR_WEIGHTS = {
    # Relative weights follow the evidence ranking (docs/TRISTREAM.md): loudness steps are the one trigger with a
    # causal manipulation behind them (Bannister 2020: raising a piece's loudness raised chill reports; brightness
    # changes did less); the entry of a voice is the most-cited qualitative trigger (Grewe 2007; Panksepp 1995;
    # vocals over-represented at chill onsets in the timestamped corpus of de Fleurian & Pearce); frequency-range
    # expansion (Guhn 2007); melodic surprise inside a predictable context (Gold 2019's inverted U; Cheung 2019);
    # tension release; the build / dip before the event (Solberg & Dibben 2019); groove and structural arrival
    # as smaller terms.
    "surge": 1.8,
    "crescendo": 0.7,
    "entry": 1.5,
    "range": 1.0,
    "surprise": 1.2,
    "resolution": 0.8,
    "silence": 0.7,
    "groove": 0.4,
    "downbeat": 0.4,
    "chorus_start": 0.6,
}
PRIOR_BIAS = -3.0


@dataclass
class FrissonFeatures:
    names: list[str]
    values: torch.Tensor  # (beats, features)


def _surprise(chords: list[str]) -> torch.Tensor:
    """Information content of each chord under a bigram model fitted to the song itself, in an inverted-U."""
    counts: dict[tuple[str, str], int] = {}
    totals: dict[str, int] = {}
    ic = torch.zeros(len(chords))
    for i in range(1, len(chords)):
        prev, cur = chords[i - 1], chords[i]
        seen = counts.get((prev, cur), 0) + 0.5
        total = totals.get(prev, 0) + 0.5 * 24
        ic[i] = -math.log(seen / total)
        counts[(prev, cur)] = counts.get((prev, cur), 0) + 1
        totals[prev] = totals.get(prev, 0) + 1
    ic = ic / (ic.max() + 1e-9)
    return 1 - (2 * ic - 1) ** 2  # peak at moderate surprise


def features_from_analysis(analysis: AnalysisResult) -> FrissonFeatures:
    beats = analysis.beats
    n = int(beats.numel())
    names = list(PRIOR_WEIGHTS)
    values = torch.zeros(n, len(names))
    if n < 4:
        return FrissonFeatures(names, values)
    rms_db = 20 * analysis.rms.clamp(min=1e-6).log10()
    centroid = analysis.centroid
    bounds = beats.tolist() + [rms_db.shape[0]]

    def beat_mean(x: torch.Tensor, i: int) -> float:
        a, b = bounds[i], max(bounds[i + 1], bounds[i] + 1)
        return float(x[a:b].mean())

    loud = torch.tensor([beat_mean(rms_db, i) for i in range(n)])
    bright = torch.tensor([beat_mean(centroid, i) for i in range(n)])
    chords = analysis.chords if len(analysis.chords) == n else ["C"] * n
    surprise = _surprise(chords)
    tension = analysis.tension
    tens = torch.tensor([beat_mean(tension, i) for i in range(n)])
    downbeat_set = set(analysis.downbeats.tolist())
    chorus_starts = {s["start_beat"] for s in analysis.segments if s["label"] in ("chorus", "drop")}
    resolution_set = set(analysis.resolutions)

    def clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    for i in range(n):
        prev2 = float(loud[max(0, i - 2) : i].mean()) if i > 0 else float(loud[i])
        surge = clamp01((float(loud[i]) - prev2) / 10)
        rise = loud[max(0, i - 8) : i + 1]
        monotone = rise.numel() > 4 and bool(rise.diff().ge(-1).all())
        crescendo = clamp01(float(rise[-1] - rise[0]) / 12) if monotone else 0.0
        bright_prev = float(bright[max(0, i - 2) : i].mean()) if i > 0 else float(bright[i])
        entry = 0.5 * clamp01((float(bright[i]) - bright_prev) / 1500) + 0.5 * surge
        bright_prev4 = float(bright[max(0, i - 4) : i].mean()) if i > 0 else float(bright[i])
        rng = clamp01((float(bright[i]) - bright_prev4) / 2500)
        resolution = 1.0 if i in resolution_set else clamp01(float(tens[i - 1] - tens[i]) * 3) if i > 0 else 0.0
        if i > 12:
            recent = float(loud[max(0, i - 4) : i].mean())
            earlier = float(loud[max(0, i - 12) : max(0, i - 4)].mean())
            silence = clamp01((earlier - recent) / 8)
        else:
            silence = 0.0
        values[i] = torch.tensor([
            surge, crescendo, entry, rng, float(surprise[i]), resolution, silence,
            analysis.syncopation * analysis.regularity, 1.0 if beats[i].item() in downbeat_set else 0.0,
            1.0 if i in chorus_starts else 0.0,
        ])  # fmt: skip
    return FrissonFeatures(names, values)
    rms_db = 20 * analysis.rms.clamp(min=1e-6).log10()
    centroid = analysis.centroid
    bounds = beats.tolist() + [rms_db.shape[0]]

    def beat_mean(x: torch.Tensor, i: int) -> float:
        a, b = bounds[i], max(bounds[i + 1], bounds[i] + 1)
        return float(x[a:b].mean())

    loud = torch.tensor([beat_mean(rms_db, i) for i in range(n)])
    bright = torch.tensor([beat_mean(centroid, i) for i in range(n)])
    chords = analysis.chords if len(analysis.chords) == n else ["C"] * n
    surprise = _surprise(chords)
    tension = analysis.tension
    tens = torch.tensor([beat_mean(tension, i) for i in range(n)])
    downbeat_set = set(analysis.downbeats.tolist())
    chorus_starts = {s["start_beat"] for s in analysis.segments if s["label"] in ("chorus", "drop")}
    resolution_set = set(analysis.resolutions)
    for i in range(n):
        prev2 = loud[max(0, i - 2) : i].mean() if i > 0 else loud[i]
        surge = float(((loud[i] - prev2) / 10).clamp(0, 1))
        rise = loud[max(0, i - 8) : i + 1]
        crescendo = (
            float(((rise[-1] - rise[0]) / 12).clamp(0, 1))
            if rise.numel() > 4 and bool(rise.diff().ge(-1).all())
            else 0.0
        )
        entry = (
            float((((bright[i] - bright[max(0, i - 2) : i].mean()) / 1500) if i > 0 else 0.0).clamp(0, 1)) * 0.5
            + surge * 0.5
        )
        rng = float(((bright[i] - bright[max(0, i - 4) : i].mean()) / 2500).clamp(0, 1)) if i > 0 else 0.0
        resolution = 1.0 if i in resolution_set else float(((tens[i - 1] - tens[i]) * 3).clamp(0, 1)) if i > 0 else 0.0
        silence = (
            float(((loud[max(0, i - 4) : i].mean() - loud[max(0, i - 12) : max(0, i - 4)].mean()) / -8).clamp(0, 1))
            if i > 12
            else 0.0
        )
        values[i] = torch.tensor([
            surge, crescendo, entry, rng, float(surprise[i]), resolution, silence,
            analysis.syncopation * analysis.regularity, 1.0 if beats[i].item() in downbeat_set else 0.0,
            1.0 if i in chorus_starts else 0.0,
        ])  # fmt: skip
    return FrissonFeatures(names, values)


def frisson_timeline(
    analysis: AnalysisResult, weights: dict[str, float] | None = None, bias: float = PRIOR_BIAS
) -> torch.Tensor:
    """Per-beat probability-like score in [0, 1] from the prior weights (or learned ones)."""
    feats = features_from_analysis(analysis)
    w = torch.tensor([(weights or PRIOR_WEIGHTS)[name] for name in feats.names])
    return torch.sigmoid(feats.values @ w + bias)


def frisson_peaks(analysis: AnalysisResult, top: int = 5, min_gap_seconds: float = 20.0) -> list[dict]:
    """The strongest moments with the features that made them: ``{"beat", "time", "score", "because"}``."""
    feats = features_from_analysis(analysis)
    scores = frisson_timeline(analysis)
    order = scores.argsort(descending=True).tolist()
    chosen: list[dict] = []
    for i in order:
        min_gap_beats = max(1, int(min_gap_seconds * analysis.tempo / 60.0))  # anticipation ~15 s + chill ~7 s
        if any(abs(i - c["beat"]) < min_gap_beats for c in chosen):
            continue
        contributions = {name: float(feats.values[i, j] * PRIOR_WEIGHTS[name]) for j, name in enumerate(feats.names)}
        because = [k for k, v in sorted(contributions.items(), key=lambda kv: -kv[1]) if v > 0.3][:3]
        chosen.append(
            {
                "beat": i,
                "time": float(analysis.beats[i]) / analysis.frame_rate,
                "score": float(scores[i]),
                "because": because,
            }
        )
        if len(chosen) >= top:
            break
    return chosen


# -------------------------------------------------------------------------------- planned additions --
INTERVENTION_GAIN = {
    # what a vocal plan can add at a moment, in the same log-odds units as the priors; from the triggers'
    # ranking with the human voice weighted highest (Panksepp 1995; Grewe 2007; Bannister 2020)
    "voice_entry": 1.5,  # a voice enters where there was none
    "rising_line": 0.7,  # a scale-wise ascent into the arrival
    "octave_leap": 0.9,  # a leap of an octave or more to a held high note
    "held_high_note": 0.8,
    "harmony_entry": 1.0,  # parts open up under or over the lead
    "breathy_to_belt": 0.9,  # intimate verse delivery followed by full voice
    "pre_entry_rest": 0.5,  # a beat of silence right before the entry
    "ad_lib_over_chorus": 0.4,
}
NATURALNESS_PENALTY = {"too_dense": -1.2, "off_grid": -1.5, "out_of_key": -2.0, "out_of_range": -1.0, "overused": -0.6}


def score_plan(analysis: AnalysisResult, plan) -> dict:
    """
    Expected frisson added by a :class:`~helix_lm.music.conductor.VocalPlan`: for each phrase, the base score of
    its entry beat plus the interventions it carries, minus naturalness penalties (too many interventions per
    minute, entries off the beat grid, notes outside key or range). Returns totals and per-phrase lines.
    """
    base = frisson_timeline(analysis)
    beats_per_minute = analysis.tempo
    lines, total, interventions = [], 0.0, 0
    for phrase in plan.phrases:
        b = min(max(phrase.start_beat, 0), base.shape[0] - 1) if base.numel() else 0
        logit = math.log(max(float(base[b]), 1e-6) / max(1 - float(base[b]), 1e-6)) if base.numel() else PRIOR_BIAS
        gain = sum(INTERVENTION_GAIN.get(tag, 0.0) for tag in phrase.interventions)
        penalty = 0.0
        if phrase.start_beat not in set(range(0, 10**6)) or (
            phrase.start_beat % 4 != 0 and "on_grid" not in phrase.interventions
        ):
            penalty += 0.0 if phrase.start_beat % 2 == 0 else NATURALNESS_PENALTY["off_grid"] * 0.5
        if phrase.flags.get("out_of_key"):
            penalty += NATURALNESS_PENALTY["out_of_key"]
        if phrase.flags.get("out_of_range"):
            penalty += NATURALNESS_PENALTY["out_of_range"]
        interventions += len(phrase.interventions)
        score = 1 / (1 + math.exp(-(logit + gain + penalty)))
        lines.append(
            {"phrase": phrase.start_beat, "section": phrase.section, "score": score, "gain": gain, "penalty": penalty}
        )
        total += score
    minutes = max(len(base) / max(beats_per_minute, 1), 1e-6)
    density = interventions / minutes
    density_penalty = NATURALNESS_PENALTY["too_dense"] * max(0.0, density - 6) / 6
    return {"total": total + density_penalty, "per_phrase": lines, "interventions_per_minute": density}


# --------------------------------------------------------------------------------------- learned head --
class FrissonHead(nn.Module):
    """
    A small network over the same per-beat features (plus a window of neighbours), to be fitted on chill
    timestamps or continuous arousal ratings. Initialized to reproduce the prior weights, so with no data it is
    the prior and with data it departs from it only as far as the labels justify.
    """

    def __init__(self, num_features: int = len(PRIOR_WEIGHTS), context: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.context = context
        self.linear = nn.Linear(num_features, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([list(PRIOR_WEIGHTS.values())]))
            self.linear.bias.fill_(PRIOR_BIAS)
        self.residual = nn.Sequential(
            nn.Conv1d(num_features, hidden, 2 * context + 1, padding=context), nn.GELU(), nn.Conv1d(hidden, 1, 1)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # (B, beats, F) -> (B, beats)
        prior = self.linear(features)[..., 0]
        learned = self.residual(features.transpose(1, 2))[:, 0]
        return torch.sigmoid(prior + learned)

    def fit(
        self, features: list[torch.Tensor], labels: list[torch.Tensor], steps: int = 200, lr: float = 1e-2
    ) -> float:
        """Binary cross-entropy on per-beat labels in [0, 1]; returns the final loss."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss = torch.zeros(())
        for _ in range(steps):
            total = torch.zeros(())
            for f, y in zip(features, labels, strict=True):
                pred = self(f[None])[0]
                total = total + nn.functional.binary_cross_entropy(pred, y.float())
            loss = total / max(len(features), 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return float(loss.detach())
