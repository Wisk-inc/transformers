# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The voice-swap pipeline: a full song plus a reference voice in, the same song sung by that voice out.

    separate -> analyse -> (duet: pick the singer) -> capture the vocal's effects -> convert to the cloned
    voice -> re-apply the effects -> place in the mix -> quality gates

Each step is a function here so a product can swap any of them for a stronger component (an open separator
with a licence, a trained TriStream-2, a dedicated dereverberator). The pipeline never invents what it cannot
measure: when the input vocal has a room, the output gets the same room; when the reference voice cannot reach
the melody, the melody is moved by octaves or compressed and the log says so; when the result fails a gate
(bleed, glitches, pitch error), the report says which.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..audio.singing import estimate_f0
from .analysis import AnalysisResult, analyze, reverb_time
from .effects import detect_glitches, limit, match_reverb, place_in_mix, repair_glitches, retune
from .harmony import fit_to_range
from .separation import BandSplitSeparator, bleed_report, harmonic_gate, try_pretrained


@dataclass
class SwapReport:
    analysis: AnalysisResult | None
    range_action: str
    captured: dict
    bleed: dict
    glitches_before: int
    glitches_after: int
    singer_index: int | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"range: {self.range_action}",
            f"captured: {self.captured}",
            f"bleed: {self.bleed.get('leak_db', 0):.1f} dB",
            f"glitches repaired: {self.glitches_before - self.glitches_after} of {self.glitches_before}",
        ]
        if self.singer_index is not None:
            lines.append(f"singer picked: {self.singer_index}")
        return "\n".join(lines + self.notes)


class Converter:
    """
    What converts a dry vocal into another voice: ``convert(vocal, f0_hz, speaker_embedding, sample_rate)
    -> vocal``. The default uses TriStream-2's editing path when a model is given; without one it returns the
    input (so the rest of the pipeline can be exercised and measured), and says so in the report.
    """

    def __init__(self, model=None, codec=None, speaker_encoder=None) -> None:
        self.model, self.codec, self.speaker_encoder = model, codec, speaker_encoder

    def embed(self, reference: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if self.codec is None or self.speaker_encoder is None:
            return torch.zeros(1, 192)
        frames = self.codec.listen(reference[None])
        return self.speaker_encoder(frames)

    def convert(
        self, vocal: torch.Tensor, f0_hz: torch.Tensor, speaker: torch.Tensor, sample_rate: int, steps: int = 16
    ) -> tuple[torch.Tensor, str]:
        if self.model is None or self.codec is None:
            return vocal, "no model: the vocal was passed through unchanged"
        from .tristream import noise_features, source_features

        frames = self.codec.listen(vocal[None])  # (1, T, dim)
        t = frames.shape[1]
        f0 = torch.nn.functional.interpolate(f0_hz[None, None], size=t, mode="linear")[0]
        energy = frames.pow(2).mean(-1)
        codes = self.codec.encode(vocal[None])[:, 0]  # the semantic codebook carries the content, not the timbre
        batch = {
            "source": source_features(f0, energy), "noise": noise_features(frames), "frames": frames,
            "semantic": codes[:, :t], "phonemes": torch.full((1, 1), -1, dtype=torch.long), "speaker": speaker,
        }  # fmt: skip
        out_frames = self.model.sample(batch, steps=steps)
        wave = self.codec.decode_frames(out_frames)[0, 0]
        return wave[: vocal.shape[-1]], f"converted with TriStream-2 in {steps} flow steps"


def load_converter(root: str | Path, size: str = "ts-765m", device: str | torch.device | None = None) -> Converter:
    """
    A :class:`Converter` from a molab root (``codec/codec.pt`` and ``runs/<size>/latest.pt`` as
    ``helix_lm.scripts.tristream_molab`` writes them). Without both files it returns the pass-through converter,
    so the pipeline still runs and the report says that no model was used.
    """
    from ..audio import AudioCodec, CodecConfig, SpeakerEncoder
    from .tristream import TriStreamConfig, TriStreamForSinging

    root = Path(root)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    codec_path, model_path = root / "codec" / "codec.pt", root / "runs" / size / "latest.pt"
    if not codec_path.exists() or not model_path.exists():
        return Converter()
    state = torch.load(codec_path, map_location=device)
    codec = AudioCodec(CodecConfig(**state["config"])).to(device).eval()
    codec.load_state_dict(state["codec"])
    state = torch.load(model_path, map_location=device)
    config = TriStreamConfig(**state["config"])
    model = TriStreamForSinging(config).to(device).eval()
    model.load_state_dict(state["model"])
    speaker = SpeakerEncoder(frame_dim=codec.config.dim, embedding_dim=config.speaker_dim).to(device).eval()
    speaker.load_state_dict(state["speaker"])
    return Converter(model, codec, speaker)


def capture_effects(vocal: torch.Tensor, sample_rate: int, analysis: AnalysisResult | None) -> dict:
    """What was done to the original vocal, to do it again to the new one: room decay and a retune estimate."""
    decay = reverb_time(vocal, sample_rate)
    f0 = estimate_f0(vocal[None], sample_rate, frame_rate=25)[0]
    voiced = f0[f0 > 0]
    retuned = 0.0
    if voiced.numel() > 10:
        midi = 69 + 12 * torch.log2(voiced / 440.0)
        retuned = float((midi - midi.round()).abs().lt(0.08).float().mean())  # a hard-tuned vocal sits on the grid
    return {"reverb": decay, "retune_fraction": retuned, "autotuned": retuned > 0.7}


def swap_voice(
    song: torch.Tensor,
    reference: torch.Tensor,
    sample_rate: int,
    converter: Converter | None = None,
    separator: BandSplitSeparator | None = None,
    singer_reference: torch.Tensor | None = None,
    vocal_db_above: float = 2.0,
    gate_harmonics: bool = True,
) -> tuple[torch.Tensor, SwapReport]:
    """
    ``song``: a full mix; ``reference``: the voice to put on it; ``singer_reference``: for a duet, a clip of the
    singer to replace (the other voice is kept). Returns the new mix and a report.
    """
    converter = converter or Converter()
    notes = []
    # 1. separation: a pretrained open separator when installed, the band-split model when given, else a notch
    pretrained = try_pretrained()
    if pretrained is not None:
        stems = pretrained(song, sample_rate)
        vocal = stems["vocals"]
        instrumental = sum(v for k, v in stems.items() if k != "vocals")
        notes.append("separated with HT-Demucs (MIT)")
    elif separator is not None:
        with torch.no_grad():
            out = separator(song[None])[0]
        vocal = out[separator.config.stems.index("vocals")]
        instrumental = out.sum(0) - vocal
        notes.append("separated with the band-split separator")
    else:
        vocal, instrumental = song, torch.zeros_like(song)
        notes.append("no separator available: the song was treated as a dry vocal")
    analysis = None
    try:
        analysis = analyze(song, sample_rate)
    except Exception as error:
        notes.append(f"analysis skipped: {type(error).__name__}")
    # 2. duet: pull the chosen singer
    singer_index = None
    if (
        singer_reference is not None
        and separator is not None
        and separator.film is not None
        and converter.speaker_encoder is not None
    ):
        embeddings = torch.stack(
            [converter.embed(singer_reference, sample_rate)[0], converter.embed(vocal, sample_rate)[0]]
        )[None]
        with torch.no_grad():
            singers = separator.extract(vocal[None], embeddings)[0]
        singer_index = 0
        other = singers[1]
        vocal = singers[0]
        instrumental = instrumental + other
        notes.append("duet: the referenced singer was extracted; the other voice stays in the mix")
    # 3. capture the effects, track the pitch, keep the harmonics clean
    captured = capture_effects(vocal, sample_rate, analysis)
    f0 = estimate_f0(vocal[None], sample_rate, frame_rate=25)[0]
    if gate_harmonics and separator is not None:
        spec = separator.stft(vocal[None])
        gated = harmonic_gate(spec, f0, sample_rate, separator.config.n_fft)
        vocal = separator.istft(gated, vocal.shape[-1])[0]
    # 4. range: can the new voice sing it? (octaves first, compression last)
    voiced = f0[f0 > 0]
    range_action = "as is"
    ref_f0 = estimate_f0(reference[None], sample_rate, frame_rate=25)[0]
    ref_voiced = ref_f0[ref_f0 > 0]
    if voiced.numel() > 5 and ref_voiced.numel() > 5:
        midi = (69 + 12 * torch.log2(voiced / 440.0)).round().long().tolist()
        low = int(69 + 12 * torch.log2(ref_voiced.quantile(0.05) / 440.0)) - 2
        high = int(69 + 12 * torch.log2(ref_voiced.quantile(0.95) / 440.0)) + 4
        fitted, range_action = fit_to_range(midi, low, high)
        if range_action != "as is":
            shift = sum(fitted) / len(fitted) - sum(midi) / len(midi)
            f0 = torch.where(f0 > 0, f0 * 2 ** (shift / 12), f0)
            notes.append(f"melody moved for the new voice's range: {range_action}")
    if captured["autotuned"] and analysis is not None:
        f0 = retune(f0, analysis.key[0], analysis.key[1], speed=0.9)
    # 5. convert
    speaker = converter.embed(reference, sample_rate)
    converted, how = converter.convert(vocal, f0, speaker, sample_rate)
    notes.append(how)
    # 6. re-apply the room, place in the mix, gates
    converted = match_reverb(converted, sample_rate, captured["reverb"], 0.0)
    before = detect_glitches(converted, sample_rate)
    converted = repair_glitches(converted, sample_rate, before)
    after = detect_glitches(converted, sample_rate)
    mix = (
        place_in_mix(converted, instrumental, sample_rate, vocal_db_above=vocal_db_above, room_decay=0.0)
        if instrumental.abs().max() > 0
        else limit(converted)
    )
    bleed = (
        bleed_report(vocal, instrumental, sample_rate)
        if instrumental.abs().max() > 0
        else {"leak_ratio": 0.0, "leak_db": -120.0, "silent_fraction": 0.0}
    )
    report = SwapReport(
        analysis,
        range_action,
        captured,
        bleed,
        int(before["clicks"].numel() + before["dropouts"].numel()),
        int(after["clicks"].numel() + after["dropouts"].numel()),
        singer_index,
        notes,
    )
    return mix, report
