# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The second round of hardening: intake for real-world audio, robust reference voices, long-form drift control,
register physics, the air band and the codec-hole / noise / residue augmentations."""

import torch

from helix_lm.audio import AudioCodec, CodecConfig, SpeakerEncoder
from helix_lm.audio.data import synthetic_voice
from helix_lm.audio.refiner import AirBandExtension, AirConfig, air_losses, harmonic_oscillator
from helix_lm.music.artifacts import air_report, pre_echo_score
from helix_lm.music.conversion import Converter, swap_voice
from helix_lm.music.effects import highpass, lowpass, reverb
from helix_lm.music.intake import (
    assess_input,
    clean_input,
    codec_holes,
    condition_invariance_loss,
    declip,
    reference_embedding,
    reference_windows,
    separation_sanity,
    trimmed_mean_embedding,
)
from helix_lm.music.longform import drift_curve, drift_rate, plan_chunks, render_long, time_warp
from helix_lm.music.register import (
    comfortable_range,
    formant_tuning,
    formants,
    hollow_score,
    register_plan,
    tract_scale,
    warp_envelope,
)
from helix_lm.music.regularize import augment, codec_hole_augment
from helix_lm.music.tristream import TriStreamConfig, TriStreamForSinging, noise_features, source_features

SR = 16000


def studio_voice(seconds: float = 2.0, sample_rate: int = SR) -> torch.Tensor:
    """A full-bandwidth, noiseless phrase with rests: pure harmonics from the oscillator."""
    frames = int(seconds * 100)
    f0 = torch.zeros(1, frames)
    f0[0, 10 : frames // 2 - 10] = 220.0
    f0[0, frames // 2 + 10 : frames - 10] = 262.0
    harmonics = int(sample_rate / 2 / 262.0) - 1
    amps = (1.0 / torch.arange(1, harmonics + 1).float())[None, None].expand(1, frames, harmonics) * 0.3
    voice = harmonic_oscillator(f0, amps, sample_rate, sample_rate // 100)[0]
    g = torch.Generator().manual_seed(0)
    floor = torch.randn(voice.numel(), generator=g) * float(voice.pow(2).mean().sqrt()) * 10 ** (-50 / 20)
    return voice + floor  # every microphone has a floor; a studio one sits 50 dB down


def test_intake_measures_and_cleans_real_world_damage():
    voice = studio_voice(2.0)
    clean = assess_input(voice, SR)
    assert clean["noise_floor_db"] < -50 and clean["bandwidth_hz"] > SR / 2 * 0.8, clean
    dirty_wave = ((reverb(voice, SR, 1.5, 0.5) + 0.02 * torch.randn(voice.numel())) * 3.0).clamp(-1, 1)
    dirty = assess_input(dirty_wave, SR)
    assert (
        clean["quality"] > dirty["quality"]
        and dirty["clipped_fraction"] > 0
        and dirty["noise_floor_db"] > clean["noise_floor_db"]
    )
    cleaned, applied = clean_input(dirty_wave, SR, dirty)
    assert applied and cleaned.shape == voice.shape and torch.isfinite(cleaned).all()
    assert assess_input(cleaned, SR)["noise_floor_db"] < dirty["noise_floor_db"]
    holed = codec_hole_augment(voice, SR, 36.0)
    assert codec_holes(holed, SR) > codec_holes(voice, SR) + 0.05
    inst = torch.randn(voice.numel()) * 0.1
    assert separation_sanity(voice, inst, voice + inst, SR)["ok"]
    cancelled = separation_sanity(voice - 0.6 * lowpass(inst, 1500, SR), inst, voice + inst, SR)
    assert cancelled["voice_band_correlation"] < 0
    declipped, runs = declip((voice * 3).clamp(-1, 1))
    assert runs > 0 and declipped.abs().max() > 1.0, "the arcs reach past the rails the clipping imposed"
    _, applied = augment(voice, SR, instrumental=inst, seed=11)
    assert isinstance(applied, dict)


def test_reference_embedding_drops_contaminated_windows():
    torch.manual_seed(0)
    sr = 8000
    codec = AudioCodec(
        CodecConfig(
            sample_rate=sr, dim=32, channels=8, strides=(2, 4, 5, 8), num_codebooks=4, codebook_size=64, code_dim=4
        )
    ).eval()
    speaker = SpeakerEncoder(frame_dim=32, embedding_dim=16).eval()
    quiet = synthetic_voice(1.2, sr) * 0.3
    shout = synthetic_voice(1.0, sr, f0=330.0)
    shout = shout / shout.abs().max()
    reference = torch.cat([quiet, torch.zeros(int(0.9 * sr)), shout, synthetic_voice(1.5, sr) * 0.3])
    windows, selection = reference_windows(reference, sr)
    assert selection["kept"] >= 1 and selection["dropped"]["whisper_or_silence"] >= 1
    e = torch.randn(6, 16)
    e[5] *= 10
    robust = trimmed_mean_embedding(e)
    assert robust.shape == (1, 16) and abs(float(robust.norm()) - 1) < 1e-5
    embedding, report = reference_embedding(reference, sr, codec, speaker)
    assert embedding.shape == (1, 16) and 0 <= report["contamination"] <= 2 and report["kept"] >= 1
    clean = torch.stack([synthetic_voice(0.5, sr), synthetic_voice(0.5, sr, f0=300.0)])
    dirty = torch.stack([reverb(c, sr, 0.8, 0.4) for c in clean])
    loss = condition_invariance_loss(speaker, codec, clean, dirty)
    assert torch.isfinite(loss) and loss >= 0
    loss.backward()
    assert any(p.grad is not None for p in speaker.parameters())


def test_longform_grid_has_no_accumulation():
    chunks = plan_chunks(2000, 750, 50, anchors=torch.tensor([700, 1450, 1900]))
    assert chunks[0] == (0, 750) or chunks[0][1] in (700, 750)
    assert chunks[-1][1] == 2000 and all(b - a <= 750 for a, b in chunks)
    assert all(chunks[i + 1][0] < chunks[i][1] for i in range(len(chunks) - 1)), "chunks overlap"
    hop = SR // 25
    out = render_long(lambda a, z: torch.ones((z - a) * hop) * (a + 1), 300, hop, 100, 10)
    assert out.shape == (300 * hop,) and out.min() > 0 and abs(float(out[:hop].mean()) - 1) < 1e-4
    assert abs(float(out[-hop:].mean()) - float(out[-hop:].mean().round())) < 0.5, (
        "the last chunk sits at its own position"
    )
    warped = time_warp(torch.linspace(100, 200, 100), 1.01)
    assert warped.numel() == 101
    assert (
        abs(drift_rate(torch.tensor([10.0, 30.0, 50.0]), torch.tensor([0.0, 10.0, 20.0]))["ms_per_minute"] - 30) < 1e-3
    )
    # a track that drifts: the controls run 1 % slow against the audio and the curve says so
    from helix_lm.music.humanize import HumanStyle, humanize

    g = torch.Generator().manual_seed(4)
    notes, t = [], 0.0
    while t < 40.0:  # 40 s of notes with irregular lengths, so the event train has no period to alias on
        length = 0.25 + 0.6 * float(torch.rand((), generator=g))
        notes.append((60 + int(torch.randint(0, 8, (), generator=g)), t, t + length * 0.85))
        t += length
    f0 = humanize(notes, HumanStyle(), 100.0, seed=0)["f0_hz"]
    amps = torch.zeros(1, f0.numel(), 3)
    amps[..., 0], amps[..., 1] = 0.3, 0.1
    wave = harmonic_oscillator(f0[None], amps, SR, SR // 100)[0]
    control = time_warp(f0[::4], 1.01)[: f0.numel() // 4]
    times, lags, reliable = drift_curve(control, wave, SR, 25.0, window_s=10.0)
    assert times.numel() >= 3
    rate = drift_rate(times, lags, reliable)
    assert rate["ms_per_minute"] < -200, rate  # the audio runs ahead of a 1 % stretched control: about -600 ms/min
    assert abs(float(drift_rate(times, torch.zeros_like(lags))["ms_per_minute"])) < 1e-6


def test_register_physics_and_envelope_warp():
    voice = synthetic_voice(1.5, SR)
    f = formants(voice, SR)
    assert len(f) == 3 and 90 < f[0] < f[1] < f[2] < 4000
    scale = tract_scale(voice, SR)
    assert 0.6 <= scale <= 1.6
    lo, hi = comfortable_range(voice, SR)
    assert 0 < lo <= hi
    easy = register_plan((100, 300), (110, 220), scale)
    high = register_plan((100, 300), (220, 900), scale)
    absurd = register_plan((100, 300), (300, 1400), scale)
    assert easy["possible"] and easy["formant_ratio"] == 1.0 and "chest" in easy["register"]
    assert 1.0 < high["formant_ratio"] <= 1.2 and high["semitones_above"] > 12
    assert not absurd["possible"] and "move the melody" in absurd["action"]
    tuned = formant_tuning(torch.tensor([200.0, 600.0, 0.0]), 500.0)
    assert tuned.tolist()[0] == 500.0 and tuned[1] > 600 and tuned[2] == 500.0
    env = torch.randn(2, 5, 32)
    assert torch.allclose(warp_envelope(env, 1.0), env, atol=1e-5)
    up = warp_envelope(env, torch.tensor([1.0, 1.3]))
    assert torch.allclose(up[0], env[0], atol=1e-5) and not torch.allclose(up[1], env[1])
    assert hollow_score(voice, voice, SR)["ok"]
    hollow = hollow_score(highpass(voice, 1500, SR), voice, SR)
    assert hollow["f1_deficit_db"] > 6 and not hollow["ok"]
    # the model accepts the quality condition and the tract ratio
    cfg = TriStreamConfig.named("ts-tiny")
    model = TriStreamForSinging(cfg).eval()
    b, t = 2, 16
    f0 = torch.full((b, t), 220.0)
    frames = torch.randn(b, t, cfg.frame_dim)
    batch = {
        "source": source_features(f0, torch.rand(b, t)), "noise": noise_features(frames), "frames": frames,
        "phonemes": torch.full((b, 1), -1, dtype=torch.long), "speaker": torch.randn(b, cfg.speaker_dim),
        "quality": torch.tensor([0.9, 0.4]), "tract_ratio": torch.tensor([1.0, 1.15]),
    }  # fmt: skip
    out = model.sample(batch, steps=2)
    assert out.shape == frames.shape and torch.isfinite(out).all()


def test_air_band_and_the_mp3_detectors():
    torch.manual_seed(0)
    cfg = AirConfig(in_rate=16000, out_rate=32000, bands=8, context_bands=16, dim=16)
    air = AirBandExtension(cfg)
    voice = synthetic_voice(0.5, 16000)[None, None]
    out = air(voice)
    assert out["wave"].shape == (1, 1, voice.shape[-1] * 2) and torch.isfinite(out["wave"]).all()
    target = torch.nn.functional.interpolate(voice, size=voice.shape[-1] * 2, mode="linear")[0] + 0.01 * torch.randn(
        1, voice.shape[-1] * 2
    )
    losses = air_losses(out, target, 32000, cutoff=7000.0)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert air.net[0].weight.grad is not None
    full = synthetic_voice(0.6, 48000)
    dull = lowpass(full, 9000.0, 48000)
    report = air_report(dull, full, 48000)
    assert report["measured"] and report["air_deficit_db"] > 3
    assert air_report(full, full, 48000)["ok"]
    assert pre_echo_score(full, full, 48000)["ok"]


def test_swap_voice_runs_intake_register_and_reports():
    from helix_lm.music.analysis import synthetic_song

    song = synthetic_song(6.0, SR, bpm=120, key=7)
    song = ((reverb(song, SR, 1.2, 0.4) + 0.01 * torch.randn(song.numel())) * 1.3).clamp(-1, 1)
    reference = synthetic_voice(1.0, SR, f0=196.0)
    mix, report = swap_voice(song, reference, SR, converter=Converter())
    text = report.summary()
    assert mix.shape == song.shape and "intake" in text and "register:" in text and "no model" in text


def test_converter_long_form_uses_one_grid():
    torch.manual_seed(0)
    sr = 8000
    codec = AudioCodec(
        CodecConfig(
            sample_rate=sr, dim=32, channels=8, strides=(2, 4, 5, 8), num_codebooks=4, codebook_size=64, code_dim=4
        )
    ).eval()
    model = TriStreamForSinging(TriStreamConfig.named("ts-tiny", frame_dim=32, speaker_dim=16, sample_rate=sr)).eval()
    speaker = SpeakerEncoder(frame_dim=32, embedding_dim=16).eval()
    converter = Converter(model, codec, speaker)
    converter.chunk_seconds = 1.0
    vocal = synthetic_voice(2.6, sr)
    f0 = torch.full((int(2.6 * 25),), 220.0)
    out, how = converter.convert(vocal, f0, speaker(codec.listen(vocal[None])), sr, steps=2)
    assert out.shape[-1] >= vocal.numel() - sr // 25 and "chunks on one grid" in how and torch.isfinite(out).all()
    assert hasattr(converter, "last_drift") and "ms_per_minute" in converter.last_drift
