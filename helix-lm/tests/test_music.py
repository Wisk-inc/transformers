# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The music stack on synthetic songs: analysis finds the tempo, key and sections; the critic and conductor
produce a sane plan; harmony, effects, separation, regularization, data and TriStream-2 run at toy scale."""

import torch

from helix_lm.audio.data import synthetic_voice, write_wave
from helix_lm.music.analysis import NOTE_NAMES, analyze, snap_midi_to_scale, synthetic_song
from helix_lm.music.conductor import plan_vocals, render_controls
from helix_lm.music.conversion import Converter, swap_voice
from helix_lm.music.data import describe_music_sources, find_audio, prepare_uploads
from helix_lm.music.effects import (
    add_vibrato,
    detect_glitches,
    duck,
    echo,
    formant_shift,
    insert_breaths,
    muffle,
    repair_glitches,
    retune,
    reverb,
)
from helix_lm.music.frisson import FrissonHead, features_from_analysis, frisson_peaks, frisson_timeline, score_plan
from helix_lm.music.harmony import climax_figure, fit_to_range, harmonize, rising_line
from helix_lm.music.regularize import ClipTags, augment, balanced_weights, breath_events, is_dry_enough, split_singers
from helix_lm.music.separation import BandSplitSeparator, SeparatorConfig, bleed_report, separation_loss, si_sdr
from helix_lm.music.tristream import TriStreamConfig, TriStreamForSinging, noise_features, source_features

SR = 16000


def test_analysis_finds_tempo_key_and_sections():
    song = synthetic_song(16.0, SR, bpm=120, key=7)
    r = analyze(song, SR)
    assert abs(r.tempo - 120) < 6 and r.key[0] == 7 and r.key[1] == "major"
    assert r.beats.numel() > 20 and r.downbeats.numel() >= 6 and r.chords[0] == "G"
    labels = [s["label"] for s in r.segments]
    assert "chorus" in labels and r.segments[0]["start_beat"] == 0
    gaps = r.beat_times().diff()
    assert (gaps - 0.5).abs().max() < 0.08, "beats sit on the 120 BPM grid"
    minor = analyze(synthetic_song(12.0, SR, bpm=92, key=2, minor=True), SR)
    assert minor.key == (2, "minor", minor.key[2]) and abs(minor.tempo - 92) < 6
    assert snap_midi_to_scale(torch.tensor([61, 66]), 7, "major").tolist() == [60, 66], (
        "C# snaps down to C; F# is in G major"
    )
    assert "BPM" in r.describe() and NOTE_NAMES[r.key[0]] in r.key_name


def test_critic_and_conductor_make_a_sane_plan():
    r = analyze(synthetic_song(16.0, SR, bpm=120, key=7), SR)
    feats = features_from_analysis(r)
    assert feats.values.shape == (r.beats.numel(), len(feats.names)) and feats.values.min() >= 0
    timeline = frisson_timeline(r)
    assert timeline.shape[0] == r.beats.numel() and 0 <= timeline.min() <= timeline.max() <= 1
    peaks = frisson_peaks(r, top=2)
    assert peaks and peaks[0]["because"]
    assert abs(peaks[0]["beat"] - 16) <= 2, "the loud half starting at beat 16 is the moment"
    plan = plan_vocals(r, singer_range=(55, 79))
    assert plan.phrases and all(p.start_beat % 4 == 0 or p.section == "build" for p in plan.phrases)
    notes = [n for p in plan.phrases for n in p.notes if n >= 0]
    assert all(55 <= n <= 79 for n in notes), "every note inside the singer's range"
    assert all(n % 12 in {7, 9, 11, 0, 2, 4, 6} for n in notes), "every note in G major"
    chorus = [p for p in plan.phrases if p.section == "chorus"]
    assert chorus and chorus[0].sliders["power"] > 0.5
    scores = score_plan(r, plan)
    assert scores["total"] > 0 and scores["interventions_per_minute"] >= 0
    controls = render_controls(plan, r)
    assert controls["f0_hz"].shape == controls["power"].shape and (controls["f0_hz"] > 0).sum() > 50
    head = FrissonHead()
    labels = (timeline > 0.3).float()
    assert head.fit([feats.values], [labels], steps=20) < 1.0
    assert "why:" in plan.describe()


def test_harmony_helpers():
    fitted, how = fit_to_range([84, 86, 88], 55, 79)
    assert how == "octave -1" and fitted == [72, 74, 76]
    squeezed, how = fit_to_range([40, 60, 90], 55, 79)
    assert how == "compressed" and min(squeezed) >= 55 and max(squeezed) <= 79
    lead = [67, 69, 71, 72]
    parts = harmonize(lead, ["G", "G", "C", "C"], parts=2, tonic=7, mode="major")
    assert len(parts) == 2 and all(p < top for top, p in zip(lead, parts[0], strict=True))
    assert all(abs(a - b) >= 3 for a, b in zip(lead, parts[0], strict=True))
    line = rising_line(60, 72, 6, 7, "major")
    assert line == sorted(line) and line[-1] >= 71
    figure = climax_figure(62, "G", 7, "major")
    assert figure[2] - figure[1] == 12 and figure[2] == figure[3]


def test_effects_and_glitch_repair():
    voice = synthetic_voice(0.6, SR)
    wet = reverb(voice, SR, decay_seconds=1.0, mix=0.3)
    assert wet.shape == voice.shape and (wet - voice).abs().mean() > 1e-3
    assert echo(voice, SR, 0.25).shape == voice.shape and muffle(voice, SR).abs().max() <= 1.0
    shifted = formant_shift(voice, SR, 1.15)
    assert shifted.shape == voice.shape and torch.isfinite(shifted).all()
    f0 = torch.full((50,), 223.0)
    tuned = retune(f0, 9, "major", speed=1.0)  # A major: 223 Hz is a few cents above A3 = 220
    assert (tuned - 220).abs().max() < 0.5
    assert add_vibrato(f0).std() > 0
    inst = torch.randn(voice.shape[0]) * 0.1
    ducked = duck(inst, voice, SR, depth_db=6)
    assert ducked.abs().mean() < inst.abs().mean()
    breathed = insert_breaths(voice, SR, [0.3])
    assert (breathed - voice).abs().sum() > 0
    broken = voice.clone()
    broken[3000] += 0.9
    broken[6000:6400] = 0
    report = detect_glitches(broken, SR)
    assert report["clicks"].numel() >= 1 and not report["ok"]
    repaired = repair_glitches(broken, SR, report)
    assert detect_glitches(repaired, SR)["clicks"].numel() < report["clicks"].numel()


def test_separator_trains_and_reports_bleed():
    torch.manual_seed(0)
    config = SeparatorConfig(
        sample_rate=SR, n_fft=256, hop=64, num_bands=6, dim=16, layers=1, stems=("vocals", "other"), speaker_dim=8
    )
    model = BandSplitSeparator(config)
    voice = synthetic_voice(0.5, SR, f0=220.0)
    other = torch.randn(voice.shape[0]) * 0.1
    mix = voice + other
    targets = torch.stack([voice, other])[None]
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = last = None
    for _ in range(25):
        est = model(mix[None])
        loss = separation_loss(est, targets, SR)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    assert last < first
    est = model(mix[None]).detach()
    assert est.shape == (1, 2, voice.shape[0])
    assert si_sdr(est[:, 0], voice[None]).item() > si_sdr(mix[None], voice[None]).item() - 3
    singers = torch.randn(1, 2, 8)
    extracted = model.extract(mix[None], singers)
    assert extracted.shape == (1, 2, voice.shape[0])
    report = bleed_report(voice, other, SR)
    assert "leak_db" in report


def test_regularization_rules():
    tags = [
        ClipTags(
            singer=f"s{i}", accent="jamaican" if i % 3 == 0 else "us_general", f0_low=100.0, f0_high=300.0 + 60 * i
        )
        for i in range(12)
    ]
    weights = balanced_weights(tags, by="accent")
    assert abs(weights.sum() - 1) < 1e-6 and weights[0] > weights[1], "the rarer accent is up-weighted"
    split = split_singers([t.singer for t in tags] + [f"x{i}" for i in range(200)], validation_fraction=0.1)
    assert 0.03 < sum(v == "validation" for v in split.values()) / len(split) < 0.25
    voice = synthetic_voice(0.8, SR)
    dry, measures = is_dry_enough(voice, SR)
    assert dry and measures["reverb"] < 0.6
    augmented, applied = augment(voice, SR, instrumental=torch.randn(voice.shape[0]) * 0.05, seed=3)
    assert augmented.shape == voice.shape and augmented.abs().max() <= 0.951
    assert isinstance(breath_events(voice, SR), list)


def test_uploads_are_prepared_and_catalogue_is_honest(tmp_path):
    path = write_wave(tmp_path / "take1.wav", synthetic_voice(1.0, SR), SR)
    (tmp_path / "sub").mkdir()
    write_wave(tmp_path / "sub" / "take2.wav", synthetic_voice(0.8, SR, f0=330.0), SR)
    files = find_audio([tmp_path])
    assert len(files) == 2
    manifest = prepare_uploads(
        [tmp_path], tmp_path / "out", sample_rate=SR, singer="tester", accent="jamaican", log=lambda *_: None
    )
    assert len(manifest["clips"]) == 2 and manifest["accepted"] >= 1 and (tmp_path / "out" / "manifest.json").exists()
    assert any("range" in g for g in manifest["gaps"]), "two clips cannot cover the registers; the gaps say so"
    table = describe_music_sources(commercial_only=True)
    assert "vocalset" in table and "gtsinger" not in table and "musdb18-hq" not in table
    assert "research" in describe_music_sources()
    assert str(path).endswith("take1.wav")


def test_tristream_trains_edits_and_predicts():
    torch.manual_seed(0)
    cfg = TriStreamConfig.named("ts-tiny")
    model = TriStreamForSinging(cfg)
    b, t, length = 2, 32, 7
    f0 = torch.where(torch.rand(b, t) > 0.2, 220.0 * (torch.rand(b, t) + 0.5), torch.zeros(b, t))
    frames = torch.randn(b, t, cfg.frame_dim)
    batch = {
        "source": source_features(f0, torch.rand(b, t)), "noise": noise_features(frames), "frames": frames,
        "phonemes": torch.randint(0, cfg.num_phonemes, (b, length)), "speaker": torch.randn(b, cfg.speaker_dim),
        "singer": torch.randint(0, cfg.num_singers, (b,)), "accent": torch.randint(0, 5, (b,)),
        "sliders": torch.rand(b, t, 10), "durations": torch.randint(1, 6, (b, length)), "f0_hz": f0,
    }  # fmt: skip
    losses = model(batch)
    assert all(torch.isfinite(v) for v in losses.values()) and 0 < losses["mask_fraction"] < 1
    losses["total"].backward()
    assert model.flow_out.weight.grad is not None and model.source_in.weight.grad is not None
    model.eval()
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[:, 8:16] = True
    edited = model.sample(batch, steps=2, mask=mask)
    assert torch.equal(edited[:, :8], frames[:, :8]) and not torch.allclose(edited[:, 8:16], frames[:, 8:16])
    assert model.predict_pitch(batch).shape == (b, t)
    assert TriStreamForSinging(TriStreamConfig.named("ts-tiny")).num_parameters() == model.num_parameters()


def test_swap_voice_runs_without_a_model():
    song = synthetic_song(6.0, SR, bpm=120, key=7)
    reference = synthetic_voice(1.0, SR, f0=196.0)
    config = SeparatorConfig(
        sample_rate=SR, n_fft=256, hop=64, num_bands=6, dim=16, layers=1, stems=("vocals", "other")
    )
    mix, report = swap_voice(song, reference, SR, converter=Converter(), separator=BandSplitSeparator(config).eval())
    assert mix.shape == song.shape and mix.abs().max() <= 0.99
    assert "no model" in " ".join(report.notes) and "range:" in report.summary()


def test_music_and_tristream_cli(tmp_path, capsys):
    from helix_lm.cli import main

    song = write_wave(tmp_path / "song.wav", synthetic_song(8.0, SR, bpm=120, key=7), SR)
    reference = write_wave(tmp_path / "ref.wav", synthetic_voice(1.0, SR, f0=196.0), SR)
    assert main(["tristream", "sizes"]) == 0 and "ts-765m" in capsys.readouterr().out
    assert main(["music", "sources", "--commercial"]) == 0 and "vocalset" in capsys.readouterr().out
    assert main(["music", "analyze", str(song), "--sample-rate", str(SR), "--top", "1"]) == 0
    assert "G major" in capsys.readouterr().out
    plan = tmp_path / "plan.json"
    assert main(["music", "plan", str(song), "--sample-rate", str(SR), "--out", str(plan)]) == 0 and plan.exists()
    out = tmp_path / "swap.wav"
    args = ["music", "swap", str(song), str(reference), "--sample-rate", str(SR), "--out", str(out)]
    assert main(args) == 0 and out.exists() and "no model" in capsys.readouterr().out
