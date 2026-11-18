# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The benchmark sets: the synthetic set and its truth, the file names of the public sets, scores and manifests."""

import json

from helix_lm.bench.data import registry
from helix_lm.bench.data.manifest import build, load, same_items, save
from helix_lm.bench.data.opencpop import parse_line
from helix_lm.bench.data.pairs import conversion_pairs
from helix_lm.bench.data.scores import note_frames, note_name_to_midi
from helix_lm.bench.data.synthetic import make_set, truth
from helix_lm.bench.data.vocalset import TEST_SINGERS, is_test_singer, parse


def test_the_synthetic_set_writes_takes_with_their_truth(tmp_path):
    items = make_set(tmp_path, takes_per_singer=1, seconds=1.0)
    assert len(items) == 3 and all(it.path.exists() for it in items)
    t = truth(items[0])
    assert t["notes"] and len(t["f0"]) > 50
    again = make_set(tmp_path / "again", takes_per_singer=1, seconds=1.0)
    assert [it.notes for it in again] == [it.notes for it in items]          # the same seed: the same set


def test_licences_say_which_sets_a_commercial_model_may_be_measured_on():
    assert "synthetic" in registry.usable_for("svc", commercial_only=True)
    assert "m4singer" not in registry.usable_for("svs", commercial_only=True)
    assert registry.get("vocalset").licence.startswith("CC BY")


def test_vocalset_names_and_its_held_out_singers():
    info = parse(__import__("pathlib").Path("f2_arpeggios_belt_a.wav"))
    assert info and info["singer"] == "f2"
    assert all(is_test_singer(s) for s in TEST_SINGERS) and not is_test_singer("f1")


def test_score_parsing():
    assert note_name_to_midi("A4") == 69 and note_name_to_midi("C#4") == 61 and note_name_to_midi("x") is None
    midi, onset = note_frames([(0.0, 0.5, 60)], 100, 100.0)
    assert int((midi > 0).sum()) == 50 and float(midi[10]) == 60 and int(onset.sum()) == 1
    line = "2001000001|啊|a|A4|0.5|0.5|0"
    row = parse_line(line)
    assert row is None or row["id"] == "2001000001"


def test_pairs_never_pair_a_singer_with_itself(tmp_path):
    items = make_set(tmp_path, takes_per_singer=2, seconds=0.5)
    for src, ref in conversion_pairs(items, per_source=2):
        assert src.singer != ref.singer


def test_a_manifest_notices_a_changed_file(tmp_path):
    items = make_set(tmp_path, takes_per_singer=1, seconds=0.5)
    m = build(items, "synthetic")
    save(m, tmp_path / "m.json")
    assert same_items(load(tmp_path / "m.json"), m)
    items[0].path.write_bytes(b"changed")
    assert not same_items(build(items, "synthetic"), m)
    assert json.loads((tmp_path / "m.json").read_text())["name"] == "synthetic"
