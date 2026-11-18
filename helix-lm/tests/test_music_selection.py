# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Choosing checkpoints and training weights from benchmark scores: the composite, checkpoint selection, hard
examples, weak registers and the held-out validation pack."""

import math

from helix_lm.music.composite import ANCHORS, composite, from_summary, scaled
from helix_lm.music.hard_examples import hardest, weights
from helix_lm.music.selection import choose, record
from helix_lm.music.validation_pack import choose as pack
from helix_lm.music.validation_pack import load, save
from helix_lm.music.weak_spots import group_weights, register_bucket


def test_the_composite_is_one_at_every_target_and_zero_at_every_floor():
    targets = {m: a[1] for m, a in ANCHORS.items()}
    floors = {m: a[0] for m, a in ANCHORS.items()}
    assert math.isclose(composite(targets), 1.0, abs_tol=1e-9) and math.isclose(composite(floors), 0.0, abs_tol=1e-9)
    assert 0.0 <= scaled("mcd_db", 1e9) <= 1.0
    assert math.isnan(composite({}))
    summary = {"tristream": {"svc": {m: [v, v, v, 5] for m, v in targets.items()}}}
    assert math.isclose(from_summary(summary, "tristream"), 1.0, abs_tol=1e-9)


def test_a_checkpoint_is_kept_only_when_it_wins_by_the_margin(tmp_path):
    a = {"summary": {"tristream": {"svc": {"mcd_db": [6.0, 6, 6, 5]}}}}
    b = {"summary": {"tristream": {"svc": {"mcd_db": [5.95, 5.95, 5.95, 5]}}}}
    c = {"summary": {"tristream": {"svc": {"mcd_db": [4.0, 4, 4, 5]}}}}
    picked = choose([("a", a), ("b", b)], current="a")
    assert picked[0] == "a"                                                # a hair better is not a change
    assert choose([("a", a), ("c", c)], current="a")[0] == "c"
    record(tmp_path / "picks.jsonl", "c", {"composite": 0.5})
    assert (tmp_path / "picks.jsonl").read_text().strip()


def test_hard_examples_get_more_weight_but_not_without_limit():
    rows = [{"item": f"i{k}", "mcd_db": float(k)} for k in range(10)]
    w = weights(rows)
    assert w["i9"] > w["i0"] and max(w.values()) <= 3.0
    assert hardest(rows, n=2) == ["i9", "i8"]


def test_weak_registers_get_more_training_weight():
    assert register_bucket(1000.0) != register_bucket(100.0)
    rows = [{"register": "high", "mcd_db": 9.0}, {"register": "low", "mcd_db": 4.0}]
    g = group_weights(rows, "register")
    assert g["high"] > g["low"]


def test_the_validation_pack_holds_only_held_out_singers_and_round_trips(tmp_path):
    records = [{"path": f"s{k % 5}_{k}.wav", "tags": {"singer": f"s{k % 5}"}, "accepted": k % 7 != 0}
               for k in range(50)]
    chosen = pack(records, held_singers={"s1", "s2"}, per_singer=3)
    assert len(chosen) == 6 and {r["tags"]["singer"] for r in chosen} == {"s1", "s2"}
    assert all(r["accepted"] for r in chosen)
    save(chosen, tmp_path / "pack.json")
    back = load(tmp_path / "pack.json")
    assert [r["path"] for r in back] == [r["path"] for r in chosen]
    assert [r["singer"] for r in back] == [r["tags"]["singer"] for r in chosen]
