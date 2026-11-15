# Copyright 2026 Nathan. Apache-2.0.
"""The frontier benchmark's tables, on results small enough to check by hand."""

from pluvium.benchmarks import fullyear as F
from pluvium.frontier import report as R

GROUPS = list(F.LEAD_GROUPS)


def scores(bss50: float, bss1: float = 0.30, crps: float = 2.0) -> dict:
    return {**{t: {"bss": bss1, "csi": 0.4, "auc": 0.8} for t in ("1", "10", "25")}, "50": {"bss": bss50, "csi": 0.1, "auc": 0.7},
            "crps": crps, "rmse": 5.0, "acc": 0.5, "sharpness": 0.7}


def place(stack: float, declared: float, online: float, rival: float, staged: bool, yearly: list) -> dict:
    """A place's result: the three hand-over designs' skill at 50 mm, what its staged storms said (the rule that
    decides) and what the yearly check said at each refit (reported only)."""
    row = {"stack": scores(stack), "declared": scores(declared), "pool_online": scores(online), "pool": scores(online - 0.01),
           "pool_record": scores(declared - 0.01), "blend_ops": scores(0.9), "ens": scores(rival), "gfs": scores(rival - 0.05),
           "ens_raw": scores(rival - 0.2)}
    return {"cells": 100, "groups": {g: row for g in GROUPS},
            "seasons": {g: {s: row for s in F.SEASONS} for g in GROUPS}, "leads": {},
            "staged": {"keep": staged, "evidence": True, "gain": 0.05 if staged else -0.015, "storms": ["2024-10-08"]},
            "audits": {"stack": [{"keep": k, "rows": 1200, "days": 40, "edge_mm": 45.0, "without": 0.004,
                                  "handed": 0.00399 if k else 0.00401, "chance_of_harm": 0.8} for k in yearly]}}


def first(top: float, rival: float) -> dict:
    row = {"stack": scores(top), "ens": scores(rival), "gfs": scores(rival - 0.05)}
    return {"groups": {g: row for g in GROUPS}, "seasons": {g: {s: row for s in F.SEASONS} for g in GROUPS}}


# coast: the hand-over helps and the staged storms keep it (the yearly check would have dropped it).
# mountains: it hurts and the staged storms switch it off.
NEW = {"coast": place(stack=0.12, declared=0.12, online=0.10, rival=0.11, staged=True, yearly=[False] * 4),
       "mountains": place(stack=0.09, declared=0.07, online=0.09, rival=0.10, staged=False, yearly=[False] * 4)}
OLD = {"coast": first(0.08, 0.11), "mountains": first(0.085, 0.10)}


def test_best_rival_ignores_pluviums_own_variants():
    value, who = R.best_rival(NEW["coast"]["groups"][GROUPS[0]], "bss", "50", True)
    assert (value, who) == (0.11, "ens")                                   # not blend_ops at 0.9, not an ablation
    assert R.best_rival({"stack": scores(0.1)}, "bss", "50", True)[1] is None
    low, who = R.best_rival(NEW["coast"]["groups"][GROUPS[0]], "crps", None, False)
    assert low == 2.0 and who in ("ens", "gfs", "ens_raw")


def test_tally_counts_improvements_and_leads():
    c = R.tally(OLD, NEW, "bss", "50")
    assert c == {"cells": 8, "improved": 8, "new_leads": 4, "old_leads": 0}   # coast leads now; mountains still behind
    assert R.tally(OLD, NEW, "bss", "50", groups=[GROUPS[0]])["cells"] == 2
    assert R.tally(OLD, NEW, "bss", "50", season="DJF")["new_leads"] == 4
    assert R.tally({}, NEW, "bss", "50") == {"cells": 8, "improved": 0, "new_leads": 4, "old_leads": 0}


def test_before_after_marks_leads_and_gains():
    lines = R.before_after(OLD, NEW, "bss", "50")
    assert len(lines) == 4 and "+0.080 -> **+0.120** (+0.110) +" in lines[2]   # ahead (bold) and improved (+)
    assert "+0.085 -> +0.090 (+0.100) +" in lines[3] and "**" not in lines[3]


def test_handover_three_ways_and_each_rules_agreement_with_the_test_year():
    titles = {"coast": "Coast", "mountains": "Mountains"}
    lines = R.handover_three_ways(NEW, GROUPS[0], titles)
    assert lines[2] == "| Coast | +0.1000 | **+0.1200** | **+0.1200** | keep |"
    assert lines[3] == "| Mountains | **+0.0900** | +0.0700 | **+0.0900** | switch off |"
    assert lines[4] == "| **Mean** | +0.0950 | +0.0950 | +0.1050 | |"       # deciding by place beats both fixed choices
    # the staged storms were right in both places; the yearly check only in the mountains
    assert R.audit_agreement(NEW, GROUPS[0]) == (2, 2) and R.audit_agreement(NEW, GROUPS[0], ["coast"]) == (1, 1)
    assert R.audit_agreement(NEW, GROUPS[0], rule="yearly") == (1, 2)
    assert R.audit_misses(NEW, GROUPS[0]) == ""                            # right in both places: nothing to confess
    assert R.audit_misses(NEW, GROUPS[0], titles, rule="yearly") == (
        "The yearly check would have been wrong in the Coast it switched off a hand-over that then would have added "
        "0.0200 of Brier skill at 50 mm on days 1-3.")
    wrong = {"coast": place(0.10, 0.12, 0.10, 0.11, staged=False, yearly=[True] * 4)}   # it would have helped
    assert R.audit_agreement(wrong, GROUPS[0]) == (0, 1) and R.audit_agreement(wrong, GROUPS[0], rule="yearly") == (1, 1)
    assert R.audit_misses(wrong, GROUPS[0], titles).startswith("The verdict was wrong in the Coast it switched off")
    assert R.audit_outcomes(NEW, GROUPS[0], ["mountains"])[0][:2] == ("mountains", False)
    assert R.kept(NEW["mountains"]) is False and R.kept(NEW["coast"], "yearly") is False and R.kept({"groups": {}}) is None
    table = R.rules_table(NEW, GROUPS[0], titles)
    assert len(table) == 4
    assert table[2] == "| Coast | -5.0% | **keep** | +0.25% | switch off (at 0 of 4 refits: keep) | +0.0200 |"
    assert table[3] == "| Mountains | +1.5% | **switch off** | +0.25% | **switch off** (at 0 of 4 refits: keep) | -0.0200 |"
    assert R.rules_table({"x": {"groups": {GROUPS[0]: {}}}}) == []


def test_names_results_on_disk_and_the_rarity_paragraph(tmp_path):
    import json

    assert R.join_names(["alps"]) == "Alps" and R.join_names(["alps", "jamaica"]) == "Alps and Jamaica"
    assert R.join_names(["alps", "jamaica", "caribbean"]) == "Alps, Jamaica and Caribbean" and R.join_names([]) == "none"
    assert R.join_names(["alps", "jamaica", "great_plains"], the=True) == "the Alps, Jamaica and the US Great Plains"
    (tmp_path / "jamaica.json").write_text(json.dumps({"cells": 450}))
    (tmp_path / "alps_tail.json").write_text(json.dumps({"rows": 1}))
    assert list(R.load_results(tmp_path)) == ["jamaica"] and list(R.load_results(tmp_path, "_tail")) == ["alps"]
    assert set(R.SEEN_FIRST) | set(R.SCORED_AFTER) == set(R.PLACE_ORDER) and not set(R.SEEN_FIRST) & set(R.SCORED_AFTER)
    assert R.rarity_lines(tmp_path / "none.json") == []
    record = {"span": ["2020-01-01T00", "2025-04-01T00"], "years": 5.1, "horizon": 20.3, "places": 450,
              "level_mm": {"1": 79.0, "5": 122.8}, "days": {"2025-10-28": {"share_ge_1y": 1.0, "share_ge_5y": 0.78,
                                                                              "share_beyond_horizon": 0.35, "median_period_y": 12.7}}}
    (tmp_path / "returns.json").write_text(json.dumps(record))
    text = R.rarity_lines(tmp_path / "returns.json")[0]
    assert "2020-01 to 2025-04 (5.1 years, 450 places" in text and "once a year at 79 mm" in text
    assert "78% more than their once-in-five-years amount" in text and "35% an amount rarer than once in 20 years" in text
    assert R.rarity_lines(tmp_path / "returns.json", day="2025-01-01") == []


def test_ablation_and_stripping_ablations():
    lines = R.ablation(NEW, board=(("Brier skill, 50 mm", "bss", "50", True),))
    assert any(line.startswith("| Pool + online weights + hand-over where the staged storms say so (Pluvium Blend) | +0.1050")
               for line in lines)
    assert any(line.startswith("| Pool + online weights + hand-over, always on (as first declared) | +0.0950") for line in lines)
    bare = R.without(NEW)
    row = bare["coast"]["groups"][GROUPS[0]]
    assert "declared" not in row and "pool_online" not in row and "stack" in row and "ens" in row
    assert "declared" in NEW["coast"]["groups"][GROUPS[0]]                 # the original is untouched
