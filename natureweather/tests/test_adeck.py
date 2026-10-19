"""The NHC guidance: deck parsing, interpolation to the leads, the fitted blends and the RI-probability readers."""

import numpy as np
import pytest

from naturev1.cyclone.aids import blend_intensity, blend_track, resolve
from naturev1.cyclone.geometry import great_circle_km
from naturev1.swarm.adeck import aids_at, interpolate, parse_adeck, parse_edeck_ri, parse_ships_ri

ADECK = """AL, 13, 2025102700, 03, OFCL,   0, 164N,  773W, 125,    0, HU,  34, NEQ,  170,  110,   60,   90,
AL, 13, 2025102700, 03, OFCL,   0, 164N,  773W, 125,    0, HU,  50, NEQ,   50,   40,   30,   40,
AL, 13, 2025102700, 03, OFCL,  12, 165N,  780W, 135,    0, HU,  34, NEQ,  170,  110,   60,   90,
AL, 13, 2025102700, 03, OFCL,  24, 171N,  782W, 140,    0, HU,  34, NEQ,  170,  110,   60,   90,
AL, 13, 2025102700, 03, HCCA,   0, 164N,  773W, 125,    0,   ,   0,    ,    0,    0,    0,    0,
AL, 13, 2025102700, 03, HCCA,  12, 165N,  779W, 127,    0,   ,   0,    ,    0,    0,    0,    0,
AL, 13, 2025102700, 03, HCCA,  24, 170N,  782W, 118,    0,   ,   0,    ,    0,    0,    0,    0,
AL, 13, 2025102700, 03, DSHP,   0, 164N,  772W, 125,    0, HU,   0,    ,    0,    0,    0,    0,
AL, 13, 2025102700, 03, DSHP,  12, 166N,  778W, 127,    0, HU,   0,    ,    0,    0,    0,    0,
AL, 13, 2025102700, 03, DSHP,  24, 169N,  781W,   0,    0, HU,   0,    ,    0,    0,    0,    0,
AL, 13, 2025102700, 03, CARQ, -12, 160N,  770W, 110,  950, HU,  34, NEQ,  170,  110,   60,   90,
AL, 13, 2025102700, 03, CARQ,   0, 164N,  773W, 125,  933, HU,  34, NEQ,  170,  110,   60,   90,
AL, 13, 2025102700, 03, AVNI,   0, 164N,  773W, 125,    0,   ,  34, NEQ,  180,  120,   50,   80,
AL, 13, 2025102706, 03, OFCL,   0, 165N,  777W, 135,    0, HU,  34, NEQ,  170,  110,   60,   90,
"""

SHIPS = """Matrix of RI probabilities
------------------------------------------------------------------------------
  RI (kt / h)  | 20/12 | 25/24 | 30/24 | 35/24 | 40/24 | 45/36 | 55/48  |65/72
------------------------------------------------------------------------------
   SHIPS-RII:     0.0%   10.1%    0.0%    0.0%    0.0%    0.0%    0.0%    0.0%
    Logistic:     0.1%    0.3%    0.2%    0.0%    0.0%    0.0%    0.0%    0.0%
    Bayesian:     0.0%    0.0%    0.0%    0.0%    0.0%    0.0%    0.0%    0.0%
   Consensus:     0.0%    3.4%   12.5%    0.0%    0.0%    0.0%    0.0%    0.0%
       DTOPS:     0.0%    0.0%    0.0%    0.0%    0.0%    0.0%    0.0%    0.0%
       SDCON:     0.0%    1.7%    0.0%    0.0%    0.0%    0.0%    0.0%    0.0%

   ##         ANNULAR HURRICANE INDEX (AHI) EP212026 TRUDY      10/19/26  12 UTC         ##
"""

EDECK = """AL, 13, 2025101812, RI, RIOC,  12, 117N,  498W,   3,  20,  50,    ,            0,           12,
AL, 13, 2025101812, RI, RIOC,  24, 117N,  498W,  12,  25,  55,    ,            0,           24,
AL, 13, 2025101812, RI, RIOC,  24, 117N,  498W,   7,  30,  60,    ,            0,           24,
AL, 13, 2025101812, RI, RIOD,  24, 117N,  498W,  14,  30,  60,    ,            0,           24,
AL, 13, 2025101812, IN, IVCN,  24,     ,      ,  68,  34,   ,    7,
"""


class _Case:
    lat, lon = 16.4, -77.3


def test_the_a_deck_is_parsed_one_row_per_tau_with_missing_values_as_nan():
    deck = parse_adeck(ADECK)
    t0 = np.datetime64("2025-10-27T00:00")
    assert set(deck) == {t0, np.datetime64("2025-10-27T06:00")}
    assert set(deck[t0]) == {"OFCL", "HCCA", "DSHP", "CARQ", "AVNI"}
    assert deck[t0]["OFCL"][0] == (16.4, -77.3, 125.0, pytest.approx(float("nan"), nan_ok=True))
    assert deck[t0]["OFCL"][24][2] == 140.0 and len(deck[t0]["OFCL"]) == 3        # the 50-kt radii row did not overwrite tau 0
    assert np.isnan(deck[t0]["DSHP"][24][2]) and deck[t0]["DSHP"][24][0] == 16.9   # a zero wind is missing, the position stays
    assert deck[t0]["CARQ"][-12][3] == 950.0 and deck[t0]["CARQ"][0][3] == 933.0


def test_aids_are_interpolated_to_the_leads_and_never_extrapolated():
    deck = parse_adeck(ADECK)
    aids = aids_at(deck, "2025-10-27T00", [6, 12, 24, 36])
    assert aids["available"] == ["DSHP", "HCCA", "OFCL"]                            # AVNI has one tau, CARQ nothing ahead
    ofcl = aids["track"]["OFCL"]
    assert ofcl[0] == pytest.approx([16.45, -77.65]) and ofcl[2] == pytest.approx([17.1, -78.2])
    assert np.isnan(ofcl[3]).all()                                                  # 36 h beyond the last tau
    assert aids["intensity"]["OFCL"][0] == pytest.approx(130.0)
    assert np.isnan(aids["intensity"]["DSHP"][2]) and aids["intensity"]["DSHP"][0] == pytest.approx(126.0)
    assert aids_at(deck, "2025-10-27T12", [6, 12])["available"] == []


def test_interpolation_crosses_the_dateline():
    aid = {0: (10.0, 179.5, 50.0, np.nan), 12: (10.0, -179.5, 60.0, np.nan)}
    v = interpolate(aid, [6])
    assert v[0, 1] == pytest.approx(180.0) or v[0, 1] == pytest.approx(-180.0)
    assert v[0, 2] == pytest.approx(55.0)


def test_track_blend_is_the_midpoint_in_kilometres_and_falls_back_cleanly():
    case = _Case()
    leads = [6, 12, 24]
    ours = np.array([[16.5, -77.6], [16.7, -78.0], [17.2, -78.4]])
    hcca = np.array([[16.5, -77.6], [16.6, -77.8], [16.8, -78.0]])
    table = {"members": ["ours", "HCCA", "TVCN"], "leads": [6, 12, 24], "weights": [[0.5, 0.5, 0.0]] * 3, "substitutes": {"HCCA": ["TVCN"]}}
    out, used = blend_track(case, ours, {"HCCA": hcca}, leads, table)
    assert used["members"] == ["HCCA"] and used["substituted"] == {}
    d_ours = great_circle_km(out[2, 0], out[2, 1], ours[2, 0], ours[2, 1])
    d_hcca = great_circle_km(out[2, 0], out[2, 1], hcca[2, 0], hcca[2, 1])
    assert d_ours == pytest.approx(d_hcca, rel=0.02) and d_ours > 20
    # HCCA absent: TVCN stands in, and that is reported
    out2, used2 = blend_track(case, ours, {"TVCN": hcca}, leads, table)
    assert used2["substituted"] == {"HCCA": "TVCN"} and np.allclose(out2, out)
    # no aid the table knows: unchanged, None
    out3, used3 = blend_track(case, ours, {"AVNI": hcca}, leads, table)
    assert used3 is None and out3 is ours or np.array_equal(out3, ours)
    # an aid missing at one lead: that lead is ours alone
    partial = hcca.copy(); partial[2] = np.nan
    out4, _ = blend_track(case, ours, {"HCCA": partial}, leads, table)
    assert np.allclose(out4[2], ours[2]) and np.allclose(out4[1], out[1])


def test_intensity_blend_renormalises_over_the_members_present():
    leads = [6, 12, 24]
    table = {"members": ["ours", "IVCN", "HWFI"], "leads": [6, 12, 24], "weights": [[0.5, 0.25, 0.25]] * 3, "substitutes": {"HWFI": ["HFAI"]}}
    ours = np.array([100.0, 110.0, 120.0])
    out, used = blend_intensity(ours, {"IVCN": np.array([100.0, 100.0, 100.0]), "HFAI": np.array([100.0, 130.0, np.nan])}, leads, table)
    assert used["members"] == ["IVCN", "HWFI"] and used["substituted"] == {"HWFI": "HFAI"}
    assert out[0] == pytest.approx(100.0)
    assert out[1] == pytest.approx(0.5 * 110 + 0.25 * 100 + 0.25 * 130)
    assert out[2] == pytest.approx((0.5 * 120 + 0.25 * 100) / 0.75)                 # HWFI missing at 24 h: weights renormalised
    same, none = blend_intensity(ours, {}, leads, table)
    assert none is None and np.array_equal(same, ours)
    chosen, subs = resolve({"HFBI": np.array([1.0])}, ["ours", "HMNI"], {"HMNI": ["HFAI", "HFBI"]})
    assert list(chosen) == ["HMNI"] and subs == {"HMNI": "HFBI"}


def test_the_shipped_tables_are_in_the_coefficients_and_sum_to_one():
    from naturev1.cyclone import coefficients

    for name in ("ADECK_TRACK", "ADECK_INTENSITY"):
        table = getattr(coefficients, name)
        W = np.asarray(table["weights"])
        assert W.shape == (len(table["leads"]), len(table["members"])) and table["members"][0] == "ours"
        assert np.allclose(W.sum(1), 1.0, atol=1e-3) and (W >= 0).all()      # rounded to 4 decimals; the blend renormalises
    assert coefficients.ADECK_TRACK["weights"][2] == pytest.approx([0.5, 0.5, 0, 0, 0])
    assert coefficients.ADECK_INTENSITY["substitutes"] == {"HWFI": ["HFAI"], "HMNI": ["HFBI"]}


def test_the_ships_text_matrix_and_the_e_deck_rows_give_the_30_kt_24_h_probability():
    table = parse_ships_ri(SHIPS)
    assert set(table) == {"SHIPS-RII", "Logistic", "Bayesian", "Consensus", "DTOPS", "SDCON"}
    assert table["Consensus"]["30/24"] == pytest.approx(0.125) and table["SHIPS-RII"]["25/24"] == pytest.approx(0.101)
    assert parse_ships_ri("no matrix here") is None
    ri = parse_edeck_ri(EDECK)[np.datetime64("2025-10-18T12:00")]
    assert ri["RIOC"][(24, 30.0)] == pytest.approx(0.07) and ri["RIOC"][(12, 20.0)] == pytest.approx(0.03)
    assert ri["RIOD"][(24, 30.0)] == pytest.approx(0.14) and "IVCN" not in ri


def test_weights_are_earned_from_the_previous_cycle_and_the_disagreement_is_reported():
    from naturev1.cyclone.aids import adaptive_weights, previous_errors

    class Now:
        lat, lon, time = 17.0, -78.0, np.datetime64("2025-10-27T00:00")

    leads = [6, 12, 24]
    # the previous cycle (12 h earlier) said: ours would be exactly here, HCCA 1 degree of longitude east
    prev = {"ours": np.array([[16.7, -77.9], [17.0, -78.0], [17.6, -78.2]]), "HCCA": np.array([[16.7, -77.4], [17.0, -77.0], [17.6, -77.2]])}
    e = previous_errors(Now(), np.datetime64("2025-10-26T12:00"), prev, leads)
    assert e["ours"] == pytest.approx(0.0, abs=1e-6) and 100 < e["HCCA"] < 110
    assert previous_errors(Now(), np.datetime64("2025-10-24T12:00"), prev, leads) == {}        # too long ago
    table = {"members": ["ours", "HCCA", "TVCN"], "leads": leads, "weights": [[0.5, 0.5, 0.0]] * 3,
             "adaptive": {"b": [0.5, 0.5, 0.0], "cap": 0.75, "eps_km": 5.0}}
    w = adaptive_weights(table, leads, e)
    assert w[0] == pytest.approx(w[1]) and 0.5 < w[0] <= 0.75 and w[2] == pytest.approx(0.5)      # earned at 6-12 h, fixed at 24 h
    assert w[0] == pytest.approx(min(0.75, 1 / (1 + np.exp(-0.5 * np.log((e["HCCA"] + 5) / 5)))))
    assert adaptive_weights(table, leads, {"ours": 10.0}) is None and adaptive_weights({**table, "adaptive": None}, leads, e) is None
    ours = np.array([[17.3, -78.1], [17.6, -78.2], [18.2, -78.4]])
    hcca = np.array([[17.3, -78.1], [17.6, -77.9], [18.2, -77.6]])
    fixed, used_fixed = blend_track(Now(), ours, {"HCCA": hcca}, leads, table)
    earned, used = blend_track(Now(), ours, {"HCCA": hcca}, leads, table, previous_errors=e)
    assert "adaptive_w_ours" not in used_fixed and np.allclose(used["adaptive_w_ours"], w)
    # at 12 h the earned blend sits nearer ours than the fixed mean does; at 24 h (b = 0) they coincide
    assert abs(earned[1, 1] - ours[1, 1]) < abs(fixed[1, 1] - ours[1, 1]) and np.allclose(earned[2], fixed[2])
    dis = used["disagreement_km"]
    assert dis[0] == pytest.approx(0.0, abs=1e-6) and 25 < dis[1] < 40 and 75 < dis[2] < 95
    assert used["previous_errors_km"] == e


def test_the_shipped_adaptive_entry_is_the_one_chosen_on_the_tuning_seasons():
    from naturev1.cyclone import coefficients

    a = coefficients.ADECK_TRACK["adaptive"]
    assert a["b"] == [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0] and a["cap"] == 0.75 and a["eps_km"] == 5.0


def test_the_gate_moves_weight_away_from_the_member_that_stands_alone_and_only_beyond_the_cone():
    from naturev1.cyclone.aids import outlier_gate

    class Now:
        lat, lon = 17.0, -78.0

    leads = [6, 12, 24]
    table = {"members": ["ours", "HCCA", "TVCN"], "leads": leads, "weights": [[0.5, 0.5, 0.0]] * 3,
             "gate": {"gamma": 0.1, "start_cone_radii": 1.5, "min_lead_h": 12, "tiebreaker": "TVCN"}}
    ours = np.array([[17.3, -78.1], [17.6, -78.2], [18.2, -79.4]])        # at 24 h ours sits ~1 degree west of the other two
    hcca = np.array([[17.3, -78.1], [17.6, -77.9], [18.2, -78.0]])
    tvcn = np.array([[17.3, -78.1], [17.6, -77.95], [18.2, -78.1]])
    cone = np.array([20.0, 30.0, 50.0])
    out, used = blend_track(Now(), ours, {"HCCA": hcca, "TVCN": tvcn}, leads, table, cone_km=cone)
    shift = used["gate_shift"]
    assert shift[0] == 0.0                                                # 6 h: below min_lead_h
    assert shift[1] == 0.0                                                # 12 h: ~30 km apart, 1 cone radius, below the start
    assert shift[2] == pytest.approx(-0.1)                                # 24 h: ~150 km apart (3 radii), ours the outlier: the full 10% away from ours
    assert used["weight_share"]["ours"] == pytest.approx((0.5 + 0.5 + 0.4) / 3)
    # HCCA the outlier: the shift goes the other way
    out2, used2 = blend_track(Now(), ours, {"HCCA": np.array([[17.3, -78.1], [17.6, -77.9], [18.2, -76.6]]), "TVCN": np.array([[17.3, -78.1], [17.6, -77.95], [18.2, -79.3]])}, leads, table, cone_km=cone)
    assert used2["gate_shift"][2] == pytest.approx(+0.1)
    # no cone, no tie-breaker: no gate, the plain split
    _, used3 = blend_track(Now(), ours, {"HCCA": hcca, "TVCN": tvcn}, leads, table)
    _, used4 = blend_track(Now(), ours, {"HCCA": hcca}, leads, table, cone_km=cone)
    assert "gate_shift" not in used3 and "gate_shift" not in used4
    assert outlier_gate({**table, "gate": None}, leads, np.zeros((3, 3, 2)), cone) is None


def test_the_shipped_gate_entry():
    from naturev1.cyclone import coefficients

    g = coefficients.ADECK_TRACK["gate"]
    assert g["gamma"] == 0.1 and g["start_cone_radii"] == 1.5 and g["min_lead_h"] == 12 and g["tiebreaker"] == "TVCN"
