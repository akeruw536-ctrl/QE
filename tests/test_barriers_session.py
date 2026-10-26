import math

from app.strategy.barriers import base_offsets, candidate_barriers
from app.strategy.risk import session_status

FIXED = {"1HZ90V": (41.457, 41.31)}


def test_fixed_offsets_are_the_xml_values_rounded_to_2dp():
    ch = candidate_barriers("1HZ90V", "auto", FIXED, 1.0, 0.001, 28000.0, 2, [1.0])
    assert len(ch) == 1 and ch[0].source == "fixed"
    assert ch[0].upper_offset == 41.46 and ch[0].lower_offset == -41.31


def test_auto_uses_vol_scaling_for_unknown_symbols():
    up, lo, src = base_offsets("R_100", "auto", FIXED, 1.0, 0.001, 1000.0, 2)
    assert src == "vol" and math.isclose(up, 0.001 * math.sqrt(2) * 1000.0) and up == lo


def test_vol_mode_at_1hz90v_sigma_reproduces_the_xml_width():
    up, _, _ = base_offsets("1HZ90V", "vol", FIXED, 1.0, 0.00104, 28150.0, 2)
    assert 40.0 < up < 43.0      # ~ +41.457 in the XML


def test_scales_and_dedupe_and_bad_input():
    ch = candidate_barriers("1HZ90V", "fixed", FIXED, 1.0, 0.0, 28000.0, 2, [0.9, 1.0, 1.0, 1.3])
    assert [c.scale for c in ch] == [0.9, 1.0, 1.3]
    assert ch[0].upper_offset < ch[1].upper_offset < ch[2].upper_offset
    assert candidate_barriers("R_100", "vol", FIXED, 1.0, 0.0, 1000.0, 2, [1.0]) == []
    assert candidate_barriers("R_100", "fixed", FIXED, 1.0, 0.001, 1000.0, 2, [1.0]) == []


def test_session_status_matches_the_xml_checks():
    assert session_status(61.99, 62, 200) == "OK"
    assert session_status(62.0, 62, 200) == "TARGET"
    assert session_status(-200.0, 62, 200) == "STOP"
    assert session_status(-199.99, 62, 200) == "OK"
    assert session_status(1e6, 0, 0) == "OK"          # 0 disables both
