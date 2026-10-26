import math

from app.strategy.signals import (DOWN, UP, TickSignalDetector, candle_confirms, direction_allowed,
                                  ema_series, is_black, is_white, last_closed_candle, reference_candle)


def test_ema_series_sma_seed_and_recursion():
    vals = [1, 2, 3, 4, 5, 6]
    out = ema_series(vals, 3)
    assert out[0] == 2.0                       # SMA of first 3
    k = 2 / 4
    assert math.isclose(out[1], 2.0 + k * (4 - 2.0))
    assert len(out) == 4
    assert ema_series([1, 2], 3) == []


def test_incremental_matches_series():
    prices = [100 + math.sin(i / 3) * 2 + (i % 5) * 0.1 for i in range(120)]
    det = TickSignalDetector(7, 14, require_tick_confirmation=False)
    det.seed(prices[:60])
    for p in prices[60:]:
        det.update(p)
    assert math.isclose(det.ema_fast, ema_series(prices, 7)[-1], rel_tol=1e-9)
    assert math.isclose(det.ema_slow, ema_series(prices, 14)[-1], rel_tol=1e-9)


def _rising_then_falling():
    return [100 + i * 0.5 for i in range(40)]


def test_cross_down_detected_exactly_once_with_tick_fall():
    det = TickSignalDetector(7, 14)
    det.seed(_rising_then_falling())
    signals = []
    p = 119.5
    for _ in range(40):
        p -= 1.5
        s = det.update(p)
        if s:
            signals.append(s)
    assert signals == [DOWN]


def test_tick_confirmation_blocks_a_cross_on_a_rising_tick():
    # force a state where fast is just above slow, then a tiny UP tick can't cross down
    det = TickSignalDetector(7, 14, require_tick_confirmation=True)
    det.seed(_rising_then_falling())
    det.ema_fast, det.ema_slow = 100.0, 99.9999      # fast barely above slow
    det.last_price = 100.0
    # a rising price keeps fast above slow: no signal at all
    assert det.update(100.5) is None


def test_cross_up_and_direction_filter():
    det = TickSignalDetector(7, 14)
    det.seed([120 - i * 0.5 for i in range(40)])
    got = []
    p = 100.5
    for _ in range(40):
        p += 1.5
        s = det.update(p)
        if s:
            got.append(s)
    assert got == [UP]
    assert direction_allowed(DOWN, "down") and not direction_allowed(UP, "down")
    assert direction_allowed(UP, "both") and direction_allowed(UP, "up")


def test_not_ready_until_enough_history_then_feed_warms_up():
    det = TickSignalDetector(7, 14)
    assert det.update(100) is None and not det.ready
    for i in range(14):
        assert det.feed(100 + i * 0.1) is None
    assert not det.ready
    det.feed(101.5)
    assert det.ready


def test_candle_helpers():
    cs = [{"epoch": 0, "open": 10, "high": 11, "low": 9, "close": 9.5},
          {"epoch": 60, "open": 9.5, "high": 10, "low": 9, "close": 9.8},
          {"epoch": 120, "open": 9.8, "high": 10, "low": 9.7, "close": 9.9}]   # forming at now=150
    assert last_closed_candle(cs, now=150, granularity_seconds=60)["epoch"] == 60
    assert reference_candle(cs, 150, 60, "forming")["epoch"] == 120
    assert reference_candle(cs, 150, 60, "closed")["epoch"] == 60
    assert is_black(cs[0]) and not is_white(cs[0])
    assert is_white(cs[1])
    assert candle_confirms(DOWN, cs[0]) and not candle_confirms(DOWN, cs[1])
    assert candle_confirms(UP, cs[1])
    assert not candle_confirms(DOWN, None)
    assert last_closed_candle([], 10, 60) is None
