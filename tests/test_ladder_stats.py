import os
import tempfile
import time

from app.data.storage import Storage


class _Log:
    def info(self, *a, **k): pass
    warning = error = debug = info


def _db():
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    s = Storage("", path, _Log()); s.init_schema()
    return s


def _add(s, tid, stake, result, pnl, shadow=0):
    s.record_trade({"trade_id": tid, "symbol": "1HZ90V", "stake": stake, "payout": stake * 1.3,
                    "opened_at": 1_700_000_000 + len(tid), "expires_at": time.time() - 300, "shadow": shadow})
    if result:
        s.settle_trade(tid, result, pnl)


def test_levels_streaks_max_stake_and_drawdown():
    s = _db()
    seq = [("w0", 1.0, "WIN", 0.3), ("l1", 1.0, "LOSS", -1.0), ("l2", 2.5, "LOSS", -2.5),
           ("l3", 6.25, "LOSS", -6.25), ("w4", 15.62, "WIN", 4.0), ("l5", 1.0, "LOSS", -1.0)]
    for t in seq:
        _add(s, *t)
    st = s.ladder_stats(shadow=False, martingale_start=1, martingale_level=5)
    assert st["max_martingale_level"] == 3          # w4 was staked after 3 straight losses
    assert st["max_loss_streak"] == 3 and st["current_loss_streak"] == 1
    assert st["max_stake"]["stake"] == 15.62 and st["max_stake"]["level"] == 3 and st["max_stake"]["result"] == "WIN"
    assert st["max_drawdown"] == 9.75               # peak 0.3 -> trough 0.3-1-2.5-6.25 = -9.45
    assert st["trades_per_level"] == {"0": 3, "1": 1, "2": 1, "3": 1}


def test_trades_per_level_and_level_reset_after_exhaustion():
    s = _db()
    for i in range(8):                                # 8 straight losses with level cap 3
        _add(s, f"x{i}", 1.0, "LOSS", -1.0)
    st = s.ladder_stats(shadow=False, martingale_start=1, martingale_level=3)
    assert st["max_martingale_level"] == 3 and st["max_loss_streak"] == 8
    assert st["trades_per_level"] == {"0": 5, "1": 1, "2": 1, "3": 1}   # entries after 4+ losses reset to level 0


def test_open_trades_count_for_max_stake_but_not_streaks_and_modes_are_separate():
    s = _db()
    _add(s, "a", 1.0, "LOSS", -1.0)
    _add(s, "b", 9.0, None, 0.0)                      # still open
    _add(s, "c", 99.0, "LOSS", -99.0, shadow=1)       # other mode
    st = s.ladder_stats(shadow=False)
    assert st["max_stake"]["stake"] == 9.0 and st["max_stake"]["result"] == "OPEN" and st["max_stake"]["level"] is None
    assert st["max_loss_streak"] == 1 and st["max_drawdown"] == 1.0
    assert s.ladder_stats(shadow=True)["max_stake"]["stake"] == 99.0


def test_empty_journal():
    st = _db().ladder_stats(shadow=False)
    assert st["max_stake"] is None and st["max_martingale_level"] == 0 and st["max_drawdown"] == 0.0
