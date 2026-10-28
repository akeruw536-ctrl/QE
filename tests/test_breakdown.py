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


def _add(s, tid, edge, regime, stake, result, pnl, width=47.0, price=28000.0, shadow=0):
    s.record_trade({"trade_id": tid, "symbol": "1HZ90V", "stake": stake, "payout": stake * 1.3, "edge": edge,
                    "regime": regime, "entry_price": price, "lower_barrier": price - width / 2,
                    "upper_barrier": price + width / 2, "opened_at": time.time() - 500,
                    "expires_at": time.time() - 300, "shadow": shadow})
    s.settle_trade(tid, result, pnl)


def test_breakdown_buckets_win_rate_pnl_and_roi():
    s = _db()
    _add(s, "a", 0.02, "NORMAL_RANGE", 1.0, "WIN", 0.3)
    _add(s, "b", 0.02, "NORMAL_RANGE", 1.0, "LOSS", -1.0)
    _add(s, "c", 0.05, "LOW_VOLATILITY_RANGE", 2.5, "WIN", 0.75)
    _add(s, "d", 0.12, "NORMAL_RANGE", 6.25, "LOSS", -6.25)
    _add(s, "e", None, "NORMAL_RANGE", 1.0, "WIN", 0.3)
    _add(s, "s", 0.02, "NORMAL_RANGE", 1.0, "LOSS", -1.0, shadow=1)       # other mode: excluded
    b = s.breakdown(shadow=False)
    edge = {x["bucket"]: x for x in b["by_claimed_edge"]}
    assert edge["<0.03"]["n"] == 2 and edge["<0.03"]["win_rate"] == 0.5 and edge["<0.03"]["pnl"] == -0.7
    assert edge["0.03-0.06"]["n"] == 1 and edge[">=0.10"]["roi"] == -1.0 and edge["unknown"]["n"] == 1
    reg = {x["bucket"]: x["n"] for x in b["by_regime"]}
    assert reg == {"LOW_VOLATILITY_RANGE": 1, "NORMAL_RANGE": 4}
    stake = {x["bucket"]: x["n"] for x in b["by_stake"]}
    assert stake == {"<=1.0": 3, "1.0-3.0": 1, "3.0-10": 1}
    assert b["by_barrier_width"][0]["n"] == 5                              # 47/28000 = 0.17% -> "<0.3%"


def test_breakdown_on_an_empty_journal():
    b = _db().breakdown(shadow=False)
    assert b == {"by_claimed_edge": [], "by_regime": [], "by_stake": [], "by_barrier_width": []}
