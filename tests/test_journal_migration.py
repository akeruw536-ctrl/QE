import os
import subprocess
import tempfile
import time

import pytest

from app.data.storage import Storage
from tools.import_journal import EXPORT_ONE_LINER, decode_export, import_journal


class _Log:
    def info(self, *a, **k): pass
    warning = error = debug = info


def _db():
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    s = Storage("", path, _Log()); s.init_schema()
    return s, path


def _populate(s):
    now = time.time()
    for i, (res, pnl) in enumerate([("WIN", 0.3), ("LOSS", -1.0), ("WIN", 0.75), (None, 0.0)]):
        s.record_trade({"trade_id": f"t{i}", "symbol": "1HZ90V", "stake": 1.0 + i, "payout": 1.3 + i,
                        "raw_probability": 0.79, "calibrated_probability": 0.78, "edge": 0.03 + i / 100,
                        "regime": "NORMAL_RANGE", "opened_at": now - 600 + i, "expires_at": now - 400 + i,
                        "shadow": 0, "contract_id": f"C{i}"})
        if res:
            s.settle_trade(f"t{i}", res, pnl)
    s.update_calibration("0.75-0.80", True)
    s.update_calibration("0.75-0.80", False)


def _export(path):
    cmd = EXPORT_ONE_LINER.replace("/app/data/quantumends.db", path)
    return subprocess.run(["sh", "-c", cmd], capture_output=True, text=True, check=True).stdout


def test_round_trip_preserves_trades_open_position_and_calibration():
    src, path = _db(); _populate(src)
    payload = decode_export(_export(path))
    assert len(payload["trades"]) == 4 and payload["calibration_buckets"]

    dst, _ = _db()
    counts = import_journal(dst, payload)
    assert counts["trades_added"] == 4 and counts["trades_skipped"] == 0

    a, b = src.performance_summary(shadow=False), dst.performance_summary(shadow=False)
    assert a == b and b["settled"] == 3
    assert [t["trade_id"] for t in dst.open_trades(shadow=False)] == ["t3"]            # re-attachable after restart
    assert dst.open_trades(shadow=False)[0]["contract_id"] == "C3"
    assert dst.ladder_stats(False) == src.ladder_stats(False)
    assert dst.consecutive_losses(None, shadow=False) == src.consecutive_losses(None, shadow=False)
    assert dst.get_calibration() == src.get_calibration() and src.get_calibration()   # bucket counts carried over


def test_import_refuses_a_non_empty_target_and_force_is_idempotent():
    src, path = _db(); _populate(src)
    payload = decode_export(_export(path))
    dst, _ = _db()
    import_journal(dst, payload)
    with pytest.raises(SystemExit):
        import_journal(dst, payload)
    again = import_journal(dst, payload, force=True)
    assert again["trades_added"] == 0 and again["trades_skipped"] == 4
    assert dst.performance_summary(shadow=False)["settled"] == 3


def test_decode_tolerates_terminal_paste_noise():
    src, path = _db(); _populate(src)
    raw = _export(path)
    noisy = "$ python3 -c ...\n" + raw.replace("\n", "\r\n") + "\n$ exit\n"
    assert len(decode_export(noisy)["trades"]) == 4
