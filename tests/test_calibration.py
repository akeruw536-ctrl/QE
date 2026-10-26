import tempfile
import os

from app.data.storage import Storage
from app.models.calibration import CalibrationTracker, bucket_for


def make_storage():
    # a real temp file, not ":memory:" -- SQLAlchemy opens a fresh connection
    # per `with engine.begin()`, and an in-memory sqlite db does not persist
    # across separate connections without pinning a StaticPool.
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Storage(database_url="", sqlite_path=path, logger=_NullLogger())
    s.init_schema()
    return s


class _NullLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


def test_bucket_for_boundaries():
    assert bucket_for(0.71) == "0.70-0.75"
    assert bucket_for(0.0) == "0.00-0.05"


def test_calibration_shrinks_toward_raw_with_no_data():
    storage = make_storage()
    tracker = CalibrationTracker(storage)
    calibrated = tracker.calibrate(0.80)
    assert calibrated == 0.80


def test_calibration_pulls_toward_observed_rate_with_evidence():
    storage = make_storage()
    tracker = CalibrationTracker(storage, min_observations_for_full_trust=10)
    # simulate a bucket that actually wins less than predicted
    for i in range(10):
        tracker.record_outcome(0.80, won=(i < 5))  # 50% observed vs 80% predicted
    calibrated = tracker.calibrate(0.80)
    assert calibrated < 0.80
