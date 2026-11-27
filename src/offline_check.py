"""Offline kill-switch self-test: blocks every network connection, then runs the full pipeline
(forecast -> allocation -> overflow routing -> simulator -> dispatch notices). Exit code 0 = passed.
Run:  python src/offline_check.py"""
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent


class NetworkBlocked(RuntimeError):
    pass


def run_offline_check():
    import pandas as pd
    import ops
    from export import build_all
    from pipeline import load_weather
    from predict import load_bundle
    import dispatch

    real_connect, real_create = socket.socket.connect, socket.create_connection

    def blocked(*a, **k):
        raise NetworkBlocked("network access attempted during offline test")

    socket.socket.connect = blocked
    socket.create_connection = blocked
    t = time.time()
    try:
        raw = load_weather(ROOT / "data" / "dengue_weekly.csv")
        bundle = load_bundle()
        r = build_all(raw, bundle, pd.Timestamp("2025-06-30"), 0.5, 12)
        s, per = ops.simulate(r["fw"], bundle, dict(zip(r["roi"].ward_id, r["roi"].crews_assigned)), r["ratios"])
        msgs = [dispatch.sms(x) for x in r["master"].head(3).itertuples()]
        ok = len(r["master"]) == 60 and len(s) == 3 and all(msgs)
        return ok, time.time() - t, {"wards": len(r["master"]), "transfers": len(r["transfers"]),
                                     "cases_prevented": round(float(s.cases_prevented.sum()), 1), "sample_sms": msgs[0]}
    finally:
        socket.socket.connect, socket.create_connection = real_connect, real_create


if __name__ == "__main__":
    ok, secs, info = run_offline_check()
    print(("PASS" if ok else "FAIL"), f"offline self-test in {secs:.1f}s", info)
    sys.exit(0 if ok else 1)
