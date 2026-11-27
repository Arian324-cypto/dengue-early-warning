"""Generate the sample dispatch exports into /exports.   python src/export.py [YYYY-MM-DD] [stock_pct] [crews]
Runs fully offline (no network, no LLM)."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import build_features, load_weather       # noqa: E402
from predict import load_bundle, predict                 # noqa: E402
import ops, dispatch                                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def ops_run(feats, preds, raw, bundle, date, stock_pct=0.6, total_crews=12, bed_factor=0.9, capacity_csv=None):
    """Run allocation, crew assignment, overflow routing and the dispatch table for one forecast week."""
    ratios = ops.fit_ratios(raw)
    fr = ops.week_frame(feats, preds, date)
    alloc = ops.allocate(fr, stock_pct, ratios)
    fw = feats[feats.date == date].reset_index(drop=True)
    crews = ops.assign_crews_by_priority(alloc, total_crews)
    roi = ops.plan_crews(fw, bundle, total_crews)
    cap = ops.bed_capacity(raw, ratios, bed_factor, capacity_csv)
    ward_over, transfers, city_beds = ops.route_overflow(alloc, cap)
    master = dispatch.build_master(alloc, crews, ward_over)
    return dict(ratios=ratios, alloc=alloc, crews=crews, ward_over=ward_over, transfers=transfers,
                city_beds=city_beds, master=master, fw=fw, roi=roi)


def build_all(raw, bundle, date, stock_pct=0.6, total_crews=12, bed_factor=0.9, capacity_csv=None):
    feats = build_features(raw).dropna(subset=bundle["cols"]).reset_index(drop=True)
    preds = predict(feats, bundle)
    r = ops_run(feats, preds, raw, bundle, date, stock_pct, total_crews, bed_factor, capacity_csv)
    r.update(feats=feats, preds=preds)
    return r


if __name__ == "__main__":
    raw = load_weather(ROOT / "data" / "dengue_weekly.csv")
    bundle = load_bundle()
    date = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2025-06-30")
    stock = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    crews = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    r = build_all(raw, bundle, date, stock, crews)
    files = dispatch.export_all(r["master"], ROOT / "exports", r["city_beds"], stock, crews)
    print("Wrote to exports/:", ", ".join(files))
