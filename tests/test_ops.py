"""Quick checks:  python tests/test_ops.py"""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import ops
from export import build_all
from pipeline import load_weather
from predict import load_bundle

raw, bundle = load_weather(ROOT / "data" / "dengue_weekly.csv"), load_bundle()
d = pd.Timestamp("2025-06-30")
full = build_all(raw, bundle, d, 1.0, 12)
assert (full["alloc"].saline_short == 0).all(), "100% stock must fully supply every ward"
tight = build_all(raw, bundle, d, 0.3, 12)
a = tight["alloc"]
assert a.saline_alloc.sum() <= int(0.3 * a.saline_need.sum()) + 1, "allocation cannot exceed the stock"
assert a.iloc[0].saline_alloc == a.iloc[0].saline_need, "highest priority ward is served first"
w, tr, c = tight["ward_over"], tight["transfers"], tight["city_beds"]
cities = w.set_index("ward_id").city
assert all(cities[t.from_ward] == cities[t.to_ward] for t in tr.itertuples()), "overflow must stay inside one city"
s, per = ops.simulate(full["fw"], bundle, {"DNCC_Ward_01": 2}, full["ratios"])
assert s.cases_prevented.iloc[0] < s.cases_prevented.iloc[2], "effect builds up over weeks 2-4"
assert (s.cases_with <= s.cases_without + 1e-9).all()
print("all ops tests passed")
