"""Train the dengue early-warning model with a chronological split (Requirement 2)."""
import json, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import build_features, feature_columns

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dengue_weekly.csv"
OUT = ROOT / "models"
TIERS = ["Low", "Medium", "High", "Critical"]
SEED = 42


def mk(loss="squared_error", q=None):
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
              l2_regularization=1.0, random_state=SEED)
    return HGB(loss=loss, quantile=q, **kw) if q else HGB(loss=loss, **kw)


def tier_of(x, thr):
    return np.digitize(x, thr)  # 0..3


def metrics(y, p):
    return {"MAE": float(np.mean(np.abs(y - p))), "RMSE": float(np.sqrt(np.mean((y - p) ** 2)))}


def main():
    t0 = time.time()
    feats = build_features(DATA)
    cols = feature_columns(feats)
    d = feats.dropna(subset=cols + ["target"]).copy()
    train = d[d.date < "2023-01-01"]; val = d[(d.date >= "2023-01-01") & (d.date < "2024-01-01")]
    test = d[d.date >= "2024-01-01"]
    # risk tier thresholds come from TRAIN-only target quantiles (no leakage)
    thr = [float(np.quantile(train.target, q)) for q in (0.60, 0.85, 0.97)]

    # baselines (train statistics only)
    ward_mean = train.groupby("ward_id").target.mean()
    d["woy"] = d.date.dt.isocalendar().week.astype(int).clip(upper=52)
    seas = train.assign(woy=train.date.dt.isocalendar().week.astype(int).clip(upper=52)).groupby(["ward_id", "woy"]).target.mean()

    report = {"rows": {"train": len(train), "val": len(val), "test": len(test)},
              "split": {"train": "2016-01 .. 2022-12", "val": "2023", "test": "2024-01 .. 2025-12 (last target week)"},
              "tier_thresholds_weekly_cases": thr, "features": cols}
    m = mk().fit(train[cols], train.target)
    report["validation_2023"] = metrics(val.target.values, m.predict(val[cols]).clip(0))

    full = pd.concat([train, val])
    model = mk().fit(full[cols], full.target)
    lo = mk(loss="quantile", q=0.10).fit(full[cols], full.target)
    hi = mk(loss="quantile", q=0.90).fit(full[cols], full.target)
    p = model.predict(test[cols]).clip(0)
    plo, phi = np.minimum(lo.predict(test[cols]), p).clip(0), np.maximum(hi.predict(test[cols]), p)
    rep = {"model": metrics(test.target.values, p)}
    rep["baseline_ward_mean"] = metrics(test.target.values, test.ward_id.map(ward_mean).fillna(0).values)
    tw = test.assign(woy=test.date.dt.isocalendar().week.astype(int).clip(upper=52))
    sb = [seas.get((w, k), ward_mean.get(w, 0)) for w, k in zip(tw.ward_id, tw.woy)]
    rep["baseline_ward_seasonal_mean"] = metrics(test.target.values, np.array(sb))
    rep["interval_80pct_coverage"] = float(np.mean((test.target.values >= plo) & (test.target.values <= phi)))
    yt, pt = tier_of(test.target.values, thr), tier_of(p, thr)
    rep["tier_accuracy"] = float(np.mean(yt == pt))
    rep["tier_within_one"] = float(np.mean(np.abs(yt - pt) <= 1))
    rep["critical_recall"] = float(np.mean(pt[yt == 3] == 3)) if (yt == 3).any() else None
    report["test_2024_2025"] = rep
    # per-year error for the dashboard (chronological validation)
    ty = test.assign(pred=p, year=test.date.dt.year)
    report["test_by_year"] = {int(y): metrics(g.target.values, g.pred.values) for y, g in ty.groupby("year")}

    OUT.mkdir(exist_ok=True)
    joblib.dump({"model": model, "lo": lo, "hi": hi, "cols": cols, "thr": thr, "tiers": TIERS,
                 "medians": full[cols].median().to_dict()}, OUT / "dengue_model.joblib", compress=3)
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    test.assign(pred=p, lo=plo, hi=phi)[["date", "ward_id", "city", "target", "pred", "lo", "hi"]].to_csv(OUT / "test_predictions.csv", index=False)
    print(json.dumps({k: report[k] for k in ("validation_2023", "test_2024_2025", "tier_thresholds_weekly_cases")}, indent=2))
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
