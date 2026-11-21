"""Fast local inference with a serialized model (joblib) - no external API calls (Requirement 3/cost)."""
import sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import build_features, DRIVER_GROUPS, HORIZON

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "dengue_model.joblib"


def load_bundle(path=MODEL_PATH):
    return joblib.load(path)


def predict(feats: pd.DataFrame, bundle) -> pd.DataFrame:
    """feats: output of build_features (rows with complete lag windows). Vectorised, ~ms for all wards."""
    X = feats[bundle["cols"]]
    p = bundle["model"].predict(X).clip(0)
    lo = np.minimum(bundle["lo"].predict(X), p).clip(0)
    hi = np.maximum(bundle["hi"].predict(X), p)
    tier = np.digitize(p, bundle["thr"])
    out = feats[["date", "city", "ward_id"]].copy()
    out["predicted_weekly_cases"] = p.round(2)
    out["low_80"], out["high_80"] = lo.round(2), hi.round(2)
    out["risk_tier"] = [bundle["tiers"][i] for i in tier]
    out["window_start"] = out["date"] + pd.to_timedelta(HORIZON[0], unit="W")
    out["window_end"] = out["date"] + pd.to_timedelta(HORIZON[1], unit="W")
    return out.reset_index(drop=True)


def explain(row: pd.Series, bundle) -> pd.DataFrame:
    """Driver breakdown: how much does each factor group push this ward's forecast above/below the
    typical (median) conditions? Occlusion: reset one group to its training median, measure the change."""
    cols = bundle["cols"]
    x = row[cols].astype(float).to_frame().T
    base = float(bundle["model"].predict(x)[0])
    rows = []
    for name, f in DRIVER_GROUPS.items():
        x2 = x.copy()
        for c in cols:
            if f(c):
                x2[c] = bundle["medians"][c]
        rows.append((name, base - float(bundle["model"].predict(x2)[0])))
    return pd.DataFrame(rows, columns=["driver", "effect_cases_per_week"]).sort_values("effect_cases_per_week")
