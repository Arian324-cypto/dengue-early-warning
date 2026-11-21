"""Weather -> ward-level multi-week lag features (Requirement 1)."""
import numpy as np
import pandas as pd

WEATHER_COLS = ["avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm", "relative_humidity_pct"]
STATIC_COLS = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
REQUIRED = ["date", "city", "ward_id"] + STATIC_COLS + WEATHER_COLS
RAIN_LAGS = list(range(0, 9))      # rainfall 0..8 weeks ago (breeding clock: 2-4 week lag matters)
CLIM_LAGS = [1, 2, 4]              # temperature / humidity lags
HORIZON = (2, 4)                   # we predict mean weekly cases over t+2 .. t+4 weeks


def load_weather(path_or_df) -> pd.DataFrame:
    df = path_or_df.copy() if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ward_id", "date"]).drop_duplicates(["ward_id", "date"]).reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Grouped per ward_id so lags never leak across wards."""
    df = load_weather(df)
    g = df.groupby("ward_id", sort=False)
    out = {}
    for k in RAIN_LAGS:
        out[f"rain_lag{k}w"] = g["rainfall_mm"].shift(k)
    for w in (2, 4, 8):
        out[f"rain_sum{w}w"] = g["rainfall_mm"].transform(lambda s, w=w: s.rolling(w, min_periods=w).sum())
    # rain accumulated in the 2-4 week "breeding window" (lags 2,3,4)
    out["rain_breed_2_4w"] = sum(out[f"rain_lag{k}w"] for k in (2, 3, 4))
    for col, short in (("avg_temp_c", "temp"), ("relative_humidity_pct", "hum")):
        out[f"{short}_now"] = df[col]
        for k in CLIM_LAGS:
            out[f"{short}_lag{k}w"] = g[col].shift(k)
        out[f"{short}_mean4w"] = g[col].transform(lambda s: s.rolling(4, min_periods=4).mean())
    out["max_temp_now"] = df["max_temp_c"]
    out["min_temp_now"] = df["min_temp_c"]
    woy = df["date"].dt.isocalendar().week.astype(int).clip(upper=52)
    out["week_sin"] = np.sin(2 * np.pi * woy / 52)
    out["week_cos"] = np.cos(2 * np.pi * woy / 52)
    feats = pd.concat([df[["date", "city", "ward_id"] + STATIC_COLS], pd.DataFrame(out)], axis=1)
    if "dengue_cases" in df.columns:
        fut = [g["dengue_cases"].shift(-h) for h in range(HORIZON[0], HORIZON[1] + 1)]
        feats["target"] = pd.concat(fut, axis=1).mean(axis=1, skipna=False)
        feats["cases_now"] = df["dengue_cases"]
    return feats


FEATURE_COLS = None


def feature_columns(feats: pd.DataFrame):
    return [c for c in feats.columns if c not in ("date", "city", "ward_id", "target", "cases_now")]


# groups used for the dashboard's "main drivers" explanation
DRIVER_GROUPS = {
    "Rainfall (2-4 week breeding lag)": lambda c: c.startswith("rain_"),
    "Temperature": lambda c: c.startswith("temp_") or c in ("max_temp_now", "min_temp_now"),
    "Humidity": lambda c: c.startswith("hum_"),
    "Season (time of year)": lambda c: c.startswith("week_"),
    "Ward vulnerability (drainage, crowding, sanitation)": lambda c: c in STATIC_COLS,
}
