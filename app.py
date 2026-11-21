"""Dengue Early-Warning Dashboard (Streamlit). Run:  streamlit run app.py"""
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from pipeline import build_features, load_weather, HORIZON            # noqa: E402
from predict import load_bundle, predict, explain                      # noqa: E402

st.set_page_config(page_title="Dengue Early Warning", page_icon="🦟", layout="wide")
TIER_COLOR = {"Low": "#2e9e5b", "Medium": "#e6b800", "High": "#e8742a", "Critical": "#c62828"}
TIERS = ["Low", "Medium", "High", "Critical"]


@st.cache_resource                       # model is loaded once per server process
def get_bundle():
    return load_bundle()


@st.cache_data(show_spinner=False)       # features + predictions computed once per dataset (cost control)
def run_pipeline(df: pd.DataFrame):
    feats = build_features(df)
    feats = feats.dropna(subset=get_bundle()["cols"]).reset_index(drop=True)
    return feats, predict(feats, get_bundle())


@st.cache_data
def read_default():
    return pd.read_csv(ROOT / "data" / "dengue_weekly.csv")


st.title("🦟 Urban Dengue Early-Warning System")
st.caption("Predicts dengue risk **2 to 4 weeks ahead** for each ward from weather (temperature, humidity, rainfall lags) "
           "and ward vulnerability. All inference runs locally from a saved model: zero API calls.")

with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload new weekly weather CSV (optional)", type="csv",
                          help="Same columns as data/dengue_weekly.csv; dengue_cases not required.")
    try:
        raw = load_weather(pd.read_csv(up)) if up else load_weather(read_default())
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
    feats, preds = run_pipeline(raw)
    if preds.empty:
        st.error("Need at least 8 consecutive weeks of weather per ward.")
        st.stop()
    st.header("Selection")
    dates = sorted(preds["date"].unique())
    # default: the peak-risk week of the most recent year, so the demo opens on a meaningful forecast
    last_year = preds[preds["date"] >= preds["date"].max() - pd.Timedelta(weeks=52)]
    peak = last_year.groupby("date")["predicted_weekly_cases"].sum().idxmax()
    asof = st.select_slider("Weather observed up to (week)", options=dates, value=peak,
                            format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"))
    cities = sorted(preds["city"].unique())
    city = st.selectbox("City", cities)
    wards = sorted(preds[preds.city == city]["ward_id"].unique())
    ward = st.selectbox("Ward", wards)

day = preds[preds["date"] == asof]
sel = day[day.ward_id == ward].iloc[0]
w0, w1 = sel.window_start.strftime("%d %b"), sel.window_end.strftime("%d %b %Y")

c1, c2, c3, c4 = st.columns(4)
color = TIER_COLOR[sel.risk_tier]
c1.markdown(f"**Risk tier ({w0} – {w1})**<div style='font-size:2.2rem;font-weight:700;color:{color}'>{sel.risk_tier}</div>",
            unsafe_allow_html=True)
c2.metric("Expected cases / week", f"{sel.predicted_weekly_cases:.1f}")
c3.metric("80% confidence range", f"{sel.low_80:.1f} – {sel.high_80:.1f}")
c4.metric("Wards at High/Critical", int(day.risk_tier.isin(["High", "Critical"]).sum()), help="All wards, same week")

left, right = st.columns([1, 1])
bundle = get_bundle()
frow = feats[(feats.ward_id == ward) & (feats.date == asof)].iloc[0]

with left:
    st.subheader("Why? Main drivers for this ward")
    ex = explain(frow, bundle)
    ex["color"] = ex.effect_cases_per_week.apply(lambda v: "#c62828" if v > 0 else "#2e9e5b")
    fig = go.Figure(go.Bar(x=ex.effect_cases_per_week, y=ex.driver, orientation="h", marker_color=ex.color))
    fig.update_layout(height=320, margin=dict(l=0, r=10, t=10, b=0), yaxis=dict(automargin=True),
                      xaxis_title="Effect on forecast (cases/week)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Red pushes risk up, green pushes it down.")

with right:
    st.subheader("Rainfall in the past 8 weeks (breeding window shaded)")
    lags = [f"rain_lag{k}w" for k in range(8, -1, -1)]
    fig = go.Figure(go.Bar(x=[f"-{k}w" if k else "now" for k in range(8, -1, -1)], y=[frow[c] for c in lags],
                           marker_color=["#e8742a" if k in (2, 3, 4) else "#7aa6d8" for k in range(8, -1, -1)]))
    fig.update_layout(height=320, margin=dict(l=0, r=10, t=10, b=0), yaxis_title="Rainfall (mm)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Orange = the 2-4 week lag window that drives mosquito breeding.")

st.subheader(f"Ward risk heatmap: all wards, forecast {w0} – {w1}")
hm = day.copy()
hm["tier_n"] = hm.risk_tier.map({t: i for i, t in enumerate(TIERS)})
hm = hm.sort_values(["city", "ward_id"])
fig = px.treemap(hm, path=[px.Constant("Bangladesh"), "city", "ward_id"], values=None, color="risk_tier",
                 color_discrete_map=TIER_COLOR,
                 hover_data={"predicted_weekly_cases": True, "low_80": True, "high_80": True})
fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, width="stretch")

with st.expander("Priority list: wards to target first", expanded=True):
    top = day.sort_values("predicted_weekly_cases", ascending=False).head(10)
    st.dataframe(top[["city", "ward_id", "risk_tier", "predicted_weekly_cases", "low_80", "high_80"]],
                 hide_index=True, width="stretch")

mpath = ROOT / "models" / "metrics.json"
if mpath.exists():
    import json
    m = json.loads(mpath.read_text())
    with st.expander("Model validation (chronological hold-out: trained on 2016-2023, tested on 2024-2025)"):
        t = m["test_2024_2025"]
        a, b, c, d = st.columns(4)
        a.metric("MAE (cases/wk)", f"{t['model']['MAE']:.2f}", f"baseline {t['baseline_ward_seasonal_mean']['MAE']:.2f}", delta_color="off")
        b.metric("RMSE", f"{t['model']['RMSE']:.2f}", f"baseline {t['baseline_ward_seasonal_mean']['RMSE']:.2f}", delta_color="off")
        c.metric("Tier accuracy", f"{t['tier_accuracy']:.0%}")
        d.metric("Within one tier", f"{t['tier_within_one']:.0%}")
        by = pd.DataFrame(m["test_by_year"]).T.reset_index().rename(columns={"index": "year"})
        st.dataframe(by, hide_index=True)
        tp = pd.read_csv(ROOT / "models" / "test_predictions.csv", parse_dates=["date"])
        s = tp.groupby("date")[["target", "pred", "lo", "hi"]].sum().reset_index()
        fig = go.Figure()
        fig.add_scatter(x=s.date, y=s.hi, line=dict(width=0), showlegend=False)
        fig.add_scatter(x=s.date, y=s.lo, fill="tonexty", line=dict(width=0), name="80% range", fillcolor="rgba(100,150,220,.25)")
        fig.add_scatter(x=s.date, y=s.target, name="Actual", line=dict(color="#222"))
        fig.add_scatter(x=s.date, y=s.pred, name="Predicted", line=dict(color="#e8742a"))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="All wards, avg cases/week")
        st.plotly_chart(fig, width="stretch")
