"""Dengue Early-Warning Dashboard (Streamlit). Run:  streamlit run app.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from pipeline import build_features, load_weather                       # noqa: E402
from predict import load_bundle, predict, explain                       # noqa: E402

st.set_page_config(page_title="Dengue Early Warning", page_icon="🦟", layout="wide")

TIERS = ["Low", "Medium", "High", "Critical"]
TIER_COLOR = {"Low": "#6FB98F", "Medium": "#E3B341", "High": "#E4752F", "Critical": "#E8466E"}
INK, PANEL, MUTED, GOLD = "#E9F2EC", "#123638", "#9DB8B0", "#F4C95D"
ACTION = {
    "Low": "Keep routine surveillance. No extra action needed yet.",
    "Medium": "Check drains and standing water in this ward and brief local teams.",
    "High": "Schedule larvicide and spraying teams for this ward this week.",
    "Critical": "Make this ward a priority now: spraying, drain clearing, and stock beds and test kits.",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700&family=Source+Sans+3:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, label {{ font-family: 'Source Sans 3', system-ui, sans-serif; }}
h1, h2, h3, .hero h1, .tier-name {{ font-family: 'Sora', system-ui, sans-serif; letter-spacing: -0.01em; }}
.stApp {{ background: radial-gradient(1200px 500px at 85% -10%, #164244 0%, #0B2325 60%); }}
.block-container {{ padding-top: 3.2rem; max-width: 1250px; }}
section[data-testid="stSidebar"] {{ background: #0A1D1F; border-right: 1px solid #1C4547; }}
.hero {{ position: relative; overflow: hidden; border: 1px solid #1F4B4D; border-radius: 22px;
  padding: 30px 36px; margin-bottom: 18px; background: linear-gradient(120deg, #0F2E30 0%, #0B2325 70%); }}
.hero h1 {{ font-size: 2.25rem; margin: 0 0 8px 0; line-height: 1.12; }}
.hero p {{ color: {MUTED}; font-size: 1.05rem; margin: 0; max-width: 560px; line-height: 1.5; }}
.hero .byline {{ margin-top: 14px; font-size: .9rem; color: {GOLD}; }}
.hero svg.art {{ position: absolute; right: 30px; top: -20px; height: 300px; opacity: .95; }}
.card {{ background: {PANEL}; border: 1px solid #1F4B4D; border-radius: 16px; padding: 18px 22px; height: 100%; }}
.tier-card {{ border-left: 8px solid var(--tc); }}
.tier-name {{ font-size: 2.4rem; font-weight: 700; color: var(--tc); line-height: 1.1; margin: 4px 0; }}
.small {{ color: {MUTED}; font-size: .9rem; }}
.big {{ font-family: 'Sora', sans-serif; font-size: 1.9rem; font-weight: 700; color: {INK}; line-height: 1.15; }}
.words {{ background: #0F2E30; border: 1px dashed #2B6466; border-radius: 16px; padding: 16px 22px; margin: 14px 0 6px 0; font-size: 1.04rem; line-height: 1.55; }}
.words b {{ color: {GOLD}; }}
h3 {{ margin-top: .6rem; }}
div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid #1F4B4D; border-radius: 14px; padding: 12px 16px; }}
@media (max-width: 800px) {{ .hero svg.art {{ opacity: .25; }} .hero h1 {{ font-size: 1.7rem; }} }}
</style>
"""

HERO = f"""
<div class="hero">
  <svg class="art" viewBox="0 0 420 330" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <g fill="none" stroke="#2F7C7E">
      <circle cx="270" cy="170" r="150" stroke-opacity=".18"/><circle cx="270" cy="170" r="115" stroke-opacity=".28"/>
      <circle cx="270" cy="170" r="80" stroke-opacity=".4"/><circle cx="270" cy="170" r="46" stroke-opacity=".55"/>
    </g>
    <g transform="translate(168 108) scale(1.25)" fill="#0B2325" stroke="{GOLD}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <ellipse cx="108" cy="36" rx="38" ry="8" transform="rotate(-24 108 36)" fill="#123638"/>
      <ellipse cx="102" cy="42" rx="30" ry="6" transform="rotate(-8 102 42)" fill="#123638"/>
      <ellipse cx="76" cy="80" rx="38" ry="9.5" transform="rotate(16 76 80)"/>
      <path d="M52 86 l4 -14 M66 90 l4 -14 M80 94 l4 -14" stroke="#FFFFFF" stroke-opacity=".7" />
      <ellipse cx="122" cy="63" rx="18" ry="13"/>
      <circle cx="150" cy="60" r="8"/>
      <path d="M157 62 L196 78" /><path d="M152 53 L166 40 M148 52 L156 38" fill="none"/>
      <path d="M118 74 L108 100 L98 128 M124 75 L130 104 L126 130 M130 72 L148 98 L160 122" fill="none"/>
    </g>
    <g fill="none" stroke="{GOLD}" stroke-opacity=".55" stroke-dasharray="3 6"><path d="M360 40 C 330 90, 360 120, 330 160"/></g>
  </svg>
  <h1>Dengue risk, weeks before the first hospital bed fills</h1>
  <p>Pick a ward and a week. The system reads recent rain, heat and humidity, plus each ward's drainage and crowding, and tells you how likely an outbreak is over the next 2 to 4 weeks.</p>
  <div class="byline">Team DEBUG · EDU AI Engineering Hackathon 2026 · runs fully offline, zero API calls</div>
</div>
"""


def style_fig(fig, height=330):
    fig.update_layout(height=height, margin=dict(l=0, r=10, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK, family="Source Sans 3, sans-serif"))
    fig.update_xaxes(gridcolor="#1F4B4D", zerolinecolor="#2B6466")
    fig.update_yaxes(gridcolor="#1F4B4D", zerolinecolor="#2B6466")
    return fig


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


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload new weekly weather CSV (optional)", type="csv",
                          help="Same columns as data/dengue_weekly.csv; dengue_cases not required.")
    try:
        raw = load_weather(pd.read_csv(up)) if up else load_weather(read_default())
    except Exception as e:
        st.error(f"Could not read this file: {e}. Check that it has the same columns as data/dengue_weekly.csv.")
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
bundle = get_bundle()
frow = feats[(feats.ward_id == ward) & (feats.date == asof)].iloc[0]
ex = explain(frow, bundle)
n_alert = int(day.risk_tier.isin(["High", "Critical"]).sum())

c1, c2, c3, c4 = st.columns([1.35, 1, 1, 1])
c1.markdown(f"""<div class="card tier-card" style="--tc:{TIER_COLOR[sel.risk_tier]}">
<div class="small">{ward} · {sel.city} · forecast {w0} to {w1}</div>
<div class="tier-name">{sel.risk_tier}</div><div class="small">risk level</div></div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class="card"><div class="small">Expected cases per week</div>
<div class="big">{sel.predicted_weekly_cases:.1f}</div></div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class="card"><div class="small">80% likely range</div>
<div class="big">{sel.low_80:.1f} to {sel.high_80:.1f}</div></div>""", unsafe_allow_html=True)
c4.markdown(f"""<div class="card"><div class="small">Wards at High or Critical</div>
<div class="big">{n_alert} <span class="small">of {len(day)}</span></div></div>""", unsafe_allow_html=True)

top = ex.sort_values("effect_cases_per_week", ascending=False).iloc[0]
rain_win = float(frow["rain_breed_2_4w"])
st.markdown(f"""<div class="words"><b>In plain words:</b> In {sel.ward_id} we expect about
<b>{sel.predicted_weekly_cases:.0f} cases a week</b> between {w0} and {w1} (likely {sel.low_80:.0f} to {sel.high_80:.0f}),
which is <b>{sel.risk_tier}</b> risk. The biggest push comes from <b>{top['driver'].split(' (')[0].lower()}</b>.
Rain that fell 2 to 4 weeks before this week totals <b>{rain_win:.0f} mm</b>, the window when mosquitoes are breeding.
<br><b>Suggested action:</b> {ACTION[sel.risk_tier]}</div>""", unsafe_allow_html=True)

left, right = st.columns(2)
with left:
    st.subheader("Why this ward? Main drivers")
    e = ex.copy()
    e["color"] = e.effect_cases_per_week.apply(lambda v: TIER_COLOR["Critical"] if v > 0 else TIER_COLOR["Low"])
    fig = go.Figure(go.Bar(x=e.effect_cases_per_week, y=e.driver, orientation="h", marker_color=e.color))
    fig.update_layout(yaxis=dict(automargin=True), xaxis_title="Effect on forecast (cases/week)")
    st.plotly_chart(style_fig(fig), width="stretch")
    st.caption("Red pushes risk up, green pulls it down. 'Ward vulnerability' combines drainage, crowding and sanitation.")

with right:
    st.subheader("Rain in the past 8 weeks")
    ks = list(range(8, -1, -1))
    fig = go.Figure(go.Bar(x=[f"-{k}w" if k else "now" for k in ks], y=[frow[f"rain_lag{k}w"] for k in ks],
                           marker_color=[GOLD if k in (2, 3, 4) else "#3E8F92" for k in ks]))
    fig.update_layout(yaxis_title="Rainfall (mm)")
    st.plotly_chart(style_fig(fig), width="stretch")
    st.caption("Gold bars mark the 2 to 4 week window that drives mosquito breeding.")

st.subheader(f"All wards, forecast {w0} to {w1}")
hm = day.sort_values(["city", "ward_id"])
fig = px.treemap(hm, path=[px.Constant("All wards"), "city", "ward_id"], color="risk_tier",
                 color_discrete_map={**TIER_COLOR, "(?)": "#CFE3DA"},
                 hover_data={"predicted_weekly_cases": True, "low_80": True, "high_80": True})
fig.update_traces(root_color="#CFE3DA", marker_line_color="#0B2325", marker_line_width=2,
                  textfont=dict(color="#0B2325", size=11))
st.plotly_chart(style_fig(fig, 430), width="stretch")

with st.expander("Wards to target first", expanded=True):
    t10 = day.sort_values("predicted_weekly_cases", ascending=False).head(10)
    st.dataframe(t10[["city", "ward_id", "risk_tier", "predicted_weekly_cases", "low_80", "high_80"]],
                 hide_index=True, width="stretch")

mpath = ROOT / "models" / "metrics.json"
if mpath.exists():
    m = json.loads(mpath.read_text())
    with st.expander("How accurate is it? Tested on 2024 to 2025, which the model never saw"):
        t = m["test_2024_2025"]
        a, b, c, d = st.columns(4)
        a.metric("Average error (cases/wk)", f"{t['model']['MAE']:.2f}", f"seasonal baseline {t['baseline_ward_seasonal_mean']['MAE']:.2f}", delta_color="off")
        b.metric("RMSE", f"{t['model']['RMSE']:.2f}", f"seasonal baseline {t['baseline_ward_seasonal_mean']['RMSE']:.2f}", delta_color="off")
        c.metric("Right risk level", f"{t['tier_accuracy']:.0%}")
        d.metric("Within one level", f"{t['tier_within_one']:.0%}")
        by = pd.DataFrame(m["test_by_year"]).T.reset_index().rename(columns={"index": "year"})
        st.dataframe(by, hide_index=True)
        tp = pd.read_csv(ROOT / "models" / "test_predictions.csv", parse_dates=["date"])
        s = tp.groupby("date")[["target", "pred", "lo", "hi"]].sum().reset_index()
        fig = go.Figure()
        fig.add_scatter(x=s.date, y=s.hi, line=dict(width=0), showlegend=False)
        fig.add_scatter(x=s.date, y=s.lo, fill="tonexty", line=dict(width=0), name="80% range", fillcolor="rgba(62,143,146,.30)")
        fig.add_scatter(x=s.date, y=s.target, name="Actual", line=dict(color=INK))
        fig.add_scatter(x=s.date, y=s.pred, name="Predicted", line=dict(color=GOLD))
        fig.update_layout(yaxis_title="All wards, cases per week")
        st.plotly_chart(style_fig(fig), width="stretch")
