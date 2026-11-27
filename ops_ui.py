"""Streamlit tabs for the Phase 2 operations modules. Imported by app.py."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import ops, dispatch  # noqa: E402

CFG = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False}


def _ctx_tiers(ctx):
    return ctx["TIER_COLOR"]


# ------------------------------------------------------------------ 1. resources
def render_resources(R, ctx):
    a, ratios = R["alloc"], R["ratios"]
    tc = _ctx_tiers(ctx)
    n_crit = int((a.deficit_level == "CRITICAL DEFICIT").sum())
    n_def = int((a.deficit_level != "Fully supplied").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regional stock available", f"{ctx['stock']:.0%}")
    c2.metric("Wards fully supplied", f"{len(a) - n_def} of {len(a)}")
    c3.metric("High/Critical wards short", n_crit)
    c4.metric("Spraying crews", f"{R['master'].crews_assigned.sum()} of {ctx['crews']}")
    if n_crit:
        st.error(f"CRITICAL SUPPLY DEFICIT: {n_crit} High/Critical-risk wards cannot be fully supplied at "
                 f"{ctx['stock']:.0%} stock. Request more stock or redirect from low-risk wards.")
    elif n_def:
        st.warning(f"{n_def} lower-risk wards are short of supplies at {ctx['stock']:.0%} stock.")
    else:
        st.success("All wards fully supplied at this stock level.")
    st.caption("Wards are served in priority order: risk level first, then expected cases x population density. "
               "Move the stock and crew sliders in the sidebar to see priorities and deficits change live.")
    st.dataframe(ops.pool_summary(a, ratios), hide_index=True, width="stretch")
    m = R["master"]
    fig = go.Figure(go.Bar(x=m.ward_id, y=m.fulfilment_pct, marker_color=[tc[t] for t in m.risk_tier],
                           customdata=m[["risk_tier", "priority_rank"]],
                           hovertemplate="%{x}<br>%{customdata[0]}, priority #%{customdata[1]}<br>%{y}% supplied<extra></extra>"))
    fig.update_layout(yaxis_title="% of need supplied", yaxis_range=[0, 105], xaxis_title="Wards in priority order (left = first served)",
                      xaxis=dict(showticklabels=False))
    st.plotly_chart(ctx["style_fig"](fig, 280), width="stretch", config=CFG)
    st.caption("Bar colour = risk level (green Low, yellow Medium, orange High, red Critical).")
    cols = ["priority_rank", "ward_id", "city", "risk_tier", "predicted_weekly_cases", "population_density_sqkm",
            "saline_alloc", "saline_need", "kits_alloc", "kits_need", "beds_alloc", "beds_need",
            "crews_assigned", "crews_needed_full", "deficit_level"]
    st.dataframe(m[cols].rename(columns={"saline_alloc": "saline allocated", "saline_need": "saline required",
                                         "kits_alloc": "kits allocated", "kits_need": "kits required",
                                         "beds_alloc": "beds allocated", "beds_need": "beds required",
                                         "crews_needed_full": "crews wanted"}),
                 hide_index=True, width="stretch", height=420)


# ------------------------------------------------------------------ 2. overflow routing
def render_overflow(R, ctx):
    w, tr, c = R["ward_over"], R["transfers"], R["city_beds"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Wards over local capacity", int((w.net_deficit > 0).sum()))
    c2.metric("Patients routed to other wards", int(w.sent_out.sum()))
    c3.metric("Unplaced (city full)", int(w.unplaced.sum()))
    if w.unplaced.sum() > 0:
        bad = ", ".join(c[c.unplaced > 0].city)
        st.error(f"Beds exhausted in: {bad}. Remaining patients cannot be placed inside the city pool.")
    st.caption("Overflow is only routed inside the same city pool, to the ward with the most free beds first. "
               "Bed capacity is a planning assumption (the dataset has no bed census): adjust it in the sidebar.")
    st.subheader("City-wide bed utilisation")
    fig = go.Figure(go.Bar(x=c.city, y=c.utilisation_pct, marker_color=ctx["GOLD"],
                           text=[f"{u:.0f}%" for u in c.utilisation_pct], textposition="outside"))
    fig.update_layout(yaxis_title="Beds in use / capacity (%)", yaxis_range=[0, 115])
    st.plotly_chart(ctx["style_fig"](fig, 260), width="stretch", config=CFG)
    st.dataframe(c.rename(columns={"bed_capacity": "total bed capacity", "projected_demand": "projected bed demand",
                                   "overflow_routed": "patients re-routed", "remaining_headroom": "remaining headroom"}),
                 hide_index=True, width="stretch")
    st.subheader("Transfers")
    if len(tr):
        t = tr.copy()
        t["route"] = t.from_ward + "  to  " + t.to_ward
        st.dataframe(t[["city", "route", "patients"]], hide_index=True, width="stretch")
    else:
        st.info("No ward is over its local bed capacity, so no transfers are needed.")
    st.subheader("Ward bed table: net deficit and headroom")
    show = w.sort_values(["city", "net_deficit"], ascending=[True, False])
    st.dataframe(show[["city", "ward_id", "risk_tier", "beds_need", "local_capacity", "net_deficit", "headroom",
                       "sent_out", "received", "final_load", "utilisation_pct", "status"]].rename(columns={
        "beds_need": "beds needed", "local_capacity": "local capacity", "net_deficit": "net deficit",
        "sent_out": "sent out", "final_load": "final load", "utilisation_pct": "utilisation %"}),
        hide_index=True, width="stretch", height=380)


# ------------------------------------------------------------------ 3. what-if simulator
def render_simulator(R, ctx, bundle):
    fw, ratios = R["fw"], R["ratios"]
    tc = _ctx_tiers(ctx)
    st.caption("Deploy larvicide / fogging crews and see the 2 to 4 week effect. The saved forecasting model is re-run "
               "with the breeding-season rain signal reduced where crews work (assumptions: full coverage cuts it by "
               f"{ops.MAX_BREEDING_CUT:.0%}; effect builds up {ops.RAMP[2]:.0%} / {ops.RAMP[3]:.0%} / {ops.RAMP[4]:.0%} in weeks 2 / 3 / 4).")
    total = st.slider("Crews to deploy", 0, 40, min(ctx["crews"], 40), key="sim_total")
    plan = ops.plan_crews(fw, bundle, total)
    st.markdown("**Edit crews per ward** (starts from the best-ROI deployment; change any number to test your own plan)")
    start = plan[["ward_id", "crews_needed_full", "crews_assigned", "roi_first_crew"]].rename(
        columns={"crews_assigned": "crews", "roi_first_crew": "ROI of first crew (cases prevented)"})
    risk = R["alloc"].set_index("ward_id")["risk_tier"]
    start.insert(1, "risk", start.ward_id.map(risk))
    edited = st.data_editor(start, hide_index=True, width="stretch", height=300, key=f"sim_editor_{total}",
                            disabled=["ward_id", "risk", "crews_needed_full", "ROI of first crew (cases prevented)"],
                            column_config={"crews": st.column_config.NumberColumn("crews", min_value=0, max_value=3, step=1)})
    used = int(edited.crews.sum())
    if used > total:
        st.warning(f"Your plan uses {used} crews but only {total} are available. Reduce some wards.")
    s, per = ops.simulate(fw, bundle, dict(zip(edited.ward_id, edited.crews)), ratios)
    tot_prev = s.cases_prevented.sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cases prevented (weeks 2 to 4)", f"{tot_prev:.0f}")
    k2.metric("Saline bags saved", f"{(s.saline_without - s.saline_with).sum():.0f}")
    k3.metric("Bed-weeks saved", f"{(s.beds_without - s.beds_with).sum():.0f}")
    k4.metric("Cases prevented per crew", f"{(tot_prev / used if used else 0):.1f}")
    fig = go.Figure()
    wk = [f"week {w}" for w in s.week_ahead]
    fig.add_bar(x=wk, y=s.cases_without, name="No intervention", marker_color="#3E8F92")
    fig.add_bar(x=wk, y=s.cases_with, name="With crews", marker_color=ctx["GOLD"])
    fig.update_layout(barmode="group", yaxis_title="Cases per week, all wards")
    st.plotly_chart(ctx["style_fig"](fig, 300), width="stretch", config=CFG)
    st.dataframe(s.assign(week_ahead=wk).rename(columns={"week_ahead": "week", "cases_without": "cases (no crews)",
                 "cases_with": "cases (with crews)", "cases_prevented": "cases prevented", "beds_without": "beds (no crews)",
                 "beds_with": "beds (with crews)", "saline_without": "saline (no crews)", "saline_with": "saline (with crews)"}
                 ).round(1), hide_index=True, width="stretch")
    st.subheader("Ward ranking by intervention ROI (cases prevented per crew)")
    if len(per):
        st.dataframe(per.rename(columns={"cases_without_wk": "cases/wk without crews",
                                         "cases_prevented_3wk": "cases prevented (3 wk)",
                                         "roi_cases_per_crew": "ROI: cases prevented per crew"}),
                     hide_index=True, width="stretch")
    else:
        st.info("Give at least one ward a crew to see the ROI ranking.")


# ------------------------------------------------------------------ 4. dispatch
def render_dispatch(R, ctx):
    m = R["master"]
    c1, c2 = st.columns([2, 1])
    ward = c1.selectbox("Ward notice", list(m.ward_id), key="disp_ward")
    lang = c2.radio("Language", ["en", "bn"], horizontal=True, format_func={"en": "English", "bn": "বাংলা"}.get, key="disp_lang")
    r = next(m[m.ward_id == ward].itertuples())
    st.markdown("**SMS**")
    txt = dispatch.sms(r, lang)
    st.code(txt, language=None)
    st.caption(f"{len(txt)} characters")
    st.markdown("**WhatsApp**")
    st.code(dispatch.whatsapp(r, lang), language=None)
    st.markdown("**Printable commissioner brief**")
    scope = st.radio("Brief covers", ["All wards"] + sorted(m.city.unique()), horizontal=True, key="disp_scope")
    g = m if scope == "All wards" else m[m.city == scope]
    cb = R["city_beds"] if scope == "All wards" else R["city_beds"][R["city_beds"].city == scope]
    wk = f"{m.iloc[0].window_start:%d %b} to {m.iloc[0].window_end:%d %b %Y}"
    page = dispatch.commissioner_brief_html(g, f"Dengue Commissioner Brief: {scope}", f"Forecast window {wk}",
                                            cb, ctx["stock"], ctx["crews"])
    b1, b2 = st.columns(2)
    b1.download_button("Download brief (HTML, open and Print to PDF)", page, file_name=f"brief_{scope.replace(' ', '_')}.html",
                       mime="text/html", width="stretch")
    if b2.button("Write all sample exports to /exports", width="stretch"):
        files = dispatch.export_all(m, ROOT / "exports", R["city_beds"], ctx["stock"], ctx["crews"])
        st.success(f"Wrote {len(files)} files to exports/")
    try:
        st.iframe(page, height=520)
    except Exception:                      # older Streamlit versions
        components.html(page, height=520, scrolling=True)
