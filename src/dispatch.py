"""Template-based dispatch notices (SMS / WhatsApp / printable HTML brief). Deterministic and offline: no LLM.

Requirement 4: ward-specific notices listing predicted cases, allocated vs required supplies, deficit alerts
and assigned spraying teams. English and Bangla templates."""
import html
from pathlib import Path
import pandas as pd

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
BN_TIER = {"Low": "কম", "Medium": "মাঝারি", "High": "উচ্চ", "Critical": "সংকটজনক"}


def bn(x) -> str:
    return str(x).translate(BN_DIGITS)


def build_master(alloc: pd.DataFrame, crews: pd.DataFrame, overflow: pd.DataFrame) -> pd.DataFrame:
    """One row per ward with everything a notice needs."""
    m = alloc.merge(crews[["ward_id", "crews_assigned", "crews_needed_full"]], on="ward_id", how="left")
    m = m.merge(overflow[["ward_id", "local_capacity", "sent_out", "received", "unplaced", "status"]], on="ward_id", how="left")
    m["crews_assigned"] = m["crews_assigned"].fillna(0).astype(int)
    m["crew_gap"] = (m["crews_needed_full"].fillna(0) - m["crews_assigned"]).clip(lower=0).astype(int)
    return m.sort_values("priority_rank").reset_index(drop=True)


def _window(r) -> str:
    return f"{r.window_start:%d %b}-{r.window_end:%d %b}"


def _short_items(r) -> str:
    return (f"Saline {r.saline_alloc}/{r.saline_need}, Kits {r.kits_alloc}/{r.kits_need}, "
            f"Beds {r.beds_alloc}/{r.beds_need}")


def _flags(r) -> list:
    f = []
    if r.deficit_level != "Fully supplied":
        f.append(f"{r.deficit_level}: short saline {r.saline_short}, kits {r.kits_short}, beds {r.beds_short}")
    if r.sent_out > 0:
        f.append(f"{int(r.sent_out)} patients to be routed to other wards in {r.city}")
    if r.received > 0:
        f.append(f"expect {int(r.received)} overflow patients from other wards")
    if r.unplaced > 0:
        f.append(f"{int(r.unplaced)} patients cannot be placed in {r.city}: request regional help")
    return f


def sms(r, lang="en") -> str:
    """Short text, kept under 160 characters in English."""
    if lang == "bn":
        s = (f"ডেঙ্গু সতর্কতা {r.ward_id}: ঝুঁকি {BN_TIER[r.risk_tier]}, প্রায় {bn(round(r.predicted_weekly_cases))} রোগী/সপ্তাহ। "
             f"স্যালাইন {bn(r.saline_alloc)}/{bn(r.saline_need)}, কিট {bn(r.kits_alloc)}/{bn(r.kits_need)}, "
             f"বেড {bn(r.beds_alloc)}/{bn(r.beds_need)}। স্প্রে দল {bn(r.crews_assigned)}।")
        if r.deficit_level != "Fully supplied":
            s += " ঘাটতি আছে!"
        return s
    s = (f"DENGUE {r.risk_tier.upper()} {r.ward_id}: ~{r.predicted_weekly_cases:.0f} cases/wk {_window(r)}. "
         f"{_short_items(r)}. Crews {r.crews_assigned}.")
    if r.deficit_level != "Fully supplied":
        s += " SHORTAGE!"
    return s


def whatsapp(r, lang="en") -> str:
    flags = _flags(r)
    if lang == "bn":
        lines = [f"*ডেঙ্গু সতর্কবার্তা: {r.ward_id} ({r.city})*",
                 f"ঝুঁকি: *{BN_TIER[r.risk_tier]}*",
                 f"পূর্বাভাস: প্রতি সপ্তাহে প্রায় {bn(round(r.predicted_weekly_cases))} রোগী "
                 f"({bn(round(r.low_80))} থেকে {bn(round(r.high_80))}), সময়: {_window(r)}",
                 f"স্যালাইন ব্যাগ: {bn(r.saline_alloc)} বরাদ্দ / {bn(r.saline_need)} প্রয়োজন",
                 f"ডায়াগনস্টিক কিট: {bn(r.kits_alloc)} বরাদ্দ / {bn(r.kits_need)} প্রয়োজন",
                 f"হাসপাতাল বেড: {bn(r.beds_alloc)} বরাদ্দ / {bn(r.beds_need)} প্রয়োজন",
                 f"স্প্রে দল: {bn(r.crews_assigned)}টি"]
        if r.deficit_level != "Fully supplied":
            lines.append("সতর্কতা: সরবরাহে ঘাটতি আছে, অতিরিক্ত মজুতের অনুরোধ করুন।")
        if r.sent_out > 0:
            lines.append(f"{bn(int(r.sent_out))} জন রোগী শহরের অন্য ওয়ার্ডে পাঠাতে হবে।")
        return "\n".join(lines)
    lines = [f"*DENGUE ALERT: {r.ward_id} ({r.city})*",
             f"Risk level: *{r.risk_tier}*",
             f"Forecast: ~{r.predicted_weekly_cases:.0f} cases/week (likely {r.low_80:.0f}-{r.high_80:.0f}), {_window(r)}",
             f"Saline bags: {r.saline_alloc} allocated / {r.saline_need} required",
             f"Diagnostic kits: {r.kits_alloc} allocated / {r.kits_need} required",
             f"Hospital beds: {r.beds_alloc} allocated / {r.beds_need} required",
             f"Spraying crews assigned: {r.crews_assigned}"]
    lines += [f"ALERT: {x}" for x in flags]
    return "\n".join(lines)


def _bar(a, n):
    pct = 100 if n == 0 else min(100, round(100 * a / n))
    col = "#2e9e6b" if pct >= 100 else ("#e4752f" if pct >= 50 else "#d6335c")
    return f'<div class="bar"><span style="width:{pct}%;background:{col}"></span></div>'


def commissioner_brief_html(m: pd.DataFrame, title: str, subtitle: str, city_beds: pd.DataFrame = None,
                            stock_pct: float = 1.0, total_crews: int = 0) -> str:
    """Printable A4 brief (use the browser's Print > Save as PDF). Self-contained: inline CSS, no network."""
    rows = []
    for r in m.itertuples():
        e = html.escape
        rows.append(f"""<tr class="{'alert' if r.deficit_level != 'Fully supplied' else ''}">
<td>{r.priority_rank}</td><td><b>{e(r.ward_id)}</b><br><small>{e(r.city)}</small></td>
<td>{e(r.risk_tier)}</td><td>{r.predicted_weekly_cases:.1f}<br><small>{r.low_80:.0f}-{r.high_80:.0f}</small></td>
<td>{r.saline_alloc} / {r.saline_need}{_bar(r.saline_alloc, r.saline_need)}</td>
<td>{r.kits_alloc} / {r.kits_need}{_bar(r.kits_alloc, r.kits_need)}</td>
<td>{r.beds_alloc} / {r.beds_need}{_bar(r.beds_alloc, r.beds_need)}</td>
<td>{r.crews_assigned}</td>
<td>{e('; '.join(_flags(r)) or 'None')}</td></tr>""")
    cb = ""
    if city_beds is not None and len(city_beds):
        cb = "<h2>City bed utilisation</h2><table class='s'><tr><th>City</th><th>Capacity</th><th>Projected demand</th><th>Utilisation</th><th>Headroom</th><th>Unplaced</th></tr>" + "".join(
            f"<tr><td>{html.escape(c.city)}</td><td>{c.bed_capacity}</td><td>{c.projected_demand}</td><td>{c.utilisation_pct:.0f}%</td>"
            f"<td>{c.remaining_headroom}</td><td>{c.unplaced}</td></tr>" for c in city_beds.itertuples()) + "</table>"
    n_def = int((m.deficit_level != "Fully supplied").sum())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>@page{{size:A4 landscape;margin:12mm}}body{{font-family:Arial,Helvetica,sans-serif;color:#111;font-size:11px}}
h1{{margin:0 0 2px;font-size:20px}}h2{{font-size:14px;margin:16px 0 4px}}.sub{{color:#555;margin-bottom:8px}}
.kpi{{display:flex;gap:10px;margin:8px 0}}.kpi div{{border:1px solid #bbb;border-radius:6px;padding:6px 10px}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:3px 5px;vertical-align:top;text-align:left}}
th{{background:#eee}}tr.alert td{{background:#fff1ec}}small{{color:#666}}.bar{{height:4px;background:#ddd;margin-top:2px}}
.bar span{{display:block;height:4px}}table.s{{width:auto}}.foot{{margin-top:10px;color:#666}}</style></head><body>
<h1>{html.escape(title)}</h1><div class="sub">{html.escape(subtitle)}</div>
<div class="kpi"><div><b>{len(m)}</b> wards</div><div><b>{n_def}</b> with supply deficit</div>
<div>Regional stock available: <b>{stock_pct:.0%}</b></div><div>Spraying crews: <b>{total_crews}</b></div></div>
<table><tr><th>Priority</th><th>Ward</th><th>Risk</th><th>Predicted cases/wk<br>(80% range)</th>
<th>Saline bags<br>alloc / required</th><th>Diagnostic kits<br>alloc / required</th><th>Beds<br>alloc / required</th>
<th>Spraying crews</th><th>Alerts and actions</th></tr>{''.join(rows)}</table>{cb}
<p class="foot">Generated locally from the saved forecasting model; deterministic templates, no external services.</p>
</body></html>"""


def export_all(m: pd.DataFrame, out_dir, city_beds=None, stock_pct=1.0, total_crews=0, top_n=5):
    """Write sample SMS / WhatsApp / HTML briefs into /exports (also callable from the dashboard)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    top = m.head(top_n)
    for lang in ("en", "bn"):
        (out / f"sms_{lang}.txt").write_text("\n\n".join(sms(r, lang) for r in top.itertuples()), encoding="utf-8")
        (out / f"whatsapp_{lang}.txt").write_text("\n\n---\n\n".join(whatsapp(r, lang) for r in top.itertuples()), encoding="utf-8")
    wk = f"{m.iloc[0].window_start:%d %b} to {m.iloc[0].window_end:%d %b %Y}"
    (out / "commissioner_brief_all_wards.html").write_text(
        commissioner_brief_html(m, "Dengue Commissioner Brief: all wards", f"Forecast window {wk}", city_beds, stock_pct, total_crews),
        encoding="utf-8")
    for city, g in m.groupby("city"):
        (out / f"commissioner_brief_{city.replace(' ', '_')}.html").write_text(
            commissioner_brief_html(g, f"Dengue Commissioner Brief: {city}", f"Forecast window {wk}",
                                    None if city_beds is None else city_beds[city_beds.city == city], stock_pct, total_crews),
            encoding="utf-8")
    m.to_csv(out / "dispatch_table.csv", index=False)
    return sorted(p.name for p in out.iterdir())
