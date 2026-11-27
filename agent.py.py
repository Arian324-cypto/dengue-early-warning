import base64
import os
import re
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from gtts import gTTS

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except Exception:
    pass

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "data" / "dengue_weekly.csv"
GUIDE_PATH = ROOT / "USER_GUIDE.md"
BASE = "https://generativelanguage.googleapis.com/v1beta"

APP_HELP = """DASHBOARD HELP (how to use this website):
- Sidebar: upload a new weekly weather CSV (optional); choose 'Weather observed up to (week)'; choose City and Ward; 'Available regional stock' slider sets the share of supplies available.
- Tabs: Forecast (risk level, expected cases per week, likely range, plain-words summary, main drivers, rain in the past 8 weeks); Resource allocation (saline, test kits, beds per ward); Hospital overflow (which hospitals may run out of beds); What-if simulator (change conditions and see the effect); Dispatch notices (messages for sending supplies).
- AI Guide page (left menu): ask in Bangla or English by typing or by the microphone. Answers can be read aloud.
- If the page looks stuck or shows an error: refresh the browser (Ctrl+F5) and try again. If the AI answer says the service is busy, wait a few seconds and ask again.
"""

SYSTEM_PROMPT = """You are the AI Guide of a Dengue Early Warning dashboard for Chattogram city. You do two jobs: (A) answer questions about dengue risk using the DATA CONTEXT, and (B) help users use this website using the DASHBOARD HELP and GUIDE TEXT.
Rules:
1. Use ONLY the numbers in the DATA CONTEXT. Never invent numbers. If the data or help text does not contain the answer, say: "This information is not available in our data." (in Bangla: "এই তথ্য আমাদের ডেটায় নেই।")
2. For website problems, give short step-by-step help from DASHBOARD HELP and GUIDE TEXT only. If the problem is not covered, suggest refreshing the page and contacting the team.
3. Reply in the same language the user used (Bangla or English).
4. Do not give medical treatment or dosage advice. For fever, rash, or severe body pain, tell the user to see a doctor or hospital promptly.
5. Prevention tips are allowed: remove stagnant water, use mosquito nets, keep drains clean.
6. Keep answers short (3-6 sentences), simple words, no markdown symbols like ** or #.
"""

_PICKED = None


def _get_key():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
    if not key:
        raise RuntimeError("GEMINI_API_KEY পাওয়া যায়নি।")
    return key


def _pick_model():
    global _PICKED
    if _PICKED:
        return _PICKED
    env = os.getenv("GEMINI_MODEL")
    if env:
        _PICKED = env
        return env
    r = requests.get(
        BASE + "/models?pageSize=100",
        headers={"x-goog-api-key": _get_key()},
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError("Model list " + str(r.status_code) + ": " + r.text[:300])
    names = []
    for m in r.json().get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            names.append(m["name"].split("/")[-1])
    bad = ("image", "tts", "live", "audio", "embedding", "thinking", "exp", "8b", "robotics")
    flash = [n for n in names if "flash" in n and not any(b in n for b in bad)]
    latest = [n for n in flash if n.endswith("-latest")]
    nolite = [n for n in flash if "lite" not in n]
    prefer = latest or nolite or flash or names
    if not prefer:
        raise RuntimeError("এই key-তে কোনো model চালু নেই।")
    _PICKED = prefer[0]
    return _PICKED


def _call(contents, system=None):
    body = {"contents": contents, "generationConfig": {"temperature": 0.2}}
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    models = [_pick_model(), "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.0-flash"]
    seen = []
    last = "unknown error"
    for model in models:
        if model in seen:
            continue
        seen.append(model)
        for attempt in range(3):
            r = requests.post(
                BASE + "/models/" + model + ":generateContent",
                headers={"x-goog-api-key": _get_key(), "Content-Type": "application/json"},
                json=body,
                timeout=60,
            )
            if r.ok:
                parts = r.json()["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts).strip()
            last = str(r.status_code) + ": " + r.text[:200]
            if r.status_code in (429, 500, 503):
                time.sleep(2 * (attempt + 1))
                continue
            break
    raise RuntimeError(last)


def load_data():
    return pd.read_csv(CSV_PATH, parse_dates=["date"])


def _guide_text():
    try:
        return GUIDE_PATH.read_text(encoding="utf-8")[:5000]
    except Exception:
        return ""


def build_context(question: str) -> str:
    df = load_data()
    last_date = df["date"].max()
    recent = df[df["date"] > last_date - pd.Timedelta(weeks=4)]

    lines = ["Data covers " + str(df["date"].min().date()) + " to " + str(last_date.date()) + ", city Chattogram, weekly, ward-level."]
    lines.append("Total dengue cases in the last 4 weeks (all wards): " + str(int(recent["dengue_cases"].sum())) + ".")

    last_week = df[df["date"] == last_date]
    lines.append(
        "Latest week (" + str(last_date.date()) + "): cases=" + str(int(last_week["dengue_cases"].sum()))
        + ", saline_bags_needed=" + str(int(last_week["saline_bags_needed"].sum()))
        + ", diagnostic_kits_needed=" + str(int(last_week["diagnostic_kits_needed"].sum()))
        + ", hospital_beds_required=" + str(int(last_week["hospital_beds_required"].sum())) + "."
    )

    top = (
        recent.groupby("ward_id")
        .agg(cases=("dengue_cases", "sum"),
             water_logging=("water_logging_risk_index", "mean"),
             sanitation=("sanitation_score", "mean"),
             density=("population_density_sqkm", "mean"))
        .sort_values("cases", ascending=False)
        .head(5)
    )
    lines.append("Top 5 wards by cases in the last 4 weeks:")
    for ward, r in top.iterrows():
        lines.append("- " + str(ward) + ": cases=" + str(int(r.cases))
                     + ", water_logging_risk=" + format(r.water_logging, ".2f")
                     + ", sanitation_score=" + format(r.sanitation, ".2f")
                     + ", density=" + str(int(r.density)))

    w = last_week[["avg_temp_c", "rainfall_mm", "relative_humidity_pct"]].mean()
    lines.append("Latest week average weather: temp=" + format(w.avg_temp_c, ".1f") + "C, rainfall="
                 + format(w.rainfall_mm, ".1f") + "mm, humidity=" + format(w.relative_humidity_pct, ".0f") + "%.")

    m = re.search(r"ward[\s_]*0*(\d{1,2})|ওয়ার্ড[\s_]*0*(\d{1,2})", question, re.IGNORECASE)
    if m:
        num = int(m.group(1) or m.group(2))
        wid = "CCC_Ward_" + format(num, "02d")
        wd = df[df["ward_id"] == wid].sort_values("date").tail(8)
        if len(wd):
            lines.append("Last 8 weeks for " + wid + ":")
            for _, r in wd.iterrows():
                lines.append("- " + str(r.date.date()) + ": cases=" + str(int(r.dengue_cases))
                             + ", rain=" + str(r.rainfall_mm) + "mm, temp=" + str(r.avg_temp_c) + "C")
        else:
            lines.append("No data found for ward " + str(num) + ".")

    lines.append("")
    lines.append(APP_HELP)
    guide = _guide_text()
    if guide:
        lines.append("GUIDE TEXT:")
        lines.append(guide)
    return "\n".join(lines)


def ask_agent(question: str, history=None) -> str:
    context = build_context(question)
    contents = []
    for m in (history or [])[-6:]:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    contents.append({"role": "user", "parts": [{
        "text": "DATA CONTEXT:\n" + context + "\n\nUSER QUESTION:\n" + question}]})
    return _call(contents, SYSTEM_PROMPT)


def transcribe(audio_bytes: bytes) -> str:
    data = base64.b64encode(audio_bytes).decode()
    contents = [{"role": "user", "parts": [
        {"inline_data": {"mime_type": "audio/wav", "data": data}},
        {"text": "Transcribe this speech exactly. It may be Bangla or English. Output only the transcript."},
    ]}]
    return _call(contents)


def speak(text: str) -> bytes:
    clean = re.sub(r"[*#_`]", "", text)
    lang = "bn" if re.search(r"[ঀ-৿]", clean) else "en"
    buf = BytesIO()
    gTTS(text=clean, lang=lang).write_to_fp(buf)
    return buf.getvalue()
