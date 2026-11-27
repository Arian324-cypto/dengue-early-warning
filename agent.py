import os
import re
from io import BytesIO
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from gtts import gTTS

load_dotenv(Path(__file__).parent / ".env")

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CSV_PATH = Path(__file__).parent / "data" / "dengue_weekly.csv"

SYSTEM_PROMPT = """You are the AI Guide of a Dengue Early Warning dashboard for Chattogram city.
Rules:
1. Use ONLY the numbers in the DATA CONTEXT. Never invent numbers. If the data does not contain it, say: "This information is not available in our data." (in Bangla: "এই তথ্য আমাদের ডেটায় নেই।")
2. Reply in the same language the user used (Bangla or English).
3. Do not give medical treatment or dosage advice. For fever, rash, or severe body pain, tell the user to see a doctor or hospital promptly.
4. Prevention tips are allowed: remove stagnant water, use mosquito nets, keep drains clean.
5. Keep answers short (3-6 sentences), simple words, no markdown symbols like ** or #.
"""


def _get_key():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
    return key


_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        key = _get_key()
        if not key:
            raise RuntimeError("GEMINI_API_KEY পাওয়া যায়নি। .env ফাইল দেখো।")
        _CLIENT = genai.Client(api_key=key)
    return _CLIENT


def load_data():
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    return df


def build_context(question: str) -> str:
    df = load_data()
    last_date = df["date"].max()
    recent = df[df["date"] > last_date - pd.Timedelta(weeks=4)]

    lines = [f"Data covers {df['date'].min().date()} to {last_date.date()}, city Chattogram, weekly, ward-level."]

    total_recent = int(recent["dengue_cases"].sum())
    lines.append(f"Total dengue cases in the last 4 weeks (all wards): {total_recent}.")

    last_week = df[df["date"] == last_date]
    lines.append(
        f"Latest week ({last_date.date()}): cases={int(last_week['dengue_cases'].sum())}, "
        f"saline_bags_needed={int(last_week['saline_bags_needed'].sum())}, "
        f"diagnostic_kits_needed={int(last_week['diagnostic_kits_needed'].sum())}, "
        f"hospital_beds_required={int(last_week['hospital_beds_required'].sum())}."
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
        lines.append(f"- {ward}: cases={int(r.cases)}, water_logging_risk={r.water_logging:.2f}, "
                     f"sanitation_score={r.sanitation:.2f}, density={int(r.density)}")

    w = last_week[["avg_temp_c", "rainfall_mm", "relative_humidity_pct"]].mean()
    lines.append(f"Latest week average weather: temp={w.avg_temp_c:.1f}C, rainfall={w.rainfall_mm:.1f}mm, humidity={w.relative_humidity_pct:.0f}%.")

    # প্রশ্নে নির্দিষ্ট ward নম্বর থাকলে তার শেষ ৮ সপ্তাহ যোগ করো
    m = re.search(r"ward[\s_]*0*(\d{1,2})|ওয়ার্ড[\s_]*0*(\d{1,2})", question, re.IGNORECASE)
    if m:
        num = int(m.group(1) or m.group(2))
        wid = f"CCC_Ward_{num:02d}"
        wd = df[df["ward_id"] == wid].sort_values("date").tail(8)
        if len(wd):
            lines.append(f"Last 8 weeks for {wid}:")
            for _, r in wd.iterrows():
                lines.append(f"- {r.date.date()}: cases={int(r.dengue_cases)}, rain={r.rainfall_mm}mm, temp={r.avg_temp_c}C")
        else:
            lines.append(f"No data found for ward {num}.")
    return "\n".join(lines)


def ask_agent(question: str, history=None) -> str:
    context = build_context(question)
    contents = []
    for m in (history or [])[-6:]:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(
        text=f"DATA CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}")]))
    resp = _client().models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
    )
    return (resp.text or "").strip()


def transcribe(audio_bytes: bytes) -> str:
    """Gemini দিয়ে বাংলা/ইংরেজি কণ্ঠস্বর থেকে লেখা বানানো।"""
    resp = _client().models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            "Transcribe this speech exactly. It may be Bangla or English. Output only the transcript.",
        ],
    )
    return (resp.text or "").strip()


def speak(text: str) -> bytes:
    clean = re.sub(r"[*#_`]", "", text)
    lang = "bn" if re.search(r"[\u0980-\u09FF]", clean) else "en"
    buf = BytesIO()
    gTTS(text=clean, lang=lang).write_to_fp(buf)
    return buf.getvalue()