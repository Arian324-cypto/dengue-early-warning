import streamlit as st
from agent import ask_agent, transcribe, speak

st.set_page_config(page_title="AI Guide", page_icon="🤖")
st.title("🤖 AI Guide / এআই গাইড")
st.caption("বাংলা বা ইংরেজিতে প্রশ্ন করো, লিখে অথবা কণ্ঠে। Ask in Bangla or English, by text or voice.")

if "chat" not in st.session_state:
    st.session_state.chat = []

voice_on = st.toggle("🔊 উত্তর পড়ে শোনাও / Read answers aloud", value=True)

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["content"])

question = None

audio = None
if hasattr(st, "audio_input"):
    audio = st.audio_input("🎤 কণ্ঠে প্রশ্ন করো / Ask by voice")
if audio is not None:
    audio_id = (audio.name, audio.size)
    if st.session_state.get("last_audio") != audio_id:
        st.session_state.last_audio = audio_id
        with st.spinner("শুনছি..."):
            try:
                question = transcribe(audio.getvalue())
            except Exception as e:
                st.error(f"কণ্ঠ বুঝতে সমস্যা: {e}")

typed = st.chat_input("প্রশ্ন লেখো / Type your question")
if typed:
    question = typed

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("ভাবছি..."):
            try:
                answer = ask_agent(question, st.session_state.chat)
            except Exception as e:
                answer = f"দুঃখিত, সমস্যা হয়েছে: {e}"
        st.write(answer)
        if voice_on and not answer.startswith("দুঃখিত"):
            try:
                st.audio(speak(answer), format="audio/mp3", autoplay=True)
            except Exception:
                st.caption("(অডিও বানানো যায়নি, ইন্টারনেট দেখো)")
    st.session_state.chat.append({"role": "user", "content": question})
    st.session_state.chat.append({"role": "assistant", "content": answer})