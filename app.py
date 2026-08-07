"""
Streamlit demo UI for the Multimodal AI Assistant.

Run with:  streamlit run app.py
(Make sure `uvicorn main:app --reload --port 8000` is running in another terminal first.)
"""

import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Multimodal AI Assistant", page_icon="🎙️", layout="centered")

st.title("Multimodal AI Assistant")
st.caption("Computer Vision + Speech-to-Speech, powered by a local Ollama LLM (qwen2.5:0.5b)")

st.markdown("### 1. Upload your inputs")
col1, col2 = st.columns(2)

with col1:
    audio_file = st.file_uploader("Speak your question (upload .wav/.mp3)", type=["wav", "mp3", "m4a"])

with col2:
    image_file = st.file_uploader("Camera frame (upload .jpg/.png)", type=["jpg", "jpeg", "png"])
    if image_file:
        st.image(image_file, caption="Camera frame preview", use_container_width=True)

st.markdown("### 2. Run the pipeline")

if st.button("Ask the assistant", type="primary", disabled=not (audio_file and image_file)):
    with st.spinner("Running STT -> CV -> LLM -> TTS pipeline..."):
        files = {
            "audio": (audio_file.name, audio_file.getvalue()),
            "image": (image_file.name, image_file.getvalue()),
        }
        try:
            response = requests.post(f"{BACKEND_URL}/assist", files=files, timeout=120)
            response.raise_for_status()
            data = response.json()

            st.markdown("### 3. Results")
            st.write("**You asked:**", data["transcribed_query"])
            st.write("**Detected objects:**", ", ".join(
                d["label"] for d in data["detected_objects"]
            ) or "none")
            st.write("**Assistant's answer:**", data["llm_answer"])

            audio_url = BACKEND_URL + data["audio_response_url"]
            st.audio(audio_url)

        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach backend: {e}")

st.markdown("---")
st.caption("All models run locally: Whisper (STT), YOLOv8 (CV), Ollama qwen2.5:0.5b (LLM), pyttsx3 (TTS). No API keys, no internet required after setup.")
