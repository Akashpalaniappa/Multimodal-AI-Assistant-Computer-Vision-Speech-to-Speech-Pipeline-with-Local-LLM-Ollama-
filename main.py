"""
FastAPI backend that orchestrates the full pipeline:

  User audio  ─┐
               ├─► STT ──► text ─┐
  Camera image ┘                 ├─► LLM (Ollama qwen2.5:0.5b) ──► answer text ──► TTS ──► audio
               └─► CV ──► objects ┘

Run with:  uvicorn main:app --reload --port 8000
"""

import os
import shutil
import uuid

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import UPLOAD_DIR
from modules.stt_module import transcribe_audio
from modules.cv_module import detect_objects, detections_to_context
from modules.llm_module import query_llm
from modules.tts_module import text_to_speech_file

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Multimodal AI Assistant (CV + STT + LLM + TTS)")
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Multimodal assistant backend is running"}


@app.post("/assist")
async def assist(audio: UploadFile = File(...), image: UploadFile = File(...)):
    """
    Main endpoint. Accepts one audio file (the spoken question) and one
    image file (the current camera frame), runs the full pipeline, and
    returns the transcript, detected objects, LLM answer, and a link to
    the generated speech audio.
    """
    session_id = str(uuid.uuid4())[:8]

    audio_path = os.path.join(UPLOAD_DIR, f"{session_id}_input.wav")
    image_path = os.path.join(UPLOAD_DIR, f"{session_id}_frame.jpg")
    output_audio_path = os.path.join(UPLOAD_DIR, f"{session_id}_response.mp3")

    # save uploads to disk
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # 1. Speech-to-text
    user_text = transcribe_audio(audio_path)

    # 2. Object detection
    detections = detect_objects(image_path)
    objects_context = detections_to_context(detections)

    # 3. LLM reasoning (local Ollama, qwen2.5:0.5b)
    answer_text = query_llm(user_text, objects_context)

    # 4. Text-to-speech
    text_to_speech_file(answer_text, output_audio_path)

    return JSONResponse({
        "session_id": session_id,
        "transcribed_query": user_text,
        "detected_objects": detections,
        "llm_answer": answer_text,
        "audio_response_url": f"/static/{session_id}_response.mp3",
    })


@app.get("/audio/{filename}")
def get_audio(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    return FileResponse(path, media_type="audio/mpeg")
