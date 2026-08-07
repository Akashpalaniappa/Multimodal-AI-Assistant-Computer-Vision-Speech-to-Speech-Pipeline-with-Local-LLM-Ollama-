# Multimodal AI Assistant — CV + Speech-to-Speech (Local LLM)

A fully offline multimodal assistant combining real-time object detection (YOLOv8),
speech-to-text (Whisper), a locally-hosted LLM (Ollama, `qwen2.5:0.5b`), and
text-to-speech — you speak a question about what the camera sees, and it answers
back out loud.

## Architecture

```
User voice ──► Whisper (STT) ──► text ──┐
                                          ├──► Ollama qwen2.5:0.5b ──► answer ──► pyttsx3 (TTS) ──► spoken response
Camera frame ──► YOLOv8 (CV) ──► objects ┘
```

## 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- The model already pulled:
  ```bash
  ollama pull qwen2.5:0.5b
  ```

## 2. Setup

```bash
cd multimodal-ai-assistant
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> First run will auto-download `yolov8n.pt` (~6MB) and the Whisper `base` model (~140MB).

## 3. Run

**Terminal 1 — start Ollama (if not already running as a service):**
```bash
ollama serve
```

**Terminal 2 — start the backend:**
```bash
uvicorn main:app --reload --port 8000
```

**Terminal 3 — start the demo UI:**
```bash
streamlit run app.py
```

Open the Streamlit link, upload a short voice recording and a photo/camera frame, and click
**Ask the assistant**.

## 4. Testing modules individually (useful for debugging/demos)

```bash
python modules/stt_module.py sample_audio.wav
python modules/cv_module.py sample_image.jpg
python modules/llm_module.py
python modules/tts_module.py
```

## 5. Project structure

```
multimodal-ai-assistant/
├── main.py                # FastAPI orchestrator (the /assist endpoint)
├── app.py                 # Streamlit demo UI
├── config.py               # all model names/paths in one place
├── requirements.txt
├── modules/
│   ├── stt_module.py       # Whisper speech-to-text
│   ├── cv_module.py        # YOLOv8 object detection
│   ├── llm_module.py       # Ollama qwen2.5:0.5b reasoning
│   └── tts_module.py       # pyttsx3 text-to-speech
└── temp_uploads/            # runtime scratch folder (auto-created)
```

## 6. Notes on `qwen2.5:0.5b`

This is a very small (0.5B parameter) model, so it's fast even on CPU-only laptops —
good for live demos. It's not as strong at reasoning as larger models, so for your
interview talking points, mention: "I chose a small local model to keep the whole
pipeline runnable on modest hardware without a GPU or API costs; the architecture is
model-agnostic, so swapping in `llama3.1:8b` or `qwen2.5:7b` is a one-line config change
if more reasoning quality is needed."

## 7. Resume bullet

> Built a fully offline multimodal assistant integrating real-time object detection
> (YOLOv8/OpenCV), speech-to-text (Whisper), a locally-hosted LLM (Ollama, qwen2.5:0.5b)
> for visual reasoning, and text-to-speech synthesis — enabling users to verbally query
> live camera feeds and receive spoken, context-aware responses.
